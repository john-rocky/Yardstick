import Foundation
import Darwin

/// Process-memory sampling via Mach `task_info`.
///
/// We report **`phys_footprint`** (`TASK_VM_INFO`) — the byte count iOS charges
/// the process and the exact value **jetsam** uses to decide what to kill. It
/// counts dirty + compressed + IOKit-attributed memory, so it tracks the real
/// shipping-app ceiling far better than `resident_size` (which omits compressed
/// pages and can under-report by hundreds of MB under memory pressure). On an
/// 8 GB device the line between "fits" and "jetsam" is precisely this number,
/// which is why it is the honest memory metric for a benchmark.
public enum MemoryMonitor {
    /// Current physical footprint in megabytes — the jetsam-relevant figure.
    /// Returns 0 on failure.
    public static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / (1024 * 1024)
    }

    /// Resident size (RSS) in megabytes. Kept for reference and back-compat with
    /// pre-`phys_footprint` runs; prefer `footprintMB()` — jetsam looks at
    /// `phys_footprint`, not `resident_size`. Returns 0 on failure.
    public static func residentMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / (1024 * 1024)
    }
}

/// Records peak physical footprint across a sliding window.
///
/// Also samples `resident_size` in the same tick. The two answer different
/// questions and a benchmark needs both: `phys_footprint` is what jetsam
/// charges the process and it does **not** count clean file-backed pages, so a
/// runtime that memory-maps its weights (LiteRT-LM maps its embedding tables,
/// llama.cpp maps the whole GGUF) reads far lower than one that wires them.
/// `resident_size` counts those mapped pages while they are resident, so the gap
/// between the two reads how much of a runtime's footprint is mapped rather than
/// wired — the asymmetry that makes a raw footprint column non-comparable across
/// runtimes. Take that gap from `medianResidentMB - medianMB`, never from the
/// peaks: mapped pages fault in and out, so peak resident is page-cache noise
/// (66-281% run-to-run, measured 2026-07-26).
public actor MemorySampler {
    private(set) var peakMB: Double = 0
    private(set) var peakResidentMB: Double = 0
    private var footprintSamples: [Double] = []
    private var residentSamples: [Double] = []
    private var task: Task<Void, Never>?

    public init() {}

    /// Median of the sampled footprints — steadier than the peak when a run has a brief
    /// allocation spike.
    public var medianMB: Double { Self.median(footprintSamples) }

    /// Median of the sampled resident sizes. **Use this, not `peakResidentMB`.** Resident
    /// size counts mapped-and-resident file pages, which the kernel faults in and evicts
    /// under memory pressure, so the instantaneous peak is dominated by page-cache noise:
    /// measured 2026-07-26, peak resident swung 66% (LiteRT-LM) and 281% (MLX) across three
    /// identical runs while the footprint held to ~1%.
    public var medianResidentMB: Double { Self.median(residentSamples) }

    /// Last sample taken — the settled value at the end of the window.
    public var finalResidentMB: Double { residentSamples.last ?? 0 }

    public var sampleCount: Int { footprintSamples.count }

    public func start(intervalMS: Int = 100) {
        stop()
        footprintSamples.removeAll()
        residentSamples.removeAll()
        peakMB = MemoryMonitor.footprintMB()
        peakResidentMB = MemoryMonitor.residentMB()
        task = Task { [weak self] in
            while !Task.isCancelled {
                let footprint = MemoryMonitor.footprintMB()
                let resident = MemoryMonitor.residentMB()
                await self?.record(footprint: footprint, resident: resident)
                try? await Task.sleep(nanoseconds: UInt64(intervalMS) * 1_000_000)
            }
        }
    }

    public func stop() {
        task?.cancel()
        task = nil
    }

    private func record(footprint: Double, resident: Double) {
        if footprint > peakMB { peakMB = footprint }
        if resident > peakResidentMB { peakResidentMB = resident }
        footprintSamples.append(footprint)
        residentSamples.append(resident)
    }

    private static func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return 0 }
        let s = xs.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }
}

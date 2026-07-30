import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Battery-delta-based energy estimation.
///
/// iOS does not expose powermetrics-style energy counters to third-party apps.
/// What we *can* read is `UIDevice.current.batteryLevel` — which on iOS 27
/// is reported in **5% steps** (0.05), not the 1% the pre-2026-07-30 version of
/// this file assumed. Evidence: across 12 energy cells of the 2026-07-28/29
/// campaign every whole-run delta read exactly 0, 5 or 10 percent, never in
/// between, while sustained decode rates reproduced across rounds within 4%
/// (results/raw/2026-07-30-gemma4-e2b-protocol/README.md). A ~600 s window
/// whose true drain is ~6–8% therefore quantizes to one step or two by phase,
/// and any J/tok derived from start/end levels swings ×2 between identical runs.
///
/// The fix is the TICK-WINDOW method: poll the level at ~1 Hz during the run,
/// record every downward level transition, and measure between transitions —
/// from the first observed tick to the last, the consumed energy is exactly
/// (tick count × 5% × pack capacity), independent of where the run started
/// inside a step. `BenchmarkRunner` counts the tokens inside that window and
/// derives J/tok on a `battery-tick-window` basis; the legacy whole-run delta
/// fields are still recorded, labeled by `energySource`.
///
/// Remaining limitations:
/// - The number includes display + radios + everything else the OS is doing.
/// - Pack capacity is a per-device estimate (table below).
/// - A window needs at least 2 transitions (1 full tick, ~5–10 min of sustained
///   decode); runs that complete no window report `nil` energy rather than a
///   fabricated figure.
public actor EnergyMonitor {
    private(set) var startedAt: CFAbsoluteTime = 0
    private(set) var startBatteryLevel: Float = -1
    private(set) var startThermalState: ProcessInfo.ThermalState = .nominal

    /// Downward battery-level transitions observed by the 1 Hz poll:
    /// (timestamp, level-after-transition). An upward transition (charging)
    /// invalidates the window.
    private(set) var transitions: [(t: CFAbsoluteTime, level: Float)] = []
    private var lastPolledLevel: Float = -1
    private var chargingDetected = false
    private var pollTask: Task<Void, Never>?

    public init() {}

    public func start() {
        #if canImport(UIKit)
        UIDevice.current.isBatteryMonitoringEnabled = true
        startBatteryLevel = UIDevice.current.batteryLevel
        #else
        startBatteryLevel = -1
        #endif
        startedAt = CFAbsoluteTimeGetCurrent()
        startThermalState = ProcessInfo.processInfo.thermalState
        transitions = []
        lastPolledLevel = startBatteryLevel
        chargingDetected = false
        #if canImport(UIKit)
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                let level = await MainActor.run { UIDevice.current.batteryLevel }
                await self?.observe(level: level)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
        #endif
    }

    private func observe(level: Float) {
        guard level >= 0 else { return }
        if lastPolledLevel < 0 { lastPolledLevel = level; return }
        if level < lastPolledLevel - 0.001 {
            transitions.append((t: CFAbsoluteTimeGetCurrent(), level: level))
        } else if level > lastPolledLevel + 0.001 {
            // Charging mid-run: the tick spacing no longer measures consumption.
            chargingDetected = true
        }
        lastPolledLevel = level
    }

    public func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    /// Number of COMPLETE ticks inside the observed window (transitions - 1).
    public func tickCount() -> Int { max(0, transitions.count - 1) }

    /// The sustain loop exits early once this is true (window complete).
    public func windowComplete(targetTicks: Int = 2) -> Bool {
        !chargingDetected && tickCount() >= targetTicks
    }

    public struct TickWindow: Sendable {
        public let start: CFAbsoluteTime
        public let end: CFAbsoluteTime
        public let ticks: Int
        public let joules: Double
        public let transitionTimes: [CFAbsoluteTime]
    }

    /// The completed measurement window, or nil when fewer than 2 transitions
    /// were observed (or charging invalidated it). J = ticks × 5% × pack × 3600.
    public func tickWindow() -> TickWindow? {
        guard !chargingDetected, transitions.count >= 2 else { return nil }
        let ticks = transitions.count - 1
        let joules = Double(ticks) * 0.05 * Self.estimatedBatteryWh() * 3600
        return TickWindow(
            start: transitions[0].t,
            end: transitions[transitions.count - 1].t,
            ticks: ticks,
            joules: joules,
            transitionTimes: transitions.map { $0.t }
        )
    }

    /// Legacy whole-run snapshot: (joulesUsed, batteryDeltaPercent, durationSeconds).
    /// Kept for the fields the schema has always carried; its joules are start/end
    /// quantized and must not be used for J/tok (see the tick window above).
    public func snapshot() -> (joules: Double?, batteryDeltaPercent: Float, durationSeconds: TimeInterval) {
        stopPolling()
        let duration = CFAbsoluteTimeGetCurrent() - startedAt
        #if canImport(UIKit)
        let endLevel = UIDevice.current.batteryLevel
        guard startBatteryLevel >= 0, endLevel >= 0 else {
            return (nil, 0, duration)
        }
        let delta = startBatteryLevel - endLevel
        guard delta > 0 else {
            return (nil, 0, duration)
        }
        let packWh = Self.estimatedBatteryWh()
        let joules = Double(packWh) * Double(delta) * 3600
        return (joules, delta * 100, duration)
        #else
        return (nil, 0, duration)
        #endif
    }

    /// Estimate battery capacity in watt-hours for the current device.
    ///
    /// Numbers are pulled from public battery datasheets / iFixit teardowns
    /// for each device model. New device identifiers fall back to 12 Wh,
    /// which is roughly the iPhone-non-Pro median.
    static func estimatedBatteryWh() -> Double {
        let model = hardwareModelIdentifier()
        switch model {
        // iPhone 17 family. Apple publishes mAh, not Wh; we convert at the
        // ~3.88 V nominal implied by the iPhone 17 Pro Max teardown
        // (5,112 mAh ↔ 19.99 Wh). The Pro ships in two pack sizes — the
        // global/physical-SIM unit (3,988 mAh ≈ 15.5 Wh) and the US eSIM-only
        // unit (4,252 mAh ≈ 16.5 Wh). The measured device is the eSIM-only
        // unit; switch to 15.5 for a physical-SIM Pro.
        case "iPhone18,1": return 16.5  // iPhone 17 Pro (US eSIM-only, 4,252 mAh)
        case "iPhone18,2": return 20.0  // iPhone 17 Pro Max (5,112 mAh, teardown)
        case "iPhone18,3": return 14.3  // iPhone 17 (3,692 mAh)
        // iPhone 16 family
        case "iPhone17,1": return 13.0  // iPhone 16 Pro
        case "iPhone17,2": return 17.0  // iPhone 16 Pro Max
        case "iPhone17,3": return 13.0  // iPhone 16
        case "iPhone17,4": return 14.0  // iPhone 16 Plus
        // iPhone 15 family
        case "iPhone16,1": return 12.7  // iPhone 15 Pro
        case "iPhone16,2": return 16.7  // iPhone 15 Pro Max
        case "iPhone15,4": return 12.4  // iPhone 15
        case "iPhone15,5": return 14.0  // iPhone 15 Plus
        // iPhone 14 family
        case "iPhone15,2": return 12.4  // iPhone 14 Pro
        case "iPhone15,3": return 16.7  // iPhone 14 Pro Max
        case "iPhone14,7": return 12.7  // iPhone 14
        case "iPhone14,8": return 16.7  // iPhone 14 Plus
        default: return 12.0
        }
    }

    private static func hardwareModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        return mirror.children.reduce(into: "") { partial, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            partial.append(Character(UnicodeScalar(UInt8(value))))
        }
    }
}

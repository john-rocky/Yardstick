import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Holds the iOS background-task assertion + idle-timer override that we
/// need while a (potentially multi-GB) model is being downloaded. iOS will
/// suspend a foreground URLSession the moment the screen locks or the app
/// backgrounds, which silently freezes HF downloads at whatever byte count
/// they happened to reach. We hold both for the duration of `loadModel`.
@MainActor
final class DownloadActivityScope {
    #if canImport(UIKit)
    private var bgTask: UIBackgroundTaskIdentifier = .invalid
    #endif

    init() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = true
        bgTask = UIApplication.shared.beginBackgroundTask(withName: "ModelDownload") {
            // Expiration handler — iOS gave up. Best-effort end.
            UIApplication.shared.endBackgroundTask(self.bgTask)
            self.bgTask = .invalid
        }
        #endif
    }

    func end() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = false
        if bgTask != .invalid {
            UIApplication.shared.endBackgroundTask(bgTask)
            bgTask = .invalid
        }
        #endif
    }

    deinit {
        #if canImport(UIKit)
        // Best-effort cleanup if end() wasn't called.
        Task { @MainActor [bgTask] in
            UIApplication.shared.isIdleTimerDisabled = false
            if bgTask != .invalid {
                UIApplication.shared.endBackgroundTask(bgTask)
            }
        }
        #endif
    }
}

/// Orchestrates one benchmark run: load model (if needed), drive a task to completion,
/// gather memory/thermal samples, and produce a `BenchmarkResult`.
public actor BenchmarkRunner {
    /// Measurement-contract stamp written into every result. Bump on any change to what a
    /// recorded field means, so an inherited number can be told apart from a fresh one.
    /// `2026-07-27-agreed-protocol-r2`: harness wall-clock rates, median/resident memory, and
    /// a recorded + overridable context budget (`--context-tokens`) — the Gemma-4 spec agreed
    /// with the LiteRT team. Cells before and after this stamp are not the same measurement.
    /// `-r2` closes the decode window at the last observed chunk rather than at end-of-stream;
    /// under `-r1` a draining runtime (LiteRT-LM past its token cap) read 15.8 tok/s on the
    /// wall-clock column against 55.2 on the engine column. Only throwaway warm-up cells were
    /// ever captured under `-r1`.
    // `public` because the Mac CLI is a separate module and stamps its own rows with it. The
    // stamp exists so a result can say which harness produced it — a row that cannot carry it
    // across the module boundary defeats the purpose.
    public static let harnessStamp = "2026-07-28-agreed-protocol-r3"

    public enum Phase: Sendable {
        case idle
        case loadingModel(progress: Double)
        case generating(tokens: Int, partialOutput: String)
        case finalizing
        case done
        case failed(String)
    }

    public struct Snapshot: Sendable {
        public let phase: Phase
        public let elapsed: TimeInterval
    }

    public struct Configuration: Sendable {
        public var runtime: any LLMRuntime
        public var model: ModelInfo
        public var task: any BenchmarkTask
        public var coldRun: Bool
        /// Force the runtime's context (KV) budget to exactly this many tokens instead of
        /// deriving it from prompt + output. The LiteRT team's Gemma-4 protocol pins it at
        /// 2048; memory cells are meaningless without it, since KV pre-allocation scales
        /// with the budget. `nil` keeps the derived sizing.
        public var contextTokens: Int?

        public init(
            runtime: any LLMRuntime,
            model: ModelInfo,
            task: any BenchmarkTask,
            coldRun: Bool,
            contextTokens: Int? = nil
        ) {
            self.runtime = runtime
            self.model = model
            self.task = task
            self.coldRun = coldRun
            self.contextTokens = contextTokens
        }
    }

    private var snapshotContinuation: AsyncStream<Snapshot>.Continuation?
    private var generationTask: Task<Void, Never>?

    public init() {}

    /// Subscribe to phase changes for UI updates.
    public func snapshots() -> AsyncStream<Snapshot> {
        AsyncStream { continuation in
            self.snapshotContinuation = continuation
        }
    }

    /// Cancel the in-flight run, if any.
    public func cancel() {
        generationTask?.cancel()
    }

    public func run(_ configuration: Configuration) async throws -> BenchmarkResult {
        var device = DeviceSnapshot.capture()
        let memorySampler = MemorySampler()
        let thermalSampler = ThermalSampler()
        let energyMonitor = EnergyMonitor()

        let baselineMB = MemoryMonitor.footprintMB()
        await thermalSampler.start()

        // Size the runtime's working context to ≈ prompt + output (no-op for dynamic-KV
        // runtimes). LiteRT-LM pre-allocates a fixed KV and rejects longer prompts, so
        // long-context tasks must size it to the prompt. ~3 chars/token is a safe
        // over-estimate (lorem-ish English); over-provisioning KV is harmless, under is fatal.
        // `--context-tokens` overrides the estimate outright — the agreed protocol pins a
        // fixed budget so memory is comparable with the model card, and silently widening it
        // to fit a prompt would put us back off-protocol without saying so.
        let promptTokenEstimate = configuration.task.prompt.count / 3 + 16
        let derivedContextTokens =
            promptTokenEstimate + configuration.task.parameters.maxTokens + 512
        let contextTokens = configuration.contextTokens ?? derivedContextTokens
        if let forced = configuration.contextTokens, forced < derivedContextTokens {
            // Not fatal — a runtime may still fit, and forcing the budget is the point —
            // but it has to be visible in the log, because the failure mode is an engine
            // rejecting the prompt several minutes into a cell.
            print("YARDSTICK_WARN context_tokens_forced=\(forced) below_estimate=\(derivedContextTokens)")
            fflush(stdout)
        }

        // 1. Load model (if not already loaded).
        var loadTime: Double?
        let currentLoaded = await configuration.runtime.loadedModelId
        if currentLoaded != configuration.model.id {
            emit(.loadingModel(progress: 0))
            await configuration.runtime.prepareContext(maxContextTokens: contextTokens)
            let loadStart = CFAbsoluteTimeGetCurrent()
            // Keep iOS from auto-locking + suspending the URLSession mid-download.
            let scope = await MainActor.run { DownloadActivityScope() }
            defer { Task { @MainActor in scope.end() } }
            try await configuration.runtime.loadModel(configuration.model) { fraction in
                Task { await self.emit(.loadingModel(progress: fraction)) }
            }
            loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        }

        let memoryAfterLoad = MemoryMonitor.footprintMB()
        await memorySampler.start()
        await energyMonitor.start()

        // 2. Run generation, accumulating chunks and timing. For sustained /
        //    energy tasks the runtime is re-prompted until `sustainSeconds` of
        //    active decode elapses, so a 1%-resolution battery delta can build
        //    up. Run-once tasks (sustainSeconds == nil) execute the body once.
        let generationStart = CFAbsoluteTimeGetCurrent()
        var firstTokenAt: CFAbsoluteTime?
        var tokenWindow: [(t: CFAbsoluteTime, n: Int)] = []
        var collectedOutput = ""
        var tokenCount = 0          // streamed-chunk count (fallback token estimate)
        var reportedTokens = 0      // runtime-reported decode tokens, summed over calls
        var promptTokens = 0        // runtime-reported prompt tokens, summed over calls
        var decodeTime = 0.0        // runtime-reported decode time, summed over calls
        var promptTime = 0.0        // runtime-reported prompt time, summed over calls
        var wallPromptTime = 0.0    // harness wall-clock: call start -> first chunk
        // Harness wall-clock decode window: FIRST chunk -> LAST chunk, and the number of
        // inter-chunk gaps that span. Deliberately not "-> end of stream": MediaPipeRuntime
        // keeps draining LiteRT-LM's stream after the token cap (abandoning it wedges the
        // next in-process run for ~10 minutes), and that drain yields nothing but does hold
        // the stream open — measured 2026-07-27, it dragged a 55.2 tok/s cell to 15.8 on the
        // wall-clock column while the engine column was unaffected. Ending at the last
        // observed chunk is both honest and uniform: every arm is timed over the span in
        // which it actually delivered tokens, with no runtime cooperation required.
        var wallDecodeTime = 0.0
        var wallDecodeGaps = 0
        var lastStopReason: GenerationInfo.StopReason = .stop
        var sawInfo = false
        var capturedError: Error?

        let sustainSeconds = configuration.task.sustainSeconds
        repeat {
            let tokensBeforeCall = tokenCount
            // Wall-clock for THIS call, measured by the harness for every runtime
            // alike — the engine-reported rates below are not comparable across
            // arms because only some engines expose their own counters.
            let callStart = CFAbsoluteTimeGetCurrent()
            var callFirstTokenAt: CFAbsoluteTime?
            var callLastTokenAt: CFAbsoluteTime?
            var callChunks = 0
            let stream = configuration.runtime.generate(
                prompt: configuration.task.prompt,
                parameters: configuration.task.parameters
            )
            do {
                for try await event in stream {
                    switch event {
                    case .chunk(let text):
                        if firstTokenAt == nil {
                            firstTokenAt = CFAbsoluteTimeGetCurrent()
                        }
                        if callFirstTokenAt == nil {
                            callFirstTokenAt = CFAbsoluteTimeGetCurrent()
                        }
                        callLastTokenAt = CFAbsoluteTimeGetCurrent()
                        callChunks += 1
                        tokenCount += 1
                        // Cap the retained transcript — a 10-minute energy run
                        // would otherwise build a multi-MB string we never use
                        // (short-chat etc. keep only the first 200 chars; the
                        // quality task keeps the whole capped string for scoring).
                        if collectedOutput.count < 4000 {
                            collectedOutput.append(text)
                        }
                        let now = CFAbsoluteTimeGetCurrent()
                        tokenWindow.append((t: now, n: tokenCount))
                        emit(.generating(tokens: tokenCount, partialOutput: String(collectedOutput.prefix(80))))
                    case .info(let i):
                        sawInfo = true
                        reportedTokens += i.generationTokenCount
                        promptTokens += i.promptTokenCount
                        decodeTime += i.generateTime
                        promptTime += i.promptTime
                        lastStopReason = i.stopReason
                    }
                }
            } catch {
                capturedError = error
            }

            let callEnd = CFAbsoluteTimeGetCurrent()
            wallPromptTime += (callFirstTokenAt ?? callEnd) - callStart
            if let first = callFirstTokenAt, let last = callLastTokenAt, callChunks >= 2 {
                wallDecodeTime += last - first
                wallDecodeGaps += callChunks - 1     // n chunks span n-1 gaps
            }

            // Sustain-loop exit conditions.
            if capturedError != nil { break }
            guard let sustain = sustainSeconds else { break }          // run-once tasks
            if tokenCount == tokensBeforeCall { break }                // produced nothing → don't spin
            if CFAbsoluteTimeGetCurrent() - generationStart >= sustain { break }
            if Task.isCancelled { break }
        } while true

        // Stamp the end of generation before teardown: sampler stops, the energy
        // snapshot and the 200 ms settle sleep below are not generation time, and
        // on a short run they inflated `totalGenerationTimeSeconds` by ~8%.
        let generationEnd = CFAbsoluteTimeGetCurrent()

        emit(.finalizing)
        await memorySampler.stop()
        await thermalSampler.stop()
        let memoryPeakMB = await memorySampler.peakMB
        let memoryPeakResident = await memorySampler.peakResidentMB
        let memoryMedian = await memorySampler.medianMB
        let memoryMedianResident = await memorySampler.medianResidentMB
        let memoryFinalResident = await memorySampler.finalResidentMB
        let energy = await energyMonitor.snapshot()

        // Refresh battery fields to end-of-run: a launch-then-unplug energy run
        // begins plugged but discharges mid-run, so the start-of-run state would
        // mislabel it. (energyJoules is still the bulletproof unplugged signal —
        // it is only non-nil when the level actually dropped.)
        let endBattery = DeviceSnapshot.currentBattery()
        device.batteryState = endBattery.state
        device.batteryLevel = endBattery.level

        // Wait briefly for transient buffers to drop.
        try? await Task.sleep(nanoseconds: 200_000_000)
        let memoryAfterMB = MemoryMonitor.footprintMB()

        if let error = capturedError {
            emit(.failed(error.localizedDescription))
            throw error
        }

        let firstTokenLatency = firstTokenAt.map { ($0 - generationStart) * 1000 } ?? 0
        let totalTime = generationEnd - generationStart

        // Prefer runtime-reported counts (summed across sustain-loop calls);
        // fall back to streamed-chunk count / wall time when a runtime emits no
        // `.info` event. For run-once tasks this reduces to the single call's
        // numbers exactly.
        let genTokens = reportedTokens > 0 ? reportedTokens : tokenCount
        let effectiveDecodeTime = decodeTime > 0 ? decodeTime : max(totalTime, 0.001)
        let decodeTokS = Double(genTokens) / effectiveDecodeTime
        let promptTokS = promptTime > 0 ? Double(promptTokens) / promptTime : 0
        let stopReason = (sawInfo ? lastStopReason : .stop).rawValue
        // Both numerator and denominator are harness-observed: chunks the harness actually
        // received, over the span it received them in. Mixing a runtime-reported token count
        // with a harness-measured window would make this neither one thing nor the other.
        let decodeTokSWall: Double? = (wallDecodeTime > 0 && wallDecodeGaps > 0)
            ? Double(wallDecodeGaps) / wallDecodeTime : nil
        let promptTokSWall: Double? = (wallPromptTime > 0 && promptTokens > 0)
            ? Double(promptTokens) / wallPromptTime : nil

        // Energy figures only when a real (>0) battery delta was observed.
        let avgPowerW: Double? = {
            guard let j = energy.joules, energy.durationSeconds > 0 else { return nil }
            return j / energy.durationSeconds
        }()
        let energyJPerTok: Double? = {
            guard let j = energy.joules, genTokens > 0 else { return nil }
            return j / Double(genTokens)
        }()

        let metrics = Metrics(
            coldRun: configuration.coldRun,
            loadTimeSeconds: loadTime,
            downloadTimeSeconds: nil,
            firstTokenLatencyMS: Int(firstTokenLatency.rounded()),
            promptTokensPerSecond: promptTokS,
            decodeTokensPerSecond: decodeTokS,
            promptTokensPerSecondWallClock: promptTokSWall,
            decodeTokensPerSecondWallClock: decodeTokSWall,
            promptTokenCount: promptTokens,
            generatedTokenCount: genTokens,
            streamedChunkCount: tokenCount,
            totalGenerationTimeSeconds: totalTime,
            cancellationLatencyMS: nil,
            stopReason: stopReason,
            memoryBaselineMB: baselineMB,
            memoryAfterLoadMB: memoryAfterLoad,
            memoryPeakDuringDecodeMB: memoryPeakMB,
            memoryAfterGenerationMB: memoryAfterMB,
            memoryPeakResidentMB: memoryPeakResident > 0 ? memoryPeakResident : nil,
            memoryMedianMB: memoryMedian > 0 ? memoryMedian : nil,
            memoryMedianResidentMB: memoryMedianResident > 0 ? memoryMedianResident : nil,
            memoryFinalResidentMB: memoryFinalResident > 0 ? memoryFinalResident : nil,
            contextTokensConfigured: contextTokens,
            harnessStamp: Self.harnessStamp,
            initialThermalState: ThermalMonitor.describe(await thermalSampler.initialState),
            peakThermalState: ThermalMonitor.describe(await thermalSampler.peakState),
            finalThermalState: ThermalMonitor.describe(await thermalSampler.finalState),
            decodeRateRollingWindow: rollingDecodeRate(window: tokenWindow, windowSeconds: 5),
            interTokenLatencyP50MS: Self.percentileMS(tokenWindow: tokenWindow, percentile: 0.50),
            interTokenLatencyP95MS: Self.percentileMS(tokenWindow: tokenWindow, percentile: 0.95),
            interTokenLatencyP99MS: Self.percentileMS(tokenWindow: tokenWindow, percentile: 0.99),
            energyJoules: energy.joules,
            batteryDeltaPercent: energy.batteryDeltaPercent,
            energyJoulesPerToken: energyJPerTok,
            averagePackagePowerW: avgPowerW,
            energyMeasurementWindowSeconds: energy.joules != nil ? energy.durationSeconds : nil,
            energySource: energy.joules != nil ? "battery-1pct" : nil
        )

        emit(.done)

        return BenchmarkResult(
            device: device,
            runtime: configuration.runtime.kind.rawValue,
            model: configuration.model,
            modelRevision: HFDownloader.resolvedRevision(hfRepoId: configuration.model.hfRepoId),
            task: configuration.task.id,
            parameters: configuration.task.parameters,
            metrics: metrics,
            // Keep the full output for the quality task (it's scored for correctness +
            // degeneracy); other tasks keep a short sample to stay lean.
            outputSample: configuration.task.id == "quality"
                ? collectedOutput : String(collectedOutput.prefix(200))
        )
    }

    private func emit(_ phase: Phase) {
        snapshotContinuation?.yield(Snapshot(phase: phase, elapsed: 0))
    }

    /// Compute a percentile of the inter-token latency distribution (ms).
    /// `tokenWindow` holds one entry per emitted `.chunk` event; the gap
    /// between consecutive entries is the inter-token latency that a chat
    /// UI sees. Returns `nil` when fewer than two tokens were captured —
    /// percentiles of a one-element sample are meaningless.
    static func percentileMS(
        tokenWindow: [(t: CFAbsoluteTime, n: Int)],
        percentile: Double
    ) -> Double? {
        guard tokenWindow.count >= 2 else { return nil }
        var gaps: [Double] = []
        gaps.reserveCapacity(tokenWindow.count - 1)
        for i in 1..<tokenWindow.count {
            let dt = tokenWindow[i].t - tokenWindow[i - 1].t
            gaps.append(dt * 1000.0)
        }
        gaps.sort()
        // Nearest-rank percentile — matches what most engineers eyeball,
        // doesn't depend on a numpy-style interpolation choice.
        let rank = max(1, Int((percentile * Double(gaps.count)).rounded(.up)))
        return gaps[min(rank - 1, gaps.count - 1)]
    }

    private func rollingDecodeRate(
        window: [(t: CFAbsoluteTime, n: Int)],
        windowSeconds: Double
    ) -> [Double] {
        guard window.count >= 2 else { return [] }
        let start = window.first!.t
        let end = window.last!.t
        let stepSeconds = 1.0
        var samples: [Double] = []
        var cursor = start + windowSeconds
        while cursor <= end + 0.01 {
            let lower = cursor - windowSeconds
            let inWindow = window.filter { $0.t >= lower && $0.t <= cursor }
            if let first = inWindow.first, let last = inWindow.last, last.t > first.t {
                let dn = Double(last.n - first.n)
                let dt = last.t - first.t
                samples.append(dt > 0 ? dn / dt : 0)
            } else {
                samples.append(0)
            }
            cursor += stepSeconds
        }
        return samples
    }
}

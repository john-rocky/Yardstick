import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
#if canImport(Darwin)
import Darwin   // exit(), fflush(), stdout
#endif

@main
struct BenchmarkApp: App {
    @StateObject private var session = AppSession()
    private let autoRun = HeadlessAutoRun.specFromLaunchArgs()

    init() {
        // Gemma-4 PLE (S=1 decode graphs): the Core AI binary runtime reads
        // COREAI_CHUNK_THRESHOLD at its first framework touch, so setting it inside
        // `loadModel` (where CoreAIRuntime sets it) is too late — the engine has already
        // chosen S=8 chunked prefill and the load dies with
        //   NDArrayDescriptor.swift:139: Shape at dimension 1 of 8 is not a valid
        //   substitution for source shape 1
        // (reproduced on this device 2026-07-27). GemmaPLEDeviceBench sets it at app start
        // for exactly this reason. Gated on a gemma4 core-ai headless launch so S=1 stepping
        // never leaks into another model's prefill measurement.
        let args = CommandLine.arguments.joined(separator: " ")
        if args.contains("core-ai") && args.contains("gemma4"),
           getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
    }

    var body: some Scene {
        WindowGroup {
            if let autoRun {
                HeadlessRunnerView(spec: autoRun)
                    .environmentObject(session)
            } else {
                RootView()
                    .environmentObject(session)
            }
        }
    }
}

@MainActor
final class AppSession: ObservableObject {
    @Published var selectedRuntime: RuntimeKind = .mlxSwift
    @Published var selectedModel: ModelInfo = ModelCatalog.defaultModel
    @Published var history: [BenchmarkResult] = []

    private(set) var runtimes: [RuntimeKind: any LLMRuntime] = [:]

    /// VLM (camera) runtimes, keyed by the engine they drive — the two
    /// camera-relevant backends only: MLX on the GPU, CoreML on the ANE.
    private(set) var vlmRuntimes: [RuntimeKind: any VLMRuntime] = [:]
    static let vlmRuntimeKinds: [RuntimeKind] = [.mlxSwift, .coreMLLLM]

    init() {
        for kind in RuntimeKind.allCases {
            runtimes[kind] = makeRuntime(for: kind)
        }
        // Deployment target is iOS 18, so both VLM backends are referenceable
        // without a runtime availability gate.
        vlmRuntimes[.mlxSwift] = MLXVLMRuntime()
        vlmRuntimes[.coreMLLLM] = CoreMLVLMRuntime()
        Task { await reloadHistory() }
    }

    func runtime(for kind: RuntimeKind) -> any LLMRuntime {
        runtimes[kind]!
    }

    func vlmRuntime(for kind: RuntimeKind) -> any VLMRuntime {
        vlmRuntimes[kind]!
    }

    /// Models the currently-selected runtime can load.
    func availableModels() -> [ModelInfo] {
        runtime(for: selectedRuntime).supportedModels
    }

    /// Ensure the selected model is one the current runtime supports;
    /// if not, fall back to the runtime's first model.
    func reconcileSelectedModel() {
        let supported = availableModels()
        if !supported.contains(where: { $0.id == selectedModel.id }), let first = supported.first {
            selectedModel = first
        }
    }

    func reloadHistory() async {
        if let loaded = try? await ResultStore.shared.load() {
            await MainActor.run { self.history = loaded }
        }
    }

    func record(_ result: BenchmarkResult) async {
        _ = try? await ResultStore.shared.save(result)
        await reloadHistory()
    }

    private func makeRuntime(for kind: RuntimeKind) -> any LLMRuntime {
        switch kind {
        case .mlxSwift:
            return MLXRuntime()
        case .llamaCpp:
            return LlamaCppRuntime()
        case .mediaPipe:
            return MediaPipeRuntime()
        case .executorch:
            return ExecuTorchRuntime()
        case .coreMLLLM:
            if #available(iOS 18, *) {
                return CoreMLRuntime()
            } else {
                return UnavailableRuntime(kind: kind, reason: "Requires iOS 18.")
            }
        case .anemll:
            if #available(iOS 18, *) {
                return AnemllRuntime()
            } else {
                return UnavailableRuntime(kind: kind, reason: "Requires iOS 18.")
            }
        case .appleFM:
            if #available(iOS 26, *) {
                return AppleFMRuntime()
            } else {
                return UnavailableRuntime(
                    kind: kind,
                    reason: "Apple Foundation Models requires iOS 26 + an Apple-Intelligence-eligible device."
                )
            }
        case .coreAI:
            // CoreAIRuntime self-reports availability via canImport: when the
            // coreai-models Swift package is linked (iOS 27 build) it runs; when
            // it isn't, it returns an unavailable stub.
            return CoreAIRuntime()
        case .cactus:
            // CactusRuntime self-reports availability via canImport(cactus):
            // vendored xcframework present -> runs; absent -> unavailable stub.
            return CactusRuntime()
        }
    }
}

/// Used when a runtime can never become available at this iOS version.
public final class UnavailableRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind
    public let isAvailable: Bool = false
    public let supportedModels: [ModelInfo] = []
    private let reason: String
    public var loadedModelId: String? { nil }

    public init(kind: RuntimeKind, reason: String) {
        self.kind = kind
        self.reason = reason
    }

    public func loadModel(_ model: ModelInfo, progress: @Sendable @escaping (Double) -> Void) async throws {
        throw LLMRuntimeError.unsupported(reason)
    }

    public func unloadModel() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { c in c.finish(throwing: LLMRuntimeError.unsupported(reason)) }
    }
}

// MARK: - Headless auto-run (CLI-drivable benchmark, no UI taps)

/// Parses the launch arguments that put the app into automated benchmark mode.
///
/// An external driver launches the app with, e.g.:
///
///     xcrun devicectl device process launch --terminate-existing \
///         --device <udid> com.iosllmbenchmark.benchmarkapp -- \
///         --yardstick-autorun --runtime llama.cpp \
///         --model-id "unsloth/Qwen3.5-2B-GGUF/Q4_K_M" \
///         --task short-chat --runs 1
///
/// Each completed run is saved to `Documents/results/` via the same
/// `ResultStore` the interactive path uses, so the export / import pipeline
/// picks it up unchanged. The process prints `YARDSTICK_*` sentinel lines and
/// `exit(0)`s when finished, so the driver can detect completion.
enum HeadlessAutoRun {
    struct Spec {
        var runtime: RuntimeKind
        var modelId: String
        var taskId: String
        var runs: Int
        /// Optional override for the energy task's sustain window (seconds).
        var sustainSeconds: Double?
        /// Optional override for the energy task's per-call output cap. Lowering
        /// this (e.g. 128) keeps each generation's context short so full-attention
        /// runtimes (MLX) stay near their burst rate instead of being dragged into
        /// their long-context regime — a fairer comparison vs SWA runtimes (CoreML).
        var maxTokens: Int?
        /// `--context-tokens <n>` forces the runtime's context (KV) budget to exactly `n`
        /// instead of letting the runner derive it from prompt + output. The protocol agreed
        /// with the LiteRT team pins Gemma-4 at 2048; without it our memory cells sat at
        /// ~660-1,849 and were not comparable with the model card's 1,450 MB at 2048.
        var contextTokens: Int?
        /// `--litert-native-benchmark <prefillTokens>x<decodeTokens>` bypasses the
        /// task/runner path and calls LiteRT-LM's own `benchmark` entry point, the
        /// analogue of Cactus's `cactus_benchmark_tokens`. litert-lm only.
        var nativeBenchmark: (prefill: Int, decode: Int)?
    }

    static func specFromLaunchArgs(_ args: [String] = CommandLine.arguments) -> Spec? {
        guard args.contains("--yardstick-autorun") else { return nil }
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        // Bad args used to fall through to the interactive UI, so an external driver
        // would sit waiting for sentinel lines that never came. Fail loudly instead.
        // `--runtime` takes a RuntimeKind raw value (`mlx-swift`, not `mlx`).
        func fatal(_ reason: String) -> Never {
            print("YARDSTICK_FATAL \(reason)")
            fflush(stdout)
            exit(2)
        }
        guard let runtimeRaw = value("--runtime") else { fatal("missing --runtime") }
        guard let runtime = RuntimeKind(rawValue: runtimeRaw) else {
            fatal("unknown runtime '\(runtimeRaw)' — expected one of "
                  + RuntimeKind.allCases.map(\.rawValue).joined(separator: ", "))
        }
        guard let modelId = value("--model-id") else { fatal("missing --model-id") }
        let taskId = value("--task") ?? "short-chat"
        let runs = max(1, Int(value("--runs") ?? "1") ?? 1)
        let sustainSeconds = value("--sustain-seconds").flatMap(Double.init)
        let maxTokens = value("--max-tokens").flatMap(Int.init)
        // A typo'd context budget must not silently fall back to the derived sizing —
        // that is the exact failure the flag exists to close.
        var contextTokens: Int?
        if let raw = value("--context-tokens") {
            guard let n = Int(raw), n > 0 else { fatal("bad --context-tokens '\(raw)'") }
            contextTokens = n
        }
        let native: (prefill: Int, decode: Int)? = value("--litert-native-benchmark").flatMap {
            let parts = $0.split(separator: "x")
            guard parts.count == 2, let p = Int(parts[0]), let d = Int(parts[1]) else { return nil }
            return (p, d)
        }
        return Spec(runtime: runtime, modelId: modelId, taskId: taskId, runs: runs,
                    sustainSeconds: sustainSeconds, maxTokens: maxTokens,
                    contextTokens: contextTokens, nativeBenchmark: native)
    }
}

/// UI-less driver view. Shows a scrolling log on-device (handy when watching
/// the phone) while running the benchmark and emitting machine-readable
/// sentinel lines to stdout for the external driver.
struct HeadlessRunnerView: View {
    let spec: HeadlessAutoRun.Spec
    @EnvironmentObject private var session: AppSession
    @State private var lines: [String] = ["yardstick headless: starting…"]
    @State private var started = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.system(.footnote, design: .monospaced))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
        .task {
            guard !started else { return }
            started = true
            #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true   // don't let the screen lock mid-run
            #endif
            await runAll()
        }
    }

    @MainActor
    private func log(_ line: String) {
        print(line)
        fflush(stdout)
        lines.append(line)
    }

    private func finish(_ code: Int32) async {
        // Give stdout + the on-disk JSON write a moment to flush before tearing down.
        try? await Task.sleep(nanoseconds: 600_000_000)
        exit(code)
    }

    private func runAll() async {
        let runtime = session.runtime(for: spec.runtime)
        guard runtime.isAvailable else {
            await log("YARDSTICK_FATAL runtime=\(spec.runtime.rawValue) not_available")
            await finish(2)
            return
        }
        guard let model = runtime.supportedModels.first(where: { $0.id == spec.modelId }) else {
            await log("YARDSTICK_FATAL model=\(spec.modelId) not_in_catalog runtime=\(spec.runtime.rawValue)")
            await finish(3)
            return
        }
        if let native = spec.nativeBenchmark {
            await runNativeBenchmark(runtime: runtime, model: model, spec: native,
                                     contextTokensValue: spec.contextTokens)
            return
        }
        guard var task = BenchmarkTaskCatalog.task(for: spec.taskId) else {
            await log("YARDSTICK_FATAL task=\(spec.taskId) unknown")
            await finish(4)
            return
        }
        // Energy task: allow the driver to tune the sustain window + per-call
        // output cap per run. A small --max-tokens (e.g. 128) keeps the context
        // short so full-attention runtimes (MLX) stay near their burst rate
        // instead of decaying in their long-context regime — a fair comparison
        // against SWA runtimes (CoreML-LLM) whose context is bounded by design.
        if spec.taskId == "energy" {
            // Tick-window era (r4): the sustain is an upper BOUND, not the window — the
            // runner exits early once two 5%-step battery transitions complete the
            // measurement window. 1800 s gives a slow-draining arm room to finish one.
            task = EnergyTask(
                sustainSeconds: spec.sustainSeconds ?? 1800,
                maxTokens: spec.maxTokens ?? 2048
            )
            // Energy cells must not start throttled (agreed protocol: "not started
            // throttled"). The 07-19 headline cells both began `fair` and read ~25%
            // below their own nominal decode rate. Battery-temperature telemetry is a
            // lagging long-window average, so the gate is the app's own thermal state
            // at launch — the same field the result JSON records — and it runs BEFORE
            // the model load spends 600 s of battery on an unrankable cell. The driver
            // sees the sentinel + exit code, cools, and retries the cell.
            let thermal = ProcessInfo.processInfo.thermalState
            if thermal != .nominal {
                await log("YARDSTICK_THERMAL_DEFER task=energy initial_thermal=\(ThermalMonitor.describe(thermal))")
                await finish(7)
                return
            }
        }

        let sustainNote = task.sustainSeconds.map { " sustain_s=\(Int($0))" } ?? ""
        let contextNote = spec.contextTokens.map { " context_tokens=\($0)" } ?? ""
        await log("YARDSTICK_BEGIN runtime=\(spec.runtime.rawValue) model=\(model.id) task=\(task.id) runs=\(spec.runs)\(sustainNote)\(contextNote)")
        for i in 1...spec.runs {
            let runner = BenchmarkRunner()
            let cold = (await runtime.loadedModelId) != model.id
            do {
                let result = try await runner.run(
                    .init(runtime: runtime, model: model, task: task, coldRun: cold,
                          contextTokens: spec.contextTokens)
                )
                _ = try? await ResultStore.shared.save(result)
                // The console transcript is the raw record a driver keeps, so it carries the
                // fields a cell can't be audited without: the context it actually ran at, the
                // harness wall-clock rates (the only cross-arm-comparable ones), and the median
                // footprint/resident pair rather than the page-cache-noisy peaks.
                await log(String(
                    format: "YARDSTICK_RUN_OK run=%d cold=%d decode_tok_s=%.2f decode_tok_s_wall=%.2f ttft_ms=%d prefill_tok_s=%.1f prefill_tok_s_wall=%.1f prompt_tokens=%d peak_mb=%.0f median_mb=%.0f median_resident_mb=%.0f ctx=%d tokens=%d thermal_initial=%@ thermal_final=%@ harness=%@",
                    i, cold ? 1 : 0,
                    result.metrics.decodeTokensPerSecond,
                    result.metrics.decodeTokensPerSecondWallClock ?? 0,
                    result.metrics.firstTokenLatencyMS,
                    result.metrics.promptTokensPerSecond,
                    result.metrics.promptTokensPerSecondWallClock ?? 0,
                    result.metrics.promptTokenCount,
                    result.metrics.memoryPeakDuringDecodeMB,
                    result.metrics.memoryMedianMB ?? 0,
                    result.metrics.memoryMedianResidentMB ?? 0,
                    result.metrics.contextTokensConfigured ?? 0,
                    result.metrics.generatedTokenCount,
                    result.metrics.initialThermalState,
                    result.metrics.finalThermalState,
                    result.metrics.harnessStamp ?? "?"
                ))
                // Energy is only present on a real, unplugged battery drop.
                if let joules = result.metrics.energyJoules {
                    await log(String(
                        format: "YARDSTICK_ENERGY run=%d state=%@ battery_delta_pct=%.1f joules=%.1f avg_w=%.2f j_per_tok=%.4f window_s=%.0f tokens=%d",
                        i, result.device.batteryState,
                        result.metrics.batteryDeltaPercent,
                        joules,
                        result.metrics.averagePackagePowerW ?? 0,
                        result.metrics.energyJoulesPerToken ?? 0,
                        result.metrics.energyMeasurementWindowSeconds ?? 0,
                        result.metrics.generatedTokenCount
                    ))
                } else {
                    await log(String(
                        format: "YARDSTICK_ENERGY run=%d state=%@ battery_delta_pct=%.1f joules=nil (run too short, or plugged in/charging)",
                        i, result.device.batteryState, result.metrics.batteryDeltaPercent
                    ))
                }
            } catch {
                await log("YARDSTICK_RUN_FAIL run=\(i) error=\(error.localizedDescription)")
            }
        }
        await log("YARDSTICK_ALL_DONE")
        await finish(0)
    }

    /// Calls LiteRT-LM's own `benchmark` entry point instead of driving a task through
    /// `BenchmarkRunner`. Reports the engine's internal prefill/decode counters, which
    /// is the only way to get a LiteRT prefill tok/s at a fixed prompt length: the
    /// streaming path leaves the counters unfinalized whenever output is capped.
    private func runNativeBenchmark(
        runtime: any LLMRuntime, model: ModelInfo, spec: (prefill: Int, decode: Int),
        contextTokensValue: Int?
    ) async {
        #if canImport(LiteRTLM)
        guard let litert = runtime as? MediaPipeRuntime else {
            await log("YARDSTICK_FATAL native_benchmark requires runtime=litert-lm")
            await finish(5)
            return
        }
        // The context length is an independent axis of the agreed protocol (2048 for Gemma-4),
        // but LiteRT-LM's own `benchmark()` ties it to prefill: without --context-tokens this
        // runs at max(prefill, decode) + 32 — 1056 at 1024x256. Both paths are allowed; only
        // one is on-protocol, so the log says which was taken instead of leaving it implicit.
        let contextTokens = contextTokensValue ?? (max(spec.prefill, spec.decode) + 32)
        if contextTokensValue == nil {
            await log("YARDSTICK_WARN native_benchmark context_tokens=\(contextTokens) (litert stock default; agreed protocol is 2048 for gemma-4 — pass --context-tokens)")
        }
        await log("YARDSTICK_BEGIN native_benchmark model=\(model.id) prefill=\(spec.prefill) decode=\(spec.decode) context_tokens=\(contextTokens)")

        // Sample memory *during* the benchmark. The published 92 MB deep-context cell was a
        // single footprint read taken after `benchmark()` returned and released the engine,
        // which is not the same quantity as the in-run peaks it was tabulated against.
        let sampler = MemorySampler()
        await sampler.start()
        do {
            let info = try await litert.nativeBenchmark(
                model, prefillTokens: spec.prefill, decodeTokens: spec.decode,
                maxNumTokens: contextTokens)
            await sampler.stop()
            let peakMB = await sampler.peakMB
            let medianMB = await sampler.medianMB
            let medianResidentMB = await sampler.medianResidentMB
            let samples = await sampler.sampleCount
            await log(String(
                format: "YARDSTICK_NATIVE_OK prefill_tokens=%d prefill_tok_s=%.2f decode_tokens=%d decode_tok_s=%.2f ttft_ms=%.1f init_s=%.2f context_tokens=%d peak_mb=%.0f median_mb=%.0f median_resident_mb=%.0f samples=%d teardown_footprint_mb=%.0f harness=%@",
                info.prefillTokenCount, info.prefillTokensPerSecond,
                info.decodeTokenCount, info.decodeTokensPerSecond,
                info.timeToFirstTokenSeconds * 1000, info.initTimeSeconds,
                contextTokens, peakMB, medianMB, medianResidentMB, samples,
                MemoryMonitor.footprintMB(),
                BenchmarkRunner.harnessStamp
            ))
            await finish(0)
        } catch {
            await sampler.stop()
            await log("YARDSTICK_FATAL native_benchmark_failed \(error.localizedDescription)")
            await finish(6)
        }
        #else
        await log("YARDSTICK_FATAL native_benchmark unavailable (LiteRTLM not linked)")
        await finish(5)
        #endif
    }
}

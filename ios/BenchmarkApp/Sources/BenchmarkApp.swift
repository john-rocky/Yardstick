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
        // Gemma-4 PLE (S=1 decode graphs): the Core AI binary runtime appears to read
        // COREAI_CHUNK_THRESHOLD at its first framework touch, so setting it inside
        // loadModel (as CoreAIRuntime does) can be too late — the engine then substitutes
        // S=8 chunked prefill and NDArrayDescriptor fatals. GemmaPLEDeviceBench sets it at
        // app start for exactly this reason. Gate it on a gemma4 core-ai headless launch so
        // S=1 stepping never leaks into other models' prefill measurements.
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
    }

    static func specFromLaunchArgs(_ args: [String] = CommandLine.arguments) -> Spec? {
        guard args.contains("--yardstick-autorun") else { return nil }
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        guard let runtimeRaw = value("--runtime"),
              let runtime = RuntimeKind(rawValue: runtimeRaw),
              let modelId = value("--model-id") else { return nil }
        let taskId = value("--task") ?? "short-chat"
        let runs = max(1, Int(value("--runs") ?? "1") ?? 1)
        let sustainSeconds = value("--sustain-seconds").flatMap(Double.init)
        let maxTokens = value("--max-tokens").flatMap(Int.init)
        return Spec(runtime: runtime, modelId: modelId, taskId: taskId, runs: runs,
                    sustainSeconds: sustainSeconds, maxTokens: maxTokens)
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
            task = EnergyTask(
                sustainSeconds: spec.sustainSeconds ?? 600,
                maxTokens: spec.maxTokens ?? 2048
            )
        }

        let sustainNote = task.sustainSeconds.map { " sustain_s=\(Int($0))" } ?? ""
        await log("YARDSTICK_BEGIN runtime=\(spec.runtime.rawValue) model=\(model.id) task=\(task.id) runs=\(spec.runs)\(sustainNote)")
        for i in 1...spec.runs {
            let runner = BenchmarkRunner()
            let cold = (await runtime.loadedModelId) != model.id
            do {
                let result = try await runner.run(
                    .init(runtime: runtime, model: model, task: task, coldRun: cold)
                )
                _ = try? await ResultStore.shared.save(result)
                await log(String(
                    format: "YARDSTICK_RUN_OK run=%d cold=%d decode_tok_s=%.2f ttft_ms=%d prefill_tok_s=%.1f prompt_tokens=%d peak_mb=%.0f tokens=%d",
                    i, cold ? 1 : 0,
                    result.metrics.decodeTokensPerSecond,
                    result.metrics.firstTokenLatencyMS,
                    result.metrics.promptTokensPerSecond,
                    result.metrics.promptTokenCount,
                    result.metrics.memoryPeakDuringDecodeMB,
                    result.metrics.generatedTokenCount
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
}

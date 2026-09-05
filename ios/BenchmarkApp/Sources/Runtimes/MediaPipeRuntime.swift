#if canImport(LiteRTLM)
import Foundation
// LiteRT-LM 0.12.0 is pre-concurrency: its `Engine` is an actor but `Conversation`
// isn't Sendable, so awaiting `engine.createConversation(...)` from this actor trips
// strict actor-isolation. `@preconcurrency` treats LiteRTLM types with the legacy
// rules (the Conversation is only ever used within this one runGenerate scope).
@preconcurrency import LiteRTLM

/// LiteRT-LM adapter — wraps Google's official `google-ai-edge/LiteRT-LM`
/// Swift API (`import LiteRTLM`, ≥ 0.12.0).
///
/// Loads `.litertlm` bundles (e.g. `litert-community/gemma-4-E2B-it-litert-lm`)
/// and drives generation through `Engine` → `Conversation.sendMessageStream`.
/// The decode backend is Metal GPU (`.gpu`); pass `.cpu()` to compare CPU.
///
/// This replaces the deprecated MediaPipe Tasks GenAI 0.10.x (`.task`) path,
/// which was iOS-only and could not read Gemma 4. The 0.12.0 package ships an
/// `ios-arm64` + `macos-arm64` xcframework, so it runs on both the iOS app
/// and the macOS `yardstick` CLI.
///
/// The runtime kind is still `.mediaPipe` (raw value `"litert-lm"`) for
/// source/JSONL stability.
public actor MediaPipeRuntime: LLMRuntime {
    public let kind: RuntimeKind = .mediaPipe
    public let isAvailable: Bool = true
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.liteRTLM

    public private(set) var loadedModelId: String?

    private var engine: Engine?
    private var modelPath: String?
    // LiteRT-LM pre-allocates KV for `maxNumTokens`, so an over-provisioned context
    // inflates peak memory for short tasks. Sized to the task's output budget by
    // `prepareContext` (called before loadModel); default covers a long-ish chat.
    private var contextBudget = 2048

    public init() {}

    /// Size the LiteRT-LM context (KV pre-allocation = `maxNumTokens`) to ≈ prompt + output.
    /// LiteRT-LM rejects any prompt longer than `maxNumTokens` (`INVALID_ARGUMENT: Input token
    /// ids are too long`), so the runner passes prompt+output here; a 128-token short-chat still
    /// gets a small KV, while a long-context task gets one sized to its prompt. (Still not
    /// directly comparable to a dynamic-KV runtime like MLX — disclosed — but no longer wrong.)
    public func prepareContext(maxContextTokens: Int) {
        contextBudget = max(256, maxContextTokens)
    }

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }

        // Enable LiteRT-LM's benchmark counters so generation can report *real*
        // tokenizer token counts + tok/s (via Conversation.getBenchmarkInfo),
        // instead of estimating from streamed chunks. Must opt in first.
        ExperimentalFlags.optIntoExperimentalAPIs()
        ExperimentalFlags.enableBenchmark = true

        let snapshot = try await HFDownloader.snapshot(for: model, runtime: kind, progress: progress)
        let modelFile = try locateModelFile(in: snapshot, expected: model.primaryFile)

        do {
            // maxNumTokens caps the working context (and the KV LiteRT-LM pre-allocates);
            // sized to the task via `prepareContext` so short tasks don't reserve a huge KV.
            // The per-run output budget is enforced separately in `runGenerate`.
            // The vision executor is NOT loaded unless a backend is named for it.
            // Omitting it still gives a working Engine and a working Conversation —
            // the failure only appears on the first message carrying an image
            // ("Vision executor should not be null, please TryLoadingVisionExecutor()
            // first"), so a load-time smoke test does not catch it. Opt in explicitly
            // for VLM bundles; leaving it nil for text bundles keeps them unchanged.
            let wantsVision = ProcessInfo.processInfo.arguments.contains("--litert-vision")
            let backend: Backend =
                ProcessInfo.processInfo.arguments.contains("--litert-cpu") ? .cpu() : .gpu
            // `--litert-max-tokens N` pins maxNumTokens regardless of the task-derived
            // contextBudget. The Metal delegate compiles shaders against the KV that
            // maxNumTokens pre-allocates; a value that disagrees with the model's
            // exported cache produces DUS shape rejections + partial delegation
            // (S26 2026-08-23: explicit 1024 turned 104/542 half4x float4 aborts into
            // 542/542 PASS on the same file), so the gate must control it explicitly.
            let launchArgs = ProcessInfo.processInfo.arguments
            var effectiveMaxTokens = contextBudget
            if let flagIdx = launchArgs.firstIndex(of: "--litert-max-tokens"),
               flagIdx + 1 < launchArgs.count,
               let forced = Int(launchArgs[flagIdx + 1]) {
                effectiveMaxTokens = forced
                print("[MediaPipeRuntime] maxNumTokens forced to \(forced) via --litert-max-tokens")
            }
            // `--litert-speculative` opts a bundle with an MTP drafter section into
            // speculative decoding. Engine init throws on bundles without one, so it
            // stays a launch-arg opt-in rather than a default.
            if launchArgs.contains("--litert-speculative") {
                ExperimentalFlags.enableSpeculativeDecoding = true
                print("[MediaPipeRuntime] speculative decoding enabled via --litert-speculative")
            }
            // Diagnostic knob for the MTP leg: the iOS default activation type is
            // fp16; the drafter/verify contract wants fp32 activations, and fp16
            // rounding of the hidden handoff is a candidate cause for acceptance
            // collapse on iOS. Forces fp32 like the desktop CLI default.
            if launchArgs.contains("--litert-fp32-act") {
                ExperimentalFlags.forceFloat32Activations = true
                print("[MediaPipeRuntime] fp32 activations forced via --litert-fp32-act")
            }
            let config = try EngineConfig(
                modelPath: modelFile.path,
                backend: backend,
                visionBackend: wantsVision ? backend : nil,
                maxNumTokens: effectiveMaxTokens,
                cacheDir: NSTemporaryDirectory()
            )
            let engine = Engine(engineConfig: config)
            try await engine.initialize()
            self.engine = engine
            self.modelPath = modelFile.path
            self.loadedModelId = model.id
        } catch {
            throw LLMRuntimeError.loadFailed(error.localizedDescription)
        }
    }

    public func unloadModel() async {
        engine = nil
        modelPath = nil
        loadedModelId = nil
    }

    /// LiteRT-LM's own benchmark entry point (`LiteRTLM.benchmark`), which prefills
    /// `prefillTokens` and decodes `decodeTokens` and reports the engine's internal
    /// counters. This is the direct analogue of Cactus's `cactus_benchmark_tokens`,
    /// so the two engines can be compared on their own instruments rather than
    /// through this app's streaming path. It is a card-reconciliation instrument
    /// (a forced prefill with no prompt), not a cross-runtime one — the cross-arm
    /// prefill row comes from the task path in `runGenerate`.
    ///
    /// Stands apart from `loadModel`: `benchmark` builds its own `Engine`, so call it
    /// on a runtime with no model loaded.
    ///
    /// LiteRT-LM's own `BenchmarkInfo` is not `Sendable`, so its fields are copied out
    /// here rather than crossing the actor boundary.
    public struct NativeBenchmarkInfo: Sendable {
        public let prefillTokenCount: Int
        public let decodeTokenCount: Int
        public let prefillTokensPerSecond: Double
        public let decodeTokensPerSecond: Double
        public let timeToFirstTokenSeconds: Double
        public let initTimeSeconds: Double
    }

    /// - Parameter maxNumTokens: the engine's context (KV) budget. Passed **explicitly and
    ///   without relying on a default** on purpose: `maxNumTokens` is a local patch to the
    ///   vendored (and gitignored) LiteRT-LM Swift package, so re-vendoring the stock package
    ///   must break the build here rather than silently reverting the context to the stock
    ///   `max(prefill, decode) + 32` — 1056 at 1024x256, where the agreed protocol says 2048.
    ///   See `ios/BenchmarkApp/Vendored-patches/README.md`.
    public func nativeBenchmark(
        _ model: ModelInfo,
        prefillTokens: Int,
        decodeTokens: Int,
        maxNumTokens: Int
    ) async throws -> NativeBenchmarkInfo {
        let snapshot = try await HFDownloader.snapshot(for: model, runtime: kind, progress: { _ in })
        let modelFile = try locateModelFile(in: snapshot, expected: model.primaryFile)
        let info = try await LiteRTLM.benchmark(
            modelPath: modelFile.path,
            backend: .gpu,
            prefillTokens: prefillTokens,
            decodeTokens: decodeTokens,
            maxNumTokens: maxNumTokens,
            cacheDir: NSTemporaryDirectory()
        )
        return NativeBenchmarkInfo(
            prefillTokenCount: info.lastPrefillTokenCount,
            decodeTokenCount: info.lastDecodeTokenCount,
            prefillTokensPerSecond: info.lastPrefillTokensPerSecond,
            decodeTokensPerSecond: info.lastDecodeTokensPerSecond,
            timeToFirstTokenSeconds: info.timeToFirstTokenInSecond,
            initTimeSeconds: info.initTimeInSecond
        )
    }

    public nonisolated func generate(
        prompt: String,
        imagePath: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGenerate(prompt: prompt, imagePath: imagePath,
                                               parameters: parameters, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public nonisolated func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGenerate(prompt: prompt, parameters: parameters, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGenerate(
        prompt: String,
        imagePath: String? = nil,
        parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async throws {
        guard let engine else { throw LLMRuntimeError.modelNotLoaded }

        // Fresh conversation per run so each measurement is independent (no
        // KV reuse across runs). No system message is injected, to keep the
        // prompt identical to the other runtimes' single-prompt path.
        // --litert-temp / --litert-topk override the task's sampler for models
        // whose greedy decode collapses (RL tunes, e.g. Nanbeige4.2: official
        // recipe temp 0.6 / top-k 20).
        let launchArgs = ProcessInfo.processInfo.arguments
        func flagValue(_ flag: String) -> String? {
            guard let i = launchArgs.firstIndex(of: flag), i + 1 < launchArgs.count else { return nil }
            return launchArgs[i + 1]
        }
        let temp = flagValue("--litert-temp").flatMap(Float.init) ?? parameters.temperature
        let topK = flagValue("--litert-topk").flatMap(Int.init) ?? 40
        let sampler = try SamplerConfig(
            topK: topK,
            topP: parameters.topP,
            temperature: temp
        )
        // --litert-thinking turns on ThinkingConfig (v0.15.0+ API; the bundle's
        // template carries the enable_thinking branch, so no model change).
        // Thinking chunks stream on a separate channel; the per-turn decode
        // counters cover thinking + answer tokens, which is what the
        // decode-under-thinking cell measures.
        let thinking = launchArgs.contains("--litert-thinking")
        // v0.12.0 is an early-preview API: `ConversationConfig`'s `systemMessage`
        // is optional here; if a future release makes it required, pass
        // `Message("")` or fall back to `engine.createConversation()`.
        let conversation = try await engine.createConversation(
            with: ConversationConfig(
                samplerConfig: sampler,
                thinkingConfig: thinking
                    ? ThinkingConfig(enableThinking: true, thinkingTokenBudget: -1)
                    : nil
            )
        )

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var firstTokenAt: CFAbsoluteTime?
        var tokenCount = 0
        var capped = false
        var cappedAt: CFAbsoluteTime? = nil

        // Text goes FIRST and the image last: on prompts with named fields the image
        // is the value of the field it follows, and putting it ahead of the
        // instructions measurably degrades the answer.
        let message = imagePath.map { Message(of: .text(prompt), .imageFile($0)) }
            ?? Message(prompt)

        for try await chunk in conversation.sendMessageStream(message) {
            try Task.checkCancellation()
            if capped {
                // Cap reached: DRAIN the stream silently instead of breaking out.
                // Abandoning the stream leaves its callback task queued and the NEXT
                // in-process run blocks ~10 min on `callback_thread_pool
                // DEADLINE_EXCEEDED` (observed on iPhone and Mac; `conversation
                // .cancel()` does not release it either). The engine stops at its
                // context bound (~maxNumTokens) within a few seconds; nothing drained
                // here is yielded or timed — measurement ended at `cappedAt`.
                continue
            }
            if firstTokenAt == nil { firstTokenAt = CFAbsoluteTimeGetCurrent() }
            let text = chunk.toString
            if !text.isEmpty {
                continuation.yield(.chunk(text))
                tokenCount += 1  // chunk tally ≈ tokens
            }
            // Cap output at the task's token budget so LiteRT-LM is measured over the
            // SAME number of generated tokens as every other runtime (fairness rule 1:
            // same maxTokens). 0.12/0.13's streaming API has no per-call cap, so at the
            // cap the DECODE measure falls back to chunk-count + wall-clock (the
            // CoreML-style measure) — the engine's own decode turn runs on to the
            // context bound and covers more tokens than the cap. Prefill counters stay
            // valid either way; see the `bench` block below.
            // (`conversation.cancel()` is NOT used here: it surfaces as a thrown
            // `CANCELLED` stream error and fails the run. Pure draining is enough —
            // the engine stops at its context bound within a few seconds.)
            if tokenCount >= parameters.maxTokens {
                capped = true
                cappedAt = CFAbsoluteTimeGetCurrent()
            }
        }

        let end = cappedAt ?? CFAbsoluteTimeGetCurrent()
        let wallPrefill = (firstTokenAt ?? end) - prefillStart
        let wallGenerate = max(end - (firstTokenAt ?? prefillStart), 0.001)

        // Prefer LiteRT-LM's own per-turn counters: real tokenizer token counts
        // and tok/s. We back-derive promptTime / generateTime so GenerationInfo's
        // computed tok/s == LiteRT's reported rates exactly. Fall back to
        // chunk-count + wall-clock if the benchmark info is unavailable (e.g. flag
        // not honored on this build).
        //
        // Prefill and decode are NOT symmetric under a cap. The prefill turn is
        // recorded at prefill completion — runtime/core/tasks.cc calls
        // TimePrefillTurnEnd() right after executor.Prefill() returns, before decode
        // starts — so where decode is later stopped cannot change it; the counters
        // are valid for every run. Decode's turn, by contrast, only reflects OUR
        // measurement window on a natural EOS finish: a capped run drains to the
        // engine's context bound, so its decode turn covers more tokens than the
        // cap. Capped → decode falls back to chunk-count + wall-clock.
        let bench = try? conversation.getBenchmarkInfo()
        let decodeBench = capped ? nil : bench
        let decodeTokens = (decodeBench.map { $0.lastDecodeTokenCount } ?? 0) > 0
            ? decodeBench!.lastDecodeTokenCount : tokenCount
        let promptTokens = bench?.lastPrefillTokenCount ?? 0
        let generateTime: TimeInterval = {
            if let b = decodeBench, b.lastDecodeTokensPerSecond > 0, b.lastDecodeTokenCount > 0 {
                return Double(b.lastDecodeTokenCount) / b.lastDecodeTokensPerSecond
            }
            return wallGenerate
        }()
        let promptTime: TimeInterval = {
            if let b = bench, b.lastPrefillTokensPerSecond > 0, b.lastPrefillTokenCount > 0 {
                return Double(b.lastPrefillTokenCount) / b.lastPrefillTokensPerSecond
            }
            return wallPrefill
        }()

        continuation.yield(.info(GenerationInfo(
            promptTokenCount: promptTokens,
            generationTokenCount: decodeTokens,
            promptTime: promptTime,
            generateTime: generateTime,
            stopReason: capped ? .length : .stop
        )))
        continuation.finish()
    }

    /// Resolve the `.litertlm` file inside a downloaded snapshot.
    /// Prefers the explicit `primaryFile`, then a standard (non-web, non-NPU)
    /// `.litertlm`, then any `.litertlm`, then a legacy `.task`.
    private func locateModelFile(in dir: URL, expected: String) throws -> URL {
        if !expected.isEmpty {
            let direct = dir.appendingPathComponent(expected)
            if FileManager.default.fileExists(atPath: direct.path) { return direct }
        }
        let contents = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        let litertlm = contents.filter { $0.pathExtension == "litertlm" }
        let isPlatformVariant: (URL) -> Bool = { url in
            let n = url.lastPathComponent.lowercased()
            return n.contains("-web") || n.contains("intel") || n.contains("qualcomm")
        }
        if let standard = litertlm.first(where: { !isPlatformVariant($0) }) { return standard }
        if let any = litertlm.first { return any }
        if let task = contents.first(where: { $0.pathExtension == "task" }) { return task }
        throw LLMRuntimeError.loadFailed("No .litertlm or .task file found in \(dir.path)")
    }
}
#else
import Foundation

/// Compile-time-disabled LiteRT-LM runtime. Add the `LiteRTLM` product from
/// `https://github.com/google-ai-edge/LiteRT-LM` (≥ 0.12.0) to enable it.
/// See `runtimes/litert-lm.md` for the integration steps.
public final class MediaPipeRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .mediaPipe
    public let isAvailable: Bool = false
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.liteRTLM
    public var loadedModelId: String? { nil }

    public init() {}

    public func loadModel(_ model: ModelInfo, progress: @Sendable @escaping (Double) -> Void) async throws {
        throw LLMRuntimeError.unsupported("LiteRTLM package not added to the project. See runtimes/litert-lm.md.")
    }

    public func unloadModel() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { c in
            c.finish(throwing: LLMRuntimeError.unsupported("LiteRTLM package not added — see runtimes/litert-lm.md."))
        }
    }
}
#endif

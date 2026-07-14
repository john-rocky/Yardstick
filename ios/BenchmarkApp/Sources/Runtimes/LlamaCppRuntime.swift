#if canImport(llama)
import Foundation
import llama
import HuggingFace

/// llama.cpp adapter.
///
/// Uses the vendored `llama.xcframework` from
/// https://github.com/ggml-org/llama.cpp/releases (downloaded by
/// `scripts/bootstrap.sh`). The Swift wrapper pattern follows
/// `examples/llama.swiftui/llama.cpp.swift/LibLlama.swift` upstream,
/// adapted to the runtime-agnostic `LLMRuntime` surface.
public actor LlamaCppRuntime: LLMRuntime {
    public let kind: RuntimeKind = .llamaCpp
    public let isAvailable: Bool = true
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.llamaCpp

    public private(set) var loadedModelId: String?

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var batch: llama_batch?

    private static let backendInit: Void = {
        llama_backend_init()
    }()

    public init() {
        _ = Self.backendInit
    }

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }

        let snapshot = try await HFDownloader.snapshot(for: model, runtime: kind, progress: progress)
        let ggufPath = snapshot.appendingPathComponent(model.primaryFile).path
        guard FileManager.default.fileExists(atPath: ggufPath) else {
            throw LLMRuntimeError.loadFailed("GGUF file not found at \(ggufPath)")
        }

        await unloadModel()

        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #endif
        guard let m = llama_model_load_from_file(ggufPath, modelParams) else {
            throw LLMRuntimeError.loadFailed("llama_model_load_from_file returned NULL")
        }
        self.model = m
        self.vocab = llama_model_get_vocab(m)

        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = 4096
        // n_batch defaults to 512. A prompt longer than that is submitted as one
        // `llama_batch` and comes back with logits that sample EOG immediately — the
        // 1K-token parity prompt generated zero tokens until this was raised.
        ctxParams.n_batch = 2048
        ctxParams.n_ubatch = 512
        ctxParams.n_threads = Int32(nThreads)
        ctxParams.n_threads_batch = Int32(nThreads)

        guard let c = llama_init_from_model(m, ctxParams) else {
            llama_model_free(m)
            self.model = nil
            throw LLMRuntimeError.loadFailed("llama_init_from_model returned NULL")
        }
        self.context = c

        self.batch = llama_batch_init(2048, 0, 1)

        self.loadedModelId = model.id
    }

    /// Build a sampler chain for one generation. `temperature == 0` means greedy, which is
    /// what every benchmark task asks for; the other adapters honour it, so this one must too.
    private func makeSampler(_ parameters: GenerationParameters) throws -> UnsafeMutablePointer<llama_sampler> {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
            throw LLMRuntimeError.generationFailed("llama_sampler_chain_init returned NULL")
        }
        if parameters.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(parameters.topP, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(parameters.temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32(parameters.seed ?? 1234)))
        }
        return chain
    }

    public func unloadModel() async {
        if var b = batch { llama_batch_free(b); _ = withUnsafeMutablePointer(to: &b) { _ in } }
        if let c = context { llama_free(c) }
        if let m = model { llama_model_free(m) }
        batch = nil
        context = nil
        model = nil
        vocab = nil
        loadedModelId = nil
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
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func runGenerate(
        prompt: String,
        parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async throws {
        guard let context, let vocab, let model else {
            throw LLMRuntimeError.modelNotLoaded
        }

        // Every call re-submits the prompt at positions 0…n-1. Without clearing the KV
        // cache first, a second call in the same process collides with the first turn's
        // entries and `llama_decode` fails — which is what a multi-run task (`--runs 4`)
        // does. Matches the fresh-conversation-per-run behaviour of the other adapters.
        llama_memory_clear(llama_get_memory(context), true)

        // Build the sampler per call so the task's parameters are honoured. The chain used
        // to be fixed at temp 0.7 / top-p 0.9, which silently ignored a greedy task and made
        // llama.cpp the only sampled runtime in an otherwise greedy comparison.
        let sampler = try makeSampler(parameters)
        defer { llama_sampler_free(sampler) }

        // Apply the model's chat template so llama.cpp matches the other adapters (rule 1);
        // parseSpecial so the template's <start_of_turn> / <|im_start|> tokenize as special
        // tokens, not literal text. VERIFY ON DEVICE: if the template also prepends <bos>,
        // drop addBOS here to avoid a double-BOS. llama.cpp rows need re-measure after this.
        let templated = applyChatTemplate(prompt)
        let promptTokens = tokenize(text: templated ?? prompt, addBOS: true,
                                    parseSpecial: templated != nil)

        var b = batch ?? llama_batch_init(Int32(max(promptTokens.count, 2048)), 0, 1)
        // Reset batch.
        b.n_tokens = 0

        // Submit prompt.
        for (i, tok) in promptTokens.enumerated() {
            llama_batch_add(&b, tok, Int32(i), [0], false)
        }
        b.logits[Int(b.n_tokens) - 1] = 1
        self.batch = b

        let prefillStart = CFAbsoluteTimeGetCurrent()
        guard llama_decode(context, b) == 0 else {
            throw LLMRuntimeError.generationFailed("llama_decode (prefill) failed")
        }
        let prefillEnd = CFAbsoluteTimeGetCurrent()

        var nCur = Int32(promptTokens.count)
        var generatedCount = 0
        var stopReason: GenerationInfo.StopReason = .length

        let decodeStart = CFAbsoluteTimeGetCurrent()
        while generatedCount < parameters.maxTokens {
            try Task.checkCancellation()

            let newToken = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, newToken) {
                stopReason = .stop
                break
            }

            let piece = tokenToPiece(token: newToken)
            if !piece.isEmpty {
                continuation.yield(.chunk(piece))
            }

            generatedCount += 1
            nCur += 1

            // Re-prepare batch for one new token.
            b.n_tokens = 0
            llama_batch_add(&b, newToken, nCur - 1, [0], true)
            self.batch = b
            guard llama_decode(context, b) == 0 else {
                stopReason = .error
                break
            }
        }
        let decodeEnd = CFAbsoluteTimeGetCurrent()

        continuation.yield(.info(GenerationInfo(
            promptTokenCount: promptTokens.count,
            generationTokenCount: generatedCount,
            promptTime: prefillEnd - prefillStart,
            generateTime: max(decodeEnd - decodeStart, 0.001),
            stopReason: stopReason
        )))
        continuation.finish()
    }

    // MARK: - Tokenization helpers (mirror upstream LibLlama.swift)

    private func tokenize(text: String, addBOS: Bool, parseSpecial: Bool = false) -> [llama_token] {
        guard let vocab else { return [] }
        let utf8Count = text.utf8.count
        let n = utf8Count + (addBOS ? 1 : 0) + 1
        let buf = UnsafeMutablePointer<llama_token>.allocate(capacity: n)
        defer { buf.deallocate() }
        let count = llama_tokenize(vocab, text, Int32(utf8Count), buf, Int32(n), addBOS, parseSpecial)
        guard count > 0 else { return [] }
        return (0 ..< Int(count)).map { buf[$0] }
    }

    /// Wrap the bare prompt in the model's own chat template (read from the GGUF metadata)
    /// so llama.cpp sees the same input as every other adapter — MLX / CoreML / Core AI /
    /// ANEMLL / LiteRT all apply their template via the tokenizer; llama.cpp must do it
    /// explicitly (fairness rule 1). Returns nil if the model has no template or it isn't in
    /// `llama_chat_apply_template`'s built-in set (Gemma / Qwen / Llama are), so the caller
    /// falls back to the bare prompt rather than failing.
    private func applyChatTemplate(_ userPrompt: String) -> String? {
        guard let model, let tmpl = llama_model_chat_template(model, nil) else { return nil }
        return "user".withCString { rolePtr in
            userPrompt.withCString { contentPtr -> String? in
                var msg = llama_chat_message(role: rolePtr, content: contentPtr)
                var buf = [CChar](repeating: 0, count: max(userPrompt.utf8.count * 2 + 256, 512))
                var n = llama_chat_apply_template(tmpl, &msg, 1, true, &buf, Int32(buf.count))
                if n > Int32(buf.count) {                // buffer too small — grow once
                    buf = [CChar](repeating: 0, count: Int(n))
                    n = llama_chat_apply_template(tmpl, &msg, 1, true, &buf, Int32(buf.count))
                }
                guard n > 0 else { return nil }
                let take = max(0, min(Int(n), buf.count))
                return String(cString: Array(buf[0..<take]) + [0])
            }
        }
    }

    private func tokenToPiece(token: llama_token) -> String {
        guard let vocab else { return "" }
        var buffer = [CChar](repeating: 0, count: 64)
        let nWritten = buffer.withUnsafeMutableBufferPointer { ptr in
            llama_token_to_piece(vocab, token, ptr.baseAddress, Int32(ptr.count), 0, false)
        }
        guard nWritten > 0 else { return "" }
        var bytes = Array(buffer.prefix(Int(nWritten)))
        bytes.append(0) // null-terminate
        return bytes.withUnsafeBufferPointer { p -> String in
            String(cString: p.baseAddress!)
        }
    }
}

// MARK: - Tiny llama_batch convenience (mirrors upstream LibLlama.swift)

private func llama_batch_add(
    _ batch: inout llama_batch,
    _ id: llama_token,
    _ pos: llama_pos,
    _ seqIds: [llama_seq_id],
    _ logits: Bool
) {
    let n = Int(batch.n_tokens)
    batch.token[n] = id
    batch.pos[n] = pos
    batch.n_seq_id[n] = Int32(seqIds.count)
    for (i, sid) in seqIds.enumerated() {
        batch.seq_id[n]?[i] = sid
    }
    batch.logits[n] = logits ? 1 : 0
    batch.n_tokens = Int32(n + 1)
}
#else
import Foundation

/// Compile-time-disabled llama.cpp runtime. Add `llama.xcframework` to the
/// project (run `scripts/bootstrap.sh`) to enable the real implementation.
public final class LlamaCppRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .llamaCpp
    public let isAvailable: Bool = false
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.llamaCpp
    public var loadedModelId: String? { nil }

    public init() {}

    public func loadModel(_ model: ModelInfo, progress: @Sendable @escaping (Double) -> Void) async throws {
        throw LLMRuntimeError.unsupported("llama.xcframework not added — run scripts/bootstrap.sh.")
    }

    public func unloadModel() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { c in
            c.finish(throwing: LLMRuntimeError.unsupported("llama.xcframework not added — run scripts/bootstrap.sh."))
        }
    }
}
#endif

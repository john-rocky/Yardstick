#if canImport(cactus)
import Foundation
import cactus

/// Cactus adapter — wraps `cactus-compute/cactus`'s C FFI (`cactus_engine.h`,
/// vendored as `Vendored/cactus-ios.xcframework`; `scripts/bootstrap.sh` builds it
/// from source with `bash apple/build.sh`).
///
/// Loads Cactus CQ bundles (directories with a `config.txt`, e.g. the unzipped
/// `Cactus-Compute/gemma-4-E2B-it` `gemma-4-e2b-it-cq4.zip`) and drives generation
/// through `cactus_init` → `cactus_complete` with a token callback — the same entry
/// points Cactus's own CLI/benchmark use. The engine applies its own chat template
/// (verified byte-identical to the HF reference for Gemma-4) and picks its default
/// backend, which is Metal GPU on Apple silicon (their published numbers' path).
///
/// Cloud handoff and telemetry are forced OFF for every call: a hybrid answer would
/// measure the cloud model, not the on-device build (fairness rule: on-device only).
public actor CactusRuntime: LLMRuntime {
    public let kind: RuntimeKind = .cactus
    public let isAvailable: Bool = true
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.cactus

    public private(set) var loadedModelId: String?

    private var model: UnsafeMutableRawPointer?

    public init() {}

    public func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard supportedModels.contains(where: { $0.id == model.id }) else {
            throw LLMRuntimeError.modelNotInCatalog(model.id)
        }
        let snapshot = try await HFDownloader.snapshot(for: model, runtime: kind, progress: progress)
        let bundleDir = try locateBundle(in: snapshot, expected: model.primaryFile)

        guard let handle = cactus_init(bundleDir.path, nil, false) else {
            throw LLMRuntimeError.loadFailed("cactus_init failed for \(bundleDir.lastPathComponent)")
        }
        self.model = handle
        self.loadedModelId = model.id
    }

    public func unloadModel() async {
        if let model { cactus_destroy(model) }
        model = nil
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
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGenerate(
        prompt: String,
        parameters: GenerationParameters,
        continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    ) async throws {
        guard let model else { throw LLMRuntimeError.modelNotLoaded }

        // Fresh KV per run so each measurement is independent (matches the other arms:
        // LiteRT gets a fresh conversation, llama.cpp clears its cache per call).
        cactus_reset(model)

        let messages = try jsonString([["role": "user", "content": prompt]])
        var options: [String: Any] = [
            "max_tokens": parameters.maxTokens,
            "temperature": parameters.temperature,
            "top_p": parameters.topP,
            // On-device measurement only: a cloud-handoff answer would score the
            // cloud model, and telemetry adds network traffic mid-run.
            "auto_handoff": false,
            "telemetry_enabled": false,
        ]
        if parameters.temperature == 0 {
            options["top_k"] = 1  // Cactus's greedy path (mirrors cactus_benchmark_tokens)
        }
        let optionsJSON = try jsonString(options)

        let handle = ModelHandle(model)
        let response: [String: Any] = try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let sink = TokenSink { text in
                    if !text.isEmpty { continuation.yield(.chunk(text)) }
                }
                let sinkPtr = Unmanaged.passRetained(sink).toOpaque()
                defer { Unmanaged<TokenSink>.fromOpaque(sinkPtr).release() }

                var buffer = [CChar](repeating: 0, count: 1 << 20)
                let rc = cactus_complete(
                    handle.raw, messages, &buffer, buffer.count,
                    optionsJSON, nil, tokenTrampoline, sinkPtr, nil, 0
                )
                let text = String(cString: buffer)
                guard rc >= 0,
                      let data = text.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else {
                    cont.resume(throwing: LLMRuntimeError.generationFailed(
                        rc < 0 ? "cactus_complete rc=\(rc): \(text.prefix(300))" : "unparseable response"))
                    return
                }
                cont.resume(returning: obj)
            }
        }

        if (response["success"] as? Bool) != true {
            let err = response["error"] as? String ?? "unknown engine error"
            throw LLMRuntimeError.generationFailed(err)
        }

        // Exact engine-reported counters (their own construct_response_json):
        // prefill_tokens / decode_tokens are tokenizer-true counts; TTFT and total
        // are wall-clock. promptTime = TTFT keeps prefill tok/s the honest
        // promptTokens/TTFT (their prefill_tps ≈ this to <0.01%, measured 2026-07-10).
        let ttftMS = (response["time_to_first_token_ms"] as? Double) ?? 0
        let totalMS = (response["total_time_ms"] as? Double) ?? 0
        let promptTokens = (response["prefill_tokens"] as? Int) ?? 0
        let decodeTokens = (response["decode_tokens"] as? Int) ?? 0

        continuation.yield(.info(GenerationInfo(
            promptTokenCount: promptTokens,
            generationTokenCount: decodeTokens,
            promptTime: ttftMS / 1000.0,
            generateTime: max((totalMS - ttftMS) / 1000.0, 0.001),
            stopReason: decodeTokens >= parameters.maxTokens ? .length : .stop
        )))
        continuation.finish()
    }

    /// Resolve the CQ bundle directory (the dir holding `config.txt`) inside a
    /// downloaded/sideloaded snapshot. Prefers `primaryFile`, then the snapshot root
    /// itself, then the first subdirectory that looks like a bundle.
    private func locateBundle(in dir: URL, expected: String) throws -> URL {
        let fm = FileManager.default
        let isBundle: (URL) -> Bool = { fm.fileExists(atPath: $0.appendingPathComponent("config.txt").path) }
        if !expected.isEmpty {
            let direct = dir.appendingPathComponent(expected)
            if isBundle(direct) { return direct }
        }
        if isBundle(dir) { return dir }
        if let contents = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil),
           let sub = contents.first(where: isBundle) {
            return sub
        }
        throw LLMRuntimeError.loadFailed(
            "No Cactus bundle (config.txt) under \(dir.path). Sideload the unzipped CQ bundle "
            + "(e.g. gemma-4-e2b-it-cq4/) into this directory.")
    }

    private func jsonString(_ obj: Any) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: obj)
        return String(data: data, encoding: .utf8)!
    }
}

/// Sendable wrapper so the raw C handle can cross into the blocking dispatch closure.
private struct ModelHandle: @unchecked Sendable {
    let raw: UnsafeMutableRawPointer
    init(_ raw: UnsafeMutableRawPointer) { self.raw = raw }
}

private final class TokenSink {
    let onToken: (String) -> Void
    init(_ onToken: @escaping (String) -> Void) { self.onToken = onToken }
}

/// C token callback → TokenSink. `user_data` is an unretained TokenSink pointer that
/// outlives the `cactus_complete` call (released in the dispatch closure's defer).
private let tokenTrampoline: cactus_token_callback = { token, _, userData in
    guard let token, let userData else { return }
    let sink = Unmanaged<TokenSink>.fromOpaque(userData).takeUnretainedValue()
    sink.onToken(String(cString: token))
}

#else
import Foundation

/// Compile-time-disabled Cactus runtime. Run `scripts/bootstrap.sh` to build and
/// vendor `cactus-ios.xcframework`, then regenerate the project (xcodegen).
public final class CactusRuntime: LLMRuntime, @unchecked Sendable {
    public let kind: RuntimeKind = .cactus
    public let isAvailable: Bool = false
    public nonisolated let supportedModels: [ModelInfo] = ModelCatalog.cactus
    public var loadedModelId: String? { nil }

    public init() {}

    public func loadModel(_ model: ModelInfo, progress: @Sendable @escaping (Double) -> Void) async throws {
        throw LLMRuntimeError.unsupported("cactus xcframework not vendored — run scripts/bootstrap.sh.")
    }

    public func unloadModel() async {}

    public func generate(prompt: String, parameters: GenerationParameters) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { c in
            c.finish(throwing: LLMRuntimeError.unsupported("cactus xcframework not vendored — run scripts/bootstrap.sh."))
        }
    }
}
#endif

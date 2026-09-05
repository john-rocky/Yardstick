import Foundation

/// Adapter protocol that every benchmarked runtime conforms to.
///
/// One implementation per runtime (MLX, llama.cpp, CoreML-LLM, …).
/// The benchmark runner only sees this surface, never the underlying SDK.
public protocol LLMRuntime: AnyObject, Sendable {
    var kind: RuntimeKind { get }
    var loadedModelId: String? { get async }

    /// Whether this runtime can be invoked at runtime.
    /// `false` means the underlying framework is not present in the build
    /// (e.g. MediaPipe XCFramework not added yet).
    var isAvailable: Bool { get }

    /// Models this runtime is able to load. The UI's model picker filters by this.
    var supportedModels: [ModelInfo] { get }

    /// Download (if needed) and load weights into memory.
    /// `progress` reports 0.0–1.0 for the download phase only; load itself is synchronous from the caller's view.
    func loadModel(
        _ model: ModelInfo,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws

    /// Release model weights and any associated GPU/ANE state.
    func unloadModel() async

    /// Stream generation events for a single prompt. The stream finishes after a `.info` event or on cancellation.
    func generate(
        prompt: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error>

    /// Same, with one image attached — the vision-language path. The image is passed
    /// as an absolute file path because that is what the LiteRT-LM Swift API takes
    /// (`Content.imageFile`), and it avoids copying decoded pixels through the harness.
    ///
    /// Default: ignore the image and fall back to text-only, so runtimes without a
    /// vision path stay compilable. A runtime that *claims* vision must override this
    /// — a silent text-only fallback would score a vision task as if it had run.
    func generate(
        prompt: String,
        imagePath: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error>

    /// Optional hint, called just before `loadModel`: size the runtime's working context
    /// (e.g. KV pre-allocation) to roughly this many tokens (≈ prompt + output budget), so it
    /// neither rejects a long prompt nor over-reserves for a short one. Runtimes that allocate
    /// KV dynamically ignore it (default no-op). LiteRT-LM pre-allocates KV to a fixed
    /// `maxNumTokens` and *rejects* any prompt longer than it, so long-context tasks must size
    /// this to the prompt, not just the output.
    func prepareContext(maxContextTokens: Int) async
}

public extension LLMRuntime {
    /// Text-only fallback. Runtimes with a vision path override this.
    func generate(
        prompt: String,
        imagePath: String,
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        generate(prompt: prompt, parameters: parameters)
    }

    func prepareContext(maxContextTokens: Int) async {}
}

public enum GenerationEvent: Sendable {
    case chunk(String)
    case info(GenerationInfo)
}

public struct GenerationInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let promptTime: TimeInterval
    public let generateTime: TimeInterval
    public let stopReason: StopReason

    public enum StopReason: String, Sendable {
        case stop
        case length
        case cancelled
        case error
    }

    public var promptTokensPerSecond: Double {
        promptTime > 0 ? Double(promptTokenCount) / promptTime : 0
    }

    public var tokensPerSecond: Double {
        generateTime > 0 ? Double(generationTokenCount) / generateTime : 0
    }
}

public enum LLMRuntimeError: LocalizedError {
    case unsupported(String)
    case modelNotLoaded
    case modelNotInCatalog(String)
    case loadFailed(String)
    case generationFailed(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let what): return "Unsupported: \(what)"
        case .modelNotLoaded: return "No model is loaded."
        case .modelNotInCatalog(let id): return "Model \(id) is not in this runtime's catalog."
        case .loadFailed(let why): return "Model load failed: \(why)"
        case .generationFailed(let why): return "Generation failed: \(why)"
        case .downloadFailed(let why): return "Model download failed: \(why)"
        }
    }
}

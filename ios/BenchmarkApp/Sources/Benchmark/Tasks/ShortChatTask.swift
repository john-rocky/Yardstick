import Foundation

public struct ShortChatTask: BenchmarkTask {
    public let id = "short-chat"
    public let title = "Short chat"
    public let summary = "128-token assistant response. Measures TTFT and steady-state decode."

    public let prompt = "Explain what on-device AI means in simple terms."

    public let parameters: GenerationParameters

    /// `maxTokens` is overridable so the thinking cell can measure decode over a
    /// thinking-length generation (~500+ tok) instead of capping mid-thought at 128.
    public init(maxTokens: Int = 128) {
        parameters = GenerationParameters(maxTokens: maxTokens, temperature: 0.0, topP: 1.0)
    }
}

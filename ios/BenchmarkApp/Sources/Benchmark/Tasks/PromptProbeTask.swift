import Foundation

/// One-off prompt probe for quality triage: runs a single caller-supplied prompt,
/// greedy (temp 0), and keeps the full output in the result JSON. The prompt
/// arrives base64-encoded over the launch args so shell quoting cannot mangle it.
///
/// Exists to answer "does a combined-prompt miss reproduce as a single question?"
/// on-device, where the desktop CLI cannot run. Headless-only: it is not in
/// `BenchmarkTaskCatalog` (there is no meaningful default prompt), the autorun
/// dispatch constructs it directly from `--prompt-b64`.
public struct PromptProbeTask: BenchmarkTask {
    public let id = "prompt-probe"
    public let title = "Prompt probe"
    public let summary = "Single caller-supplied prompt, greedy. Full output kept for post-hoc scoring."
    public let parameters: GenerationParameters
    public let prompt: String

    public init(prompt: String, maxTokens: Int = 256) {
        self.prompt = prompt
        self.parameters = GenerationParameters(maxTokens: maxTokens, temperature: 0.0, topP: 1.0)
    }
}

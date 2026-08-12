import Foundation

/// Long-context prefill + decode-at-depth. Parameterised by an approximate prompt
/// length so a sweep (2K → 8K → 32K) can show how prefill throughput and decode
/// rate hold up as the KV cache grows — the "does decode stay flat under long
/// context" question. The nominal length is approximate; the *actual* prompt token
/// count is recorded as `promptTokenCount` in the JSONL, and the report plots the
/// real count, so the `-8k` / `-32k` ids are just labels.
public struct LongContextTask: BenchmarkTask {
    public let id: String
    public let title: String
    public let summary: String
    public let parameters: GenerationParameters
    private let blocks: Int

    /// One filler block ≈ this many tokens (a ~40-word lorem paragraph + an index
    /// tag). Used only to turn a nominal target into a block count; ground truth is
    /// the runtime-recorded `promptTokenCount`.
    private static let tokensPerBlock = 55

    /// - Parameters:
    ///   - id: stable task id (e.g. `long-context`, `long-context-8k`).
    ///   - targetTokens: approximate prompt length to build toward.
    ///   - maxTokens: decode budget after prefill (held equal across the sweep so
    ///     decode-rate-at-depth is comparable).
    /// When true the prompt asks for a long enumerated answer instead of one sentence.
    ///
    /// The one-sentence tail was fine for prefill but silently broke decode-at-depth: every
    /// arm hit EOS after 15-33 tokens (measured 2026-07-26), so "decode tok/s at p=1024" was
    /// a rate computed over a couple of dozen tokens and swung 30-38% run to run. A task that
    /// wants a decode rate has to keep the model generating for the whole budget.
    private let forceLongOutput: Bool

    public init(id: String = "long-context", targetTokens: Int = 2048, maxTokens: Int = 128,
                forceLongOutput: Bool = false) {
        self.id = id
        self.blocks = max(1, targetTokens / Self.tokensPerBlock)
        self.forceLongOutput = forceLongOutput
        let approx = targetTokens >= 1000 ? "~\(targetTokens / 1000)K" : "~\(targetTokens)"
        self.title = "Long-context prefill (\(approx) tok)"
        self.summary = "\(approx)-token prompt, \(maxTokens)-token output. "
            + "Prefill throughput (TTFT, prefill tok/s) and decode rate at depth."
        self.parameters = GenerationParameters(maxTokens: maxTokens, temperature: 0.0, topP: 1.0)
    }

    public var prompt: String {
        let lorem = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus bibendum velit non augue \
        ultricies, a vestibulum ipsum porttitor. Sed at nulla a justo viverra dictum. Vivamus blandit \
        velit at lectus pulvinar pellentesque. Mauris dictum massa ut nisi tristique consequat.
        """
        var pieces: [String] = []
        pieces.reserveCapacity(blocks + 1)
        for index in 0 ..< blocks {
            pieces.append("[\(index)] \(lorem)")
        }
        if forceLongOutput {
            // Enumerated, self-continuing instruction: the model keeps producing until the
            // token budget stops it, which is what makes decode-at-depth measurable.
            pieces.append("\n\nUsing the passage above as context, list 25 distinct things "
                + "on-device AI lets a phone do that a cloud model cannot. Number every item "
                + "and give each one two full sentences of explanation. Do not stop early.")
        } else {
            pieces.append("\n\nFinish with one sentence: what on-device AI lets a phone do that a cloud model cannot.")
        }
        return pieces.joined(separator: "\n")
    }
}

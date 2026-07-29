import Foundation

/// Mirrors the LLM protocol of `cactus benchmark` (Cactus's `cactus-engine/tests/test_benchmark.cpp`)
/// so its published numbers can be compared against the runtimes in this repo:
/// a ~1K-token prompt, exactly 100 decoded tokens, greedy.
///
/// Drive it with `--runs 4` and drop the first row: Cactus runs one warmup pass and
/// means the following three, keeping the model loaded and resetting only the KV cache
/// between passes. `BenchmarkRunner` reloads a model only when the id changes, so runs
/// 2–4 of a single autorun invocation are warm in the same way.
///
/// Two metric definitions differ from the rest of this repo's tasks and are recomputed
/// from the JSONL by `scripts/cactus_parity_report.py` rather than read off `metrics`:
///
/// - Prefill is reported as `promptTokenCount / TTFT`. That is the only prefill rate
///   every engine here can produce, and it is what Cactus emits as `ttft_prompt_tps`.
///   Cactus's headline `prefill_tps` is a narrower, compute-only figure — it divides by
///   `cache_prime_ms - cache_state_copy_ms`, excluding the KV-cache handoff between its
///   separate prefill and decode graphs (`cactus-engine/src/complete.cpp`).
/// - Decode is reported as `(generatedTokens - 1) / (totalTime - TTFT)`, matching
///   Cactus's `decode_tps`, which drops the first token with the prefill.
public struct CactusParityTask: BenchmarkTask {
    public let id = "cactus-parity"
    public let title = "Cactus parity (1K prefill / 100 decode)"
    public let summary = "~1K-token prompt, 100-token output, greedy. "
        + "Matches `cactus benchmark`'s LLM protocol for a like-for-like prefill/decode comparison."

    public let parameters = GenerationParameters(maxTokens: 100, temperature: 0.0, topP: 1.0)

    /// 17 blocks lands the Gemma-4 chat-templated prompt at 1022 tokens (measured with
    /// `mlx-community/gemma-4-e2b-it-4bit`'s tokenizer). Cactus truncates to exactly 1000.
    /// Both fall in the same 8×128 prefill-chunk bucket (1024 slots), so neither side pays
    /// for an extra chunk. Throughput normalises by the token count each runtime reports,
    /// so the 22-token difference does not bias the rate.
    private static let blocks = 17

    public init() {}

    public var prompt: String {
        let lorem = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus bibendum velit non augue \
        ultricies, a vestibulum ipsum porttitor. Sed at nulla a justo viverra dictum. Vivamus blandit \
        velit at lectus pulvinar pellentesque. Mauris dictum massa ut nisi tristique consequat.
        """
        var pieces: [String] = []
        pieces.reserveCapacity(Self.blocks + 1)
        for index in 0 ..< Self.blocks {
            pieces.append("[\(index)] \(lorem)")
        }
        pieces.append("\n\nFinish with one sentence: what on-device AI lets a phone do that a cloud model cannot.")
        return pieces.joined(separator: "\n")
    }
}

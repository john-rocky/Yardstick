# 2026-08-04 LiteRT-LM v0.15.0 iPhone protocol re-capture + thinking cells (Gemma-4-E2B)

**Build:** BenchmarkApp with `Vendored/LiteRT-LM` at tag v0.15.0 (binaryTarget =
release `CLiteRTLM.xcframework.zip`, checksum d6ccf6b5… — verified by local shasum
against the tag's Package.swift). App additions this session: `--litert-thinking`
launch flag (ThinkingConfig(enableThinking: true, budget −1) into ConversationConfig),
short-chat honors `--max-tokens`, harnessStamp records `+litert-thinking`.
**Conditions:** iPhone 17 Pro (iOS 27b3), UNPLUGGED 75–80%, serial launches with
120–180 s cooldowns, one run per launch. Driver: `scripts/bench_litert_0150_iphone.sh`,
chronology `drive.log`, raw JSON in `pulled/`.

## Results (median of warm launches, run 1 discarded by position)

| cell | v0.15.0 (this run) | published (v0.13.1/0.14-era) | Δ |
|---|---|---|---|
| short-chat decode tok/s | **62.1** [61.9..62.3, spread 0.6%] | 61.1 warm | +1.6% |
| short-chat TTFT | 61 ms | — | |
| short-chat memory (median footprint) | 488 MB | 497 MB | −1.8% |
| short-chat memory (median resident) | ~900 MB | 849 MB | **+6% — watch item** |
| long-context p=1,081 prefill tok/s | **3,653** [1,991..3,731] | 3,513 | +4.0% |
| decode at depth | 58.1 (engine) / 60.9 (wall) | ~58.5 | ≈0 (flat through depth holds) |
| deep-context footprint | 751 MB | 732 MB | +2.6% |
| deep-context resident | 996 MB | 849 MB* | **+17% — watch item** (*if the published resident was the deep cell) |

## Thinking cells (n=4, `+litert-thinking`, max-tokens 1400, deterministic greedy)

Every run generated the identical 1,267-token thinking turn (greedy determinism ✓).

| metric | value |
|---|---|
| total turn (1,267 tok incl. thought) | ~42.8 s |
| first *visible* token (thought is filtered from the stream) | **13.1 s** [13.0..13.6] |
| engine-average decode over the turn | **29.6 tok/s** [29.2..30.1] — vs 62.1 thinking-OFF = **2.1× slowdown** |
| memory footprint | 479 MB (≈ OFF) |

Reading: on-device, thinking mode COSTS decode throughput on LiteRT (2.1× on the
engine-average), unlike depth (flat). Softer than Core AI's 34.2→11.7 (2.9×), and the
memory cost is nil. The campaign expectation "LiteRT stays flat" was about
decode-at-depth, which held; thinking-mode throughput is a different axis and is not flat.

## Caveats / instrumentation notes

- Thermal: the first 5 short-chat launches started `serious`/`fair` (residual heat from
  plugged smoke runs), last 3 `fair`; long-context and thinking cells reached `nominal`
  starts by mid-run. The in-app gate records but does not defer. Evidence it did not
  bite: decode spread across serious→fair starts is 0.6%.
- `decodeTokensPerSecondWallClock` under thinking (42.8) is NOT interpretable as
  answer-phase decode: chunk-count semantics are ambiguous when the thought channel is
  filtered. Quote the engine counter and the direct wall-clock times only.
- Resident-memory (+6% short / +17% deep vs published) needs a same-session 0.14 A/B
  before it is called a 0.15 regression — cross-session resident comparisons violate the
  session-variance rule; footprint (the jetsam-relevant number) is flat-to-lower.
- Two plugged smoke cells (06:28 / 06:31 UTC, coldRun, charging) are in `pulled/` for
  completeness — NOT protocol cells; excluded from every median above.
- Energy: not re-captured — the trigger (large speed movement) is not met on device
  (+1.6% decode).

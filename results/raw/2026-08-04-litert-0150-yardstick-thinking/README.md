# 2026-08-04 LiteRT-LM v0.15.0 yardstick-surface GSM8K (Gemma-4-E2B, M4 Max GPU)

**Instrument:** litert-mac-verify (xcodebuild yardstick — the exact surface of the published
v0.13.1 86.0 row) rebuilt against the v0.15.0 GitHub release xcframework
(CLiteRTLM_mac.zip, checksum d23cf189… verified against the tag's Package.swift and by
local shasum). Fork bump: ~/code/swift-litert-lm `liteRTLMVersion v0.13.1 → v0.15.0`,
official Swift wrapper re-vendored at the tag, fork-local maxNumImages patch re-applied
(upstream 0.15 already carries visualTokenBudget + the error case). `--thinking` added to
litert-mac-verify → `LiteRTChat(thinking:)` → `ConversationConfig(thinkingConfig:)`.
Protocol: pinned n=100, greedy (topK1/topP1.0/temp0), canonical extractor
(scripts/parity_gsm8k.py, run_litertlm now threads --thinking).

## Results

| cell | ctx (total) | GSM8K | note |
|---|---|---:|---|
| OFF | 2048 (the agreed-protocol budget) | **88.0** | the version re-check of the published 86.0 row: +2.0 build delta v0.13.1→v0.15.0, same surface, same protocol (`run.log`) |
| ON | 2048 | 88.0 (±0) | **honest footnote, not the thinking row**: at 2048 *total* context the thinking trace (~800 tok median) eats the budget — 2 runs died at the ceiling (pred=None q9/q22), 4 fresh wrongs vs the 4096 run. Thinking needs context headroom to pay off |
| OFF | 4096 (headroom; budget semantics comparable to the other thinking-table arms) | **89.0** | wrong set identical to the pip-0.15 pair's OFF (11 questions, same list) (`run-ctx4096.log`) |
| ON | 4096 | **92.0 (+3.0)** | = Core AI thinking / bf16 anchor. Wrong set identical to the pip pair's ON (8 questions incl. runaway-thinking q9/q44 counted wrong) |

## Cross-validation

Two independent implementations of released 0.15.0 — the PyPI dylib (Python API,
`gsm8k_litert-gemma4-e2b-pip015gpu-*`) and this Swift xcframework surface — produce
**identical per-question outcomes** at matched context semantics (OFF 89.0 / ON 92.0,
same wrong lists). The July main-CPU reference pair (87.0→90.0) shows the same +3.
Chain across builds/surfaces: 86.0 (v0.13.1 Swift) → 88.0 (v0.15.0 Swift, ctx2048) →
89.0/92.0 (v0.15.0 both surfaces, ctx4096 pair).

Reports: `results/quality/gsm8k_litert-gemma4-e2b-v0150-{off,thinking}.json` (ctx2048),
`...-ctx4096-{off,thinking}.json`. Raw logs: `run.log`, `run-ctx4096.log` here.

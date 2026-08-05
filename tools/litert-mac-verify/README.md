# litert-mac-verify — the Mac GSM8K quality instrument ("yardstick" surface)

The Swift/xcframework surface that produced the published LiteRT GSM8K rows. It loads a
`.litertlm` through `LiteRTChat` (GPU; the same wrapper surface as the iOS app's engine
route) and prints the generation as a single greppable `OUTPUT: [...]` line, which
`scripts/parity_gsm8k.py --which int4` scores with the canonical extractor.

## Build

```bash
swift build --package-path tools/litert-mac-verify -c release
```

`scripts/parity_gsm8k.py` looks for the binary at
`tools/litert-mac-verify/.build/release/litert-mac-verify`; set `LITERT_MAC_VERIFY` to
point elsewhere. The engine dependency is `john-rocky/swift-litert-lm` pinned by exact
revision (tag `yardstick-2026-08-04`): official LiteRT-LM v0.15.0 release binaryTargets
(checksum-verified zips from google-ai-edge/LiteRT-LM releases) + the official Swift
wrapper re-vendored at that tag (`ThinkingConfig`) + the fork's `LiteRTChat(thinking:)`
passthrough. Keep the pin in sync with `environment.lock.json`.

## The rows this instrument produced

Model: `litert-community/gemma-4-E2B-it-litert-lm` @ `9262660` (the pinned June
snapshot — see `environment.lock.json`), pinned n=100, greedy (topK 1 / topP 1.0 /
temp 0), canonical extractor. **`--max-tokens` here is the TOTAL context**
(`EngineConfig.maxNumTokens`), not a generation budget (fairness-rules rule 3).

| row | instrument state | command (from repo root) |
|---|---|---|
| **88.0** (ctx2048, thinking OFF — version re-check of the published 86.0) | this pin (v0.15.0) | `python3 scripts/parity_gsm8k.py --which int4 --greedy --max-tokens 2048 --n 100 --litertlm <model> --tag litert-gemma4-e2b-v0150-off` |
| **89.0** (ctx4096, OFF) | this pin | same with `--max-tokens 4096`, tag `...-ctx4096-off` |
| **92.0** (ctx4096, thinking ON) | this pin | add `--thinking`, tag `...-ctx4096-thinking` |
| 86.0 (published v0.13.1 row) | fork `main@c8eae63` (v0.13.1 pin, pre-`ThinkingConfig`) | historical: re-pin the dependency to that revision and drop the `--thinking`/`ThinkingConfig` lines from `main.swift` (the wrapper there predates the type). The 2026-08-04 re-measure of this row at v0.15.0 is the 88.0 above (+2.0 build delta, same surface + protocol). |

Raw capture: `results/raw/2026-08-04-litert-0150-yardstick-thinking/` (README + both
run logs). Cross-validation: the released 0.15.0 pip dylib reproduces the ctx4096 pair
per-question (`results/quality/gsm8k_litert-gemma4-e2b-pip015gpu-*`).

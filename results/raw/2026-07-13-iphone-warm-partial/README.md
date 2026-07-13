# iPhone warm re-capture — Tier 1 PARTIAL (2026-07-13, NOT publishable)

Console-only (driver v1 did not copy device-jsonl back). Warm = median of runs 2-4.
DO NOT cite until reproducibility + persistence + Core AI are fixed (see next-session brief).

## Captured (console `tier1_console.txt`)
| model | runtime | build | cold | warm (r2-4 med) | note |
|---|---|---|---|---|---|
| Qwen3-0.6B | MLX | Release | 177.9 | ~179 | flat; but old Release-cold file reads 125.8 — 1.4x session variance UNEXPLAINED |
| Qwen3-0.6B | LiteRT | Release | 121.0 | ~120 | flat; matches published 118.6 |
| Qwen3-1.7B | MLX | Release | 54.8 | ~62 | ramps 57->62->62 (warmup or thermal?) |
| Qwen3-1.7B | LiteRT | — | FAIL | — | litert-local/qwen3-1.7b-int4 not on HF (download err) — needs side-load |
| Qwen3-0.6B/1.7B | Core AI ane/gpu | — | FAIL | — | bundle 'qwen3_0_6b_ane/gpu' not on device — needs assemble+side-load |

## Preliminary signal (NOT confirmed)
If MLX 0.6B warm ~179 holds, LiteRT 120 = 0.67x MLX — i.e. the published "118.6 vs 119.6 tie"
was LiteRT-Release vs MLX-**Debug** (119.6 is a Debug row). Would confirm a Debug/Release
contamination on top of the cold/warm issue. MUST reproduce cleanly before claiming.

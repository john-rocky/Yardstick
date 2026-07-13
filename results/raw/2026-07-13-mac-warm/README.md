# Mac M4 Max warm sweep — REPO harness (2026-07-13 night)

**Protocol**: `yardstick run --task short-chat --runtime <rt> --model <id> --runs 5`
(this repo's CLI, commit 97ae5f7+; run 1 = cold, runs 2-5 = warm, per-run JSONL with
`coldRun` flags; short-chat = ~19-tok prompt, 128 tok, greedy). macOS 27.0 **26A5378j**
(recorded in every jsonl), Mac Studio M4 Max, GPU idle, 30 s between models, all runs
`initialThermalState=nominal`. LiteRT-LM v0.13.1 via MediaPipeRuntime (WebGPU→Metal);
MLX via MLXRuntime (mlx-swift 0.31.4 / mlx-swift-lm b95dc780).

**Harness note**: this sweep is the first to use the MediaPipeRuntime stream-drain fix
(see commit): the token-cap no longer abandons the stream, which removed the ~10-min
inter-run stall. Measurement window ends at the cap timestamp — decode/ITL semantics
are unchanged vs June.

**Retry pass**: 7 litert cells were re-run ~15 min after the main sweep with a CLI
rebuilt to add two catalog entries (Gemma3-1B-IT, Phi-4-mini-instruct — catalog-only
diff, no measurement code change) and locally staged litert-local dirs.

## Decode tok/s (cold = run 1 / warm = median runs 2-5)

| Model | MLX cold/warm | LiteRT cold/warm | June ref (cold-era) | notes |
|---|---|---|---|---|
| Qwen3-0.6B | 561.6 / 553.8 | 271.1 / 270.4 | MLX 455† / — | † June used 512p/1024g protocol, not short-chat |
| Qwen3-1.7B | 324.2 / 325.0 | 94.9 / **171.1** (int4-mixed) | MLX 322.7 / LiteRT "115.8" | June LiteRT artifact unidentifiable (int4-labelled, int8-gate mismatch, file deleted) → re-baselined on int4-mixed; cold 94.9 = first-run cache build (litert-mac-verify warm 172.6 cross-validates) |
| Qwen3-4B | 163.1 / 162.2 | 111.2 / 110.9 | — | |
| Gemma-4-E2B | (blocked: upstream loader) | 133.0 / **155.9** | card 152-160@1029tok ✓ | +17% warm ramp; short-chat protocol |
| DeepSeek-R1-1.5B | 330.8 / 332.8 | 119.3 / 119.0 (q8) | 323.6 / 115.9 | ✓ reproduces |
| TinySwallow-1.5B | 328.8 / 328.0 | 120.4 / 120.6 (q8) | 326.7 / 119.6 | ✓ |
| VibeThinker-1.5B | (not swept — June 176.3§ provenance TBC) | 120.3 / 120.4 (q8) | 176.3§ / 119.6 | ✓ litert |
| Gemma3-1B | 328.4 / 328.2 | 182.0 / 181.3 (official int4) | 345.5 / 185.7 (self-conv) | official int4 ≈ self-conv −2.4%; MLX −5% vs June |
| OLMo-2-1B | (no mlx repo) | 135.1 / 136.3 | — / 140.2 | ✓ (−3%) |
| Llama-3.2-3B | 207.6 / 208.1 | 92.9 / 93.3 | 208.0 / 94.0 | ✓✓ |
| SmolLM3-3B | 196.0 / 196.6 | 90.9 / **ANOMALY** | 196.0 / 91.2 | litert warm DECLINES in-process 91.4→78.9→53.2→55.8 (ITL only +20% → inter-chunk stalls accumulate; cause unknown, needs investigation). Cold matches June. Do NOT quote a warm median for this cell |
| Ministral-3-3B | (June: mlx can't load) | 91.7 / 92.0 | — / 92.8 | ✓ |
| Phi-4-mini | 169.0 / 169.1 | 42.0 / 64.9 (q8) | 167.4 / 67.7 | warm ≈ June −4%; cold 42 = first-run cache build |

## Headline observations (LOCAL — not published; user review pending)

1. **The June Mac table reproduces within ±5% on today's OS/harness** for every cell with the
   same artifact — Mac numbers were already steady-state; no cold-bias correction needed there.
2. **Cold ≈ warm on Mac** except: E2B (+17% engine ramp), Qwen3-1.7B-int4 & Phi-4-mini
   (first-run cache build depresses cold), SmolLM3 (declining anomaly).
3. LiteRT/MLX warm ratios (Mac): 0.49 (0.6B), 0.53 (1.7B int4-mixed), 0.68 (4B), 0.36 (DeepSeek q8),
   0.37 (TinySwallow q8), 0.55 (Gemma3-1B int4), 0.45 (Llama), 0.38 (Phi-4 q8) — int8 rows carry the
   2x byte penalty as documented in June.

Raw: one `<runtime>-<model>.jsonl` per cell (5 runs each), `*.stderr.log`, `FAILURES.txt`
(first-pass catalog misses, superseded by the retry pass).

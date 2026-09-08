# Where the Apple-GPU decode gap lives, two months later — LiteRT-LM 0.17.0 on M4 Max

**Re-capture of the [2026-07-07 Metal decomposition](metal-profile-m4max.md) on LiteRT-LM
0.17.0, same cells, same protocol, same machine — plus the version ladder in between, a
per-kernel-family cost split obtained by skipping dispatches inside the running engine, and a
kernel experiment that shows which change to an int4 decode GEMV pays.**

**Hardware:** Mac Studio · Apple M4 Max · 40-core GPU (**546 GB/s**) · 128 GB · macOS 27.0 (26A5416b;
07-07 ran 26A5353q).
**Runtimes:** LiteRT-LM **0.17.0** — there is no GitHub release for it (latest release: v0.16.1,
2026-08-18); the only shipped macOS runtime is the PyPI pair `litert-lm==0.17.0` +
`litert-lm-api==0.17.0` (uploaded 2026-09-04), whose `liblitert-lm.dylib` statically links the
**WebGPU (Dawn → Metal) GPU delegate** — the accelerator log still reads `GPU WebGPU`, adapter
`Apple M4 Max … backend=Metal`, the same architecture the v0.13.1 xcframework had on 07-07. The
build corresponds to LiteRT-LM `38fb04e8` (2026-09-04, last commit before the 0.18.0 bump) and pins
LiteRT `9fe5be45` (2026-08-27); 0.13.1 pinned LiteRT `a412f505` (2026-06-01). MLX-LM 0.31.3 (still
the latest) on mlx 0.32.2.
**Protocol:** identical to 07-07 — greedy steady-state decode of the same long essay prompt,
Engine/session API (Python API instead of the Swift wrapper: `Engine` → session with
`top_k=1, temperature=0` → `run_prefill` → `run_decode_async`), per-token timestamps on the
streamed chunks, steady rate excludes the first 32 tokens; 1200–1500 generated tokens per run;
Instruments `Metal System Trace` attached 3 s into decode for 6 s; cells interleaved ABAB, 3 rounds
(round 1 traced), 15 s cooldown + thermal + GPU-idle gate before every launch. "Streamed GB" is now
read from the artifact's flatbuffer (all constants of the decode subgraph minus gather-only
tables), not from file size. Date: 2026-09-08. Author: john-rocky. Primary records:
[`results/raw/2026-09-08-m4max-metal-profile-0170/`](../results/raw/2026-09-08-m4max-metal-profile-0170/).

## Headline — per-token decomposition, LiteRT-LM 0.17.0 (steady decode, GPU hardware channels)

| cell | tok/s (median n=3, spread) | token ms | GPU-busy ms | GPU-idle ms | busy % | streamed GB | in-kernel GB/s | %-roof |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LiteRT Qwen3-4B mixed-int4 (2.66 GB file) | 113.6 (0.3%) | 8.80 | 8.80 | 0.01 | 99.9% | 2.263 | 257 | 47% |
| MLX Qwen3-4B 4-bit (2.26 GB) | 154.3 (0.5%) | 6.48 | 6.42 | 0.03 | 99.6% | 2.26 | 352 | 64% |
| LiteRT DeepSeek-R1-1.5B **q8** (1.83 GB file) | 125.2 (1.0%) | 7.99 | 8.02 | 0.01 | 99.8% | 1.544 | 193 | 35% |
| LiteRT DeepSeek-R1-1.5B **int4** (1.11 GB file) | 129.8 (1.3%) | 7.70 | 7.63 | 0.01 | 99.9% | 0.869 | 114 | 21% |
| MLX DeepSeek-R1-1.5B 4-bit (1.00 GB) | 309.6 (0.4%) | 3.23 | 3.23 | 0.01 | 99.6% | 1.00 | 310 | 57% |

`in-kernel GB/s` = streamed weight bytes / GPU-busy time per token (weights-only lower bound, the
07-07 follow-up-4 convention); `%-roof` against 546 GB/s. Same artifacts as 07-07
(`litert-community/Qwen3-4B@84cc5a35`, `litert-community/DeepSeek-R1-Distill-Qwen-1.5B@2f8b8ee9`,
`mlboydaisuke/DeepSeek-R1-Distill-Qwen-1.5B-LiteRT@3c8f42f9`, `mlx-community/*-4bit`).

**Against 07-07 (v0.13.1 xcframework):** LiteRT Qwen3-4B 102.5 → 113.6 (+11%), DeepSeek q8
111.4 → 125.2 (+12%), DeepSeek int4 128.0 → 129.8 (+1%); MLX 153.3 → 154.3 and 292.0 → 309.6.
A same-day control with the 0.13.1 *wheel* (same driver, same OS build) reads Qwen 106.6, q8
116.4, int4 137.5 — i.e. the instrument/OS drift since 07-07 is +4/+4/+7% and the runtime change
0.13.1 → 0.17.0 is **+6.6% / +7.6% / −5.6%**.

**Qwen3-4B iso-int4: the deficit is now 2.32 ms/token (07-07: 3.24), and it is all in-kernel:**

| bucket | ms/token | share | how measured |
|---|--:|--:|---|
| in-kernel bandwidth deficit (257 vs 352 GB/s on 2.263 GB) | 2.38 | **~100%** | GPU-busy time vs streamed bytes |
| extra bytes streamed (2.263 vs 2.26 GB) | 0.00 | 0% | flatbuffer constants minus gather-only tables |
| GPU idle (submit/sample bubbles) | −0.02 | 0% | gap analysis (0.01 vs 0.03 ms) |

## What changed since 07-07, and where in the version ladder

Same-day traces of the 0.13.1 wheel reproduce the 07-07 structure exactly, so the comparison
below is runtime-only (same OS, same instrument, same hour):

| cell | GPU-busy ms/token 0.13.1 → 0.17.0 | idle ms/token 0.13.1 → 0.17.0 | blits/token | encoders/token |
|---|--:|--:|--:|--:|
| Qwen3-4B int4 | 8.75 → 8.80 | **0.69 → 0.01** | 3.0 → 2.0 | 12.8 → 12.9 |
| DeepSeek q8 | 8.12 → 8.02 | **0.69 → 0.01** | 3.0 → 2.0 | 12.2 → 12.1 |
| DeepSeek int4 | **6.73 → 7.63** | **0.64 → 0.01** | 3.0 → 2.3 | 10.9 → 12.5 |

1. **The two per-token idle bubbles are gone.** 07-07 finding 2 (a ~0.4 ms gap after the readback
   blit while the CPU samples/streams, plus a ~0.25 ms gap after the last encoder while the CPU
   re-encodes) is still there on 0.13.1 today — after-blit 0.43 ms + after-last-encoder 0.26 ms,
   ~2 gaps/token — and is **0.01 ms/token on 0.17.0** with one blit fewer per token. That
   0.64–0.69 ms/token is the whole of the q8 and Qwen speed-ups; their kernel time did not move
   (±1%).
2. **The ladder puts the fix in 0.15.0.** Same driver, same artifacts, median of 2 (tok/s):

   | litert-lm-api | Qwen3-4B int4 | DeepSeek q8 | DeepSeek int4 |
   |---|--:|--:|--:|
   | 0.13.1 (2026-06-03) | 106.6 | 116.4 | 137.5 |
   | 0.14.0 (2026-07-08) | 108.4 | 118.7 | **120.7** |
   | 0.15.0 (2026-08-04) | **115.7** | **126.3** | 132.0 |
   | 0.16.1 (2026-08-18) | 116.1 | 126.8 | 132.3 |
   | 0.17.0 (2026-09-04, n=3) | 113.6 | 125.2 | 129.8 |

   0.14.0 → 0.15.0 is +6–7% on every cell. The v0.15.0 release notes mention only the Apple FM
   adapter, CLI config and JS changes; the 248 commits in that window carry the candidates —
   `42d10355` "Asynchronous API" (07-23), `5f43d8a0`/`0c435964` exposing
   `gpu_decode_steps_per_sync` and the GPU performance flags (07-21), `f9472878`
   `gpu_enable_metal_residency_set` (07-16), and the LiteRT pin moving from `622f1f3c` (06-29) to
   `3cb830ad9` — the "Update Dawn to v20260720 and use modern WebGPU cache callbacks" commit
   itself (07-28), one commit after "external WebGPU instance and flush callback" (07-27). Which
   of them removed the bubbles is **not established here**; only the version is.
3. **DeepSeek int4 regressed in 0.14.0 and never recovered.** On the third-party gs32 artifact
   the kernel time rose 6.73 → 7.63 ms/token between 0.13.1 and 0.14.0 while the q8 build's did
   not move, and greedy output diverges from 0.13.1 after ~1100 tokens (numerics changed). The
   official Qwen3-4B artifact kept its kernel time. The v0.14.0 notes say nothing about it;
   LiteRT-side candidates in the pin window (`a412f505` → `622f1f3c`) are the
   FullyConnected-generator commits of 06-12/06-17.
4. The 0.17.0 tail (0.16.1 → 0.17.0: −2%) is within the day's drift band; the 07-07 follow-up-2
   observation that int4-blockwise artifacts fail on v0.14.0-alpha.0 does not apply to any release.

## Is the int4-vs-int8 anomaly still there? Yes — wider

Whole-token, streamed basis (07-07 follow-up 4 convention): DeepSeek q8 193 GB/s vs int4 114 GB/s
⇒ per-byte penalty **1.7×** (07-07: 1.55×). int4 reads 56% of q8's bytes and decodes 4% faster.

The per-family split below isolates the GEMVs and makes it starker.

## Per-family cost split without counters: skipping dispatches inside the engine

Headless GPU counters are still unavailable (methodology), so the split was measured by
**ablation**: a `DYLD_INSERT_LIBRARIES` shim (`scripts/metal-profile/mtl_dump.m`) records every
Metal shader source and dispatch the delegate issues, and can then *skip* dispatches whose kernel
source hash is on a list once the prefill is done. Skipping a family and re-timing steady decode
gives that family's cost; comparing the output text with an unmodified run says whether the
skipped results were consumed. 300-token runs at a 512-token context (shorter than the protocol
runs, hence the faster baselines; only differences are used):

**Qwen3-4B, 0.17.0 — 8.74 ms/token (99.9% GPU-busy) splits as:**

| family (dispatches per layer) | ms/token | share |
|---|--:|--:|
| 7 weight GEMVs: q, k, v, o, gate, up, down | **6.39** | 73% |
| attention: scores, softmax, PV (6) | 0.75 | 9% |
| RMSNorm + Q/K-norm (2) | 0.41 | 5% |
| everything else: RoPE, cache writes, embedding, output head, sampler (14 + prelude) | 1.19 | 14% |

The layer GEMVs stream 1.769 GB of int4 + 0.221 GB of fp16 scales per token ⇒ **311 GB/s, 57% of
the roof, inside the GEMV family** (the whole-token 257 GB/s dilutes it with the 2.35 ms of
non-GEMV kernels — 22 small dispatches per layer).

**DeepSeek-R1-1.5B GEMV family (layer weights only, output head excluded):**

| build | GEMV ms/token | bytes | GB/s | %-roof |
|---|--:|--:|--:|--:|
| q8, 0.17.0 | 3.52 | 1.311 GB int8 | 372 | 68% |
| int4, 0.17.0 | **4.17** | 0.632 GB int4 + 0.079 GB scales | **170** | 31% |
| int4, 0.13.1 (same day) | 3.02 | same | 235 | 43% |

**The int4 GEMVs take longer than the int8 GEMVs in absolute time while reading 54% of the
bytes** — a 2.2× per-byte penalty at the kernel level on 0.17.0 (1.6× on 0.13.1). The anomaly is
entirely inside the weight GEMV kernels: the non-GEMV remainder is 3.8 ms (int4) vs 4.6 ms (q8),
the q8 graph having 41 dispatches per layer to int4's ~30.

## What the dispatch dump adds, at results level

Two things the dump corrects in the 07-07 text:

- **The first token is not the steady state.** Prefill and decode token 1 run one pipeline set
  (40 dispatches per layer); after the first decode step the engine compiles a second set, and
  steady decode runs 29 dispatches per layer, 7 of them the weight GEMVs. A dump that stops after
  ~3000 dispatches describes the wrong kernels — 07-07 finding 3 ("the dequant is fused — encoder
  counts do not grow") happened to describe steady state without having seen it.
- **The steady-state weight GEMVs are 5 kernel sources** (3 of them shared between the Qwen3-4B
  and the DeepSeek int4 builds), one dispatch per matrix per token. Their source hashes are in the
  records (that is what the ablation keys on); the sources are the vendor's and stay offline.

## The fix-direction experiment: a baseline that matches the engine, one change at a time

[`experiments/int4-dequant-fusion/gemv_variants.swift`](../experiments/int4-dequant-fusion/gemv_variants.swift)
times an int4 GEMV written in the K-sliced, threadgroup-reduction style common to WGSL delegates
(`L`: 16 channel-groups × 16 K-slices per 256-thread threadgroup, fp16-arithmetic nibble unpack,
per-32-K scale and zero) over the whole per-token weight set (36 layers × 7 matrices, distinct
buffers) in one command buffer, GPU-side. `L` lands at 6.6–7.2 ms/token against the 6.39 ms the
ablation measured inside LiteRT-LM, close enough to serve as the baseline. Variants change one
thing each (Qwen3-4B shapes, two runs, drift ±5%):

| variant | change | ms/token | GB/s (int4+scale+zero) |
|---|---|--:|--:|
| `L` | K-sliced threadgroup-reduction baseline (matches the engine's GEMV family within 4–12%) | 6.6–7.2 | 316–343 |
| `L_int` | integer nibble extraction instead of fp16 arithmetic | 7.3–7.9 | 286–313 (slower) |
| `L_nozp` | scale only, no zero-point | 6.3–6.8 | 335–362 |
| `L_u4` | 16-byte loads (2 K-steps per load) | 6.5–7.0 | 325–349 |
| `L_ks8` / `L_ks4` | 8 / 4 K-slices per threadgroup | 8.7 / 14.7 | 245 / 149 |
| `L_ks32` / `L_ks64` | 32 / 64 K-slices (8 / 4 channel-groups) | 6.3 / 6.05 | 361 / 375 |
| **`C`** | **row-major `[N][K]`, one simdgroup per output row, lanes stride along K, `simd_sum` reduction** | **5.2–5.5** | **409–438** |
| `C_ftrick` | `C` with the fp16-arithmetic unpack | 5.5 | 412 |

DeepSeek-R1-1.5B shapes: `L` 3.0–3.3 ms vs `C` 2.2–2.3 ms (−28%); `L_ks64` 2.66.

Reading: the nibble arithmetic is not the lever (fp16 arithmetic is the *faster* unpack on this
GPU), loads and zero-points are worth a few percent, and the K-slicing depth matters only when it
starves occupancy. **The decomposition is the lever:** a one-simdgroup-per-row layout with a
simdgroup reduction and no threadgroup memory streams the same bytes 22–28% faster than the
K-sliced threadgroup decomposition. On Qwen3-4B that is ~1.2 ms/token out of the 6.39 ms GEMV
family (token 8.8 → ~7.6 ms, +16% tok/s); the remaining ~1.2 ms of the gap to MLX sits in the 22
non-GEMV dispatches per layer. For the DeepSeek int4 build the baseline design already runs those
shapes in 3.0 ms against the engine's 4.17, so the reachable saving there is ~1.9 ms/token — the
whole int4-vs-int8 anomaly.

Caveats: `L` matches the engine's GEMV time but is not its code, so the variant deltas transfer as
directions, not as numbers. No DRAM counters were available; every GB/s here is weights-over-time.

## Reproduce

```bash
# runtime: pip wheels (no GitHub release for 0.17.0)
uv venv ~/venvs/lt0170run --python 3.14 && uv pip install --python ~/venvs/lt0170run/bin/python litert-lm==0.17.0
# one cell, traced: PID + DECODE_START markers let profile_cell.sh attach xctrace mid-decode
scripts/metal-profile/profile_cell.sh litert_qwen3_4b_int4 out/ traces/ 3 6 -- \
  ~/venvs/lt0170run/bin/python scripts/metal-profile/litert_decode_driver.py qwen3_4b_mixed_int4.litertlm \
  --prompt "<essay prompt>" --backend gpu --max-num-tokens 1536 --max-output-tokens 1200 --out out/cell.json
python3 scripts/metal-profile/analyze_cell.py traces/litert_qwen3_4b_int4.gpu.xml <tok_per_s> 2.263
python3 scripts/metal-profile/gap_structure.py traces/litert_qwen3_4b_int4.gpu.xml <tok_per_s>
# streamed GB from the artifact; kernel dump + ablation; GEMV baseline and variants
scripts/metal-profile/litertlm_weight_bytes.py model.litertlm
clang -fobjc-arc -dynamiclib -framework Metal -framework Foundation scripts/metal-profile/mtl_dump.m -o libmtl_dump.dylib
MTL_DUMP_DIR=dump MTL_DUMP_SKIP_AFTER=1750 MTL_DUMP_SKIP_HASHES=<hashes> DYLD_INSERT_LIBRARIES=$PWD/libmtl_dump.dylib \
  ~/venvs/lt0170run/bin/python scripts/metal-profile/litert_decode_driver.py ... --max-output-tokens 300
cd experiments/int4-dequant-fusion && swiftc -O gemv_variants.swift -o gemv_variants && ./gemv_variants 20
```

Method and tooling traps (first-token-only dumps, short-run tok/s, the fp16-intermediate
microbenchmark that measures the wrong path): [`methodology/metal-profiling.md`](../methodology/metal-profiling.md).
`.trace` bundles, XML exports, dispatch-geometry summaries and the extracted vendor shader
sources are kept offline; the results directory holds the run records, trace analyses, kernel
hashes and the ablation table.

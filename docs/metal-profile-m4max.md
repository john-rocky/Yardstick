# Where the Apple-GPU decode gap lives — Metal-level decomposition (M4 Max)

**A kernel-level answer to the question the headline tables raise: when LiteRT-LM decodes at
~0.6–0.67× of MLX on the same Apple GPU at the same weight size, where does the time go?**
Instruments `Metal System Trace` attached mid-decode + a `powermetrics` cross-check, decomposing
each cell into GPU-idle / in-kernel-bandwidth / extra-bytes, and isolating the int4-vs-int8
anomaly inside one runtime.

**Hardware:** Mac Studio · Apple M4 Max · 40-core GPU (**546 GB/s**) · 128 GB · macOS 27.0 (26A5353q).
**Runtimes:** LiteRT-LM v0.13.1 release xcframework (GPU = WebGPU/Dawn→Metal; accelerator log
`GPU WebGPU`) via `litert-mac-verify --backend gpu`; MLX-LM 0.31.3 (`--temp 0.0`).
**Protocol:** steady-state greedy decode (512–1800 tok), long essay prompt; trace analysis on the
`metal-gpu-intervals` table; "weight GB" = artifact file size (same convention as the
[main report](litert-community-vs-mlx-coreai.md)). Decode rates reproduced within 3–10% of the
published Mac numbers before profiling. Date: 2026-07-07. Author: john-rocky.

## Headline — per-token decomposition (steady decode, measured on GPU hardware channels)

| cell | tok/s | token ms | GPU-busy ms | GPU-idle ms | busy % | in-kernel GB/s | %-roof |
|---|--:|--:|--:|--:|--:|--:|--:|
| LiteRT Qwen3-4B mixed-int4 (2.66 GB) | 102.5 | 9.76 | 9.08 | 0.68 | 93.0% | 293 | 54% |
| MLX Qwen3-4B 4-bit (2.26 GB) | 153.3 | 6.52 | 6.50 | 0.02 | 99.7% | 348 | 64% |
| LiteRT DeepSeek-R1-1.5B **q8** (1.83 GB) | 111.4 | 8.97 | 8.33 | 0.64 | 92.8% | 220 | 40% |
| LiteRT DeepSeek-R1-1.5B **int4** (1.11 GB) | 128.0 | 7.81 | 7.09 | 0.73 | 90.7% | 157 | 29% |
| MLX DeepSeek-R1-1.5B 4-bit (1.00 GB) | 292.0 | 3.42 | 3.31 | 0.11 | 96.7% | 302 | 55% |

`in-kernel GB/s` = weight bytes / GPU-busy time per token (weights-only lower bound on achieved
DRAM read bandwidth inside kernels). `%-roof` = against the 546 GB/s ceiling, busy-time basis.

**Qwen3-4B iso-int4: the 0.67× vs MLX = 3.24 ms/token deficit, split three ways:**

| bucket | ms/token | share | how measured |
|---|--:|--:|---|
| in-kernel bandwidth deficit (293 vs 348 GB/s) | 1.44 | **44%** | GPU-busy time vs bytes |
| extra bytes in the artifact (2.66 vs 2.26 GB) | 1.14 | **35%** | file sizes, at MLX's 348 GB/s |
| GPU idle (submit/sample bubbles) | 0.66 | **20%** | gap analysis |

## Findings

1. **The delegate keeps the GPU 91–93% busy.** Dawn's many-small-command-buffer pattern is not
   the story: MLX submits *more* command buffers per token (16 vs LiteRT's 9–12) and idles less.
   The gap is first kernel efficiency, then bytes, then scheduling.
2. **GPU idle = exactly two bubbles per token, ~250–450 µs each** (99% of all idle): (a) after
   the logits-readback blit, while the CPU samples/detokenizes/streams the token; (b) after the
   last compute encoder, while the CPU re-encodes the next token's command buffers. The cost is
   a near-fixed 0.64–0.91 ms/token across models, so its share grows as kernels get faster
   (7% at 9-ms tokens → 13% on gemma-4-E2B's 6-ms tokens). MLX pipelines both (gap p99 ≈ 1 µs).
3. **The int4 anomaly is a kernel story** (q8 vs int4, same runtime, same model): int4 reads 39%
   fewer weight bytes but is only 15% faster. The dequant *is* fused — encoder count per token
   does not grow (11 vs 12), blits identical (3/token), no extra passes — but per-byte kernel
   time rises **1.40×** (dominant GEMV encoders get *longer*: p50 1598 → 1912 µs on fewer
   bytes). KV re-reads (~14 MB/token ≈ 1%) can't explain it. The int4×int8 blockwise-gs32 GEMV
   is dequant/ALU-limited, not DRAM-bound; MLX runs gs32 affine-4-bit dequant on the same model
   at 302 GB/s, so blockwise dequant per se is not the limiter. At its own q8 kernel's per-byte
   efficiency the int4 build would decode ~173 tok/s (+35%); at MLX's, ~230 tok/s (+80%). The
   same anomaly shows in the published iPhone cells (q8 65% vs int4 56% of ceiling).
4. **powermetrics cross-check (idle-machine baseline 0.01 W / 666 MHz):** the GPU sits at its
   top frequency state (**1578 MHz**) in every cell, both runtimes — the in-kernel deficit is
   iso-clock, not power management. And the power signature agrees with (3): the int4 build
   draws **2.8 W more** than q8 (20.2 vs 17.4 W) while reading 39% fewer bytes ⇒ **energy per
   token is identical (148 vs 147 mJ)** — today int4 buys capacity, not efficiency. Per-token
   energy: LiteRT Qwen3-4B 262 mJ vs MLX 164; LiteRT DeepSeek 147–148 vs MLX 71.
5. **gemma-4-E2B control:** identical dispatch structure (14 encoders + 3 blits + the same two
   bubbles per token; idle 0.91 ms). Gemma's speed comes from reading fewer bytes per token
   (PLE row-lookup + QAT mixed 4/8-bit), not from a privileged dispatch path on macOS.
6. Roofline % is device-relative: even MLX reaches only 53–64% of the 546 GB/s part (vs 86% on
   iPhone's ~80 GB/s). Compare relative gaps across devices, not absolute %. (Consistent with
   the [Gemma-4-12B study](gemma4-12b-mac.md): MLX 68%, LiteRT 54%-roof at 12B.)

Caveat: without DRAM-byte counters (see methodology), "kernel reads extra bytes" vs "kernel
achieves lower GB/s" cannot be separated inside the 44% bucket; the idle and per-byte-time
measurements do not depend on it.

## Follow-up (same day): three falsification experiments

1. **The GPU top-k sampler was already active** in every LiteRT cell above: the release build
   statically links `LiteRtTopKWebGpuSampler` and uses it when the dlopen'd dylib is absent
   (the "GPU sampler unavailable" warning is followed by a static-C-API fallback, not CPU
   sampling). This refines bubble (a)'s content: the readback blit carries the *sampled id*,
   not the logits vector; the CPU-side work in the bubble is detokenize/stream/stop-check plus
   the next step's encode. The two-bubble structure and its fixed ~0.7 ms/token cost stand.
2. **Runtime v0.14.0-alpha.0 does not change the loop:** DeepSeek q8 decodes 112.5–114.5 tok/s
   (v0.13.1: 111.4); the trace shows a leaner dispatch (2 blits, 7 cmdbufs/token vs 3 and 9–12)
   but the same two bubbles (idle 0.98 ms/token). Separately, all int4-blockwise artifacts fail
   WebGPU delegate preparation on that alpha (`Read selector with single argument can be used
   only with linear storage types`, ml_drift merge_nodes) — including the official
   litert-community Qwen3-4B artifact — so the int4 comparison cannot be repeated on it.
3. **Bigger quant blocks make the int4 kernel *worse*, not better.** If the kernel were limited
   by per-group dequant overhead, gs128 (4× fewer groups) should approach q8's per-byte
   efficiency. Measured on a same-model pair (SmolLM2-1.7B, OCTAV int4, identical dispatch:
   10 encoders / 3 blits / 8 cmdbufs / same idle):

   | | block32 | block128 |
   |---|--:|--:|
   | artifact | 1.168 GB | 1.088 GB (−6.9%) |
   | decode tok/s (2 runs) | 99.2 / 99.0 | 93.7 / 93.5 |
   | GPU-busy ms/token | 9.69 | 10.30 |
   | in-kernel GB/s | 121 | 106 (−12%) |

   Per-byte kernel time is 13% worse at gs128 ⇒ the int4×int8 GEMV is *specialized for gs32*
   rather than dequant-group-count-limited, and no converter-side block-size choice mitigates
   the in-kernel bucket. This narrows finding (3): the fix is inside the kernel itself.
4. **Streamed bytes ≠ stored bytes — the "extra artifact bytes" bucket dissolves.** Inspecting
   the Qwen3-4B artifact's flatbuffer: 2.652 GB of buffers = 1.817 GB int4 block weights +
   0.227 GB f16 scales + 0.219 GB int4 output head + **0.389 GB int8 embedding table**. The
   embedding feeds an `EMBEDDING_LOOKUP` (gather) — decode reads ~2.5 KB of it per token, not
   the full table. The tied-vocab MLX artifact stores that matrix once at 4-bit and streams it
   through the lm_head matmul. Net: **per-token streamed weights are 2.263 GB for LiteRT vs
   2.26 GB for MLX — parity.** The headline decomposition must be restated: the 3.24 ms/token
   gap is **~80% in-kernel efficiency + ~20% idle, ~0% extra bytes** (the 0.40 GB file-size
   delta is a download/RAM cost, not a decode-time cost). On a streamed-bytes basis the
   in-kernel numbers become: LiteRT Qwen3-4B int4 **249 GB/s**, DeepSeek q8 **192**, DeepSeek
   int4 **124** (MLX cells unchanged — their vocab matrix is streamed). The int4-vs-q8
   per-byte penalty rises from 1.40× to **1.55×**, and the kernel-fix sizing from +35/+80% to
   **+47% (at q8 efficiency) / +115% (at MLX efficiency)**. When reproducing with
   `analyze_cell.py`, pass streamed GB (file size minus gather-only tables), not file GB.
5. **iPhone A19 confirms the int4 penalty — but milder, and it is implementation-specific.**
   Same DeepSeek-R1-1.5B q8/int4 pair on an iPhone 17 Pro (A19 GPU, LiteRT-LM v0.13.1 iOS
   xcframework = **native Metal**, not WebGPU/Dawn), decode-level (per-encoder Metal traces do not
   finalize for headless export on iOS 27β — see methodology; at 25–40 tok/s the idle bubble is
   <3%, so decode GB/s is a tight in-kernel proxy):

   | model (A19) | decode tok/s | streamed GB/s | %-of-q8 per-byte |
   |---|--:|--:|--:|
   | int4 | 40.3 | 35.5 | 85% |
   | q8 | 26.1 | 41.8 | 100% |

   int4 is only **1.54×** faster than q8 while reading 55% of the bytes (byte-ideal 1.54→1.82×) ⇒
   an int4 per-byte penalty of **1.18× on A19, versus 1.55× on the Mac WebGPU path**. The penalty
   is real on both, but the native Metal int4 GEMV closes most of it — evidence the deficit lives
   in the *kernel implementation* (WebGPU/Dawn) more than in the int4×int8 blockwise scheme itself,
   and that a reworked Mac kernel has an existence proof to aim at. (Consistent with the published
   iPhone ceiling figures: q8 65% vs int4 56%, ratio 0.86 ≈ the 0.85 measured here.)

## Reproduce

```bash
# decode run (LiteRT; greedy, Engine API, GPU backend)
litert-mac-verify <model>.litertlm "<long prompt>" --max-tokens 900 --backend gpu
# attach Metal System Trace DURING decode (--launch deadlocks the WebGPU runtime; see methodology)
xcrun xctrace record --template 'Metal System Trace' --time-limit 6s \
  --output cell.trace --attach <pid>
# export + analyze the GPU intervals table
xcrun xctrace export --input cell.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output cell_gpu.xml
python3 scripts/metal-profile/analyze_cell.py cell_gpu.xml <tok_per_s> <weight_gb>
```

Full method, tooling gotchas (xctrace `--launch` deadlock, headless-counter limitation,
contamination gate): [`methodology/metal-profiling.md`](../methodology/metal-profiling.md).
Analysis scripts: [`scripts/metal-profile/`](../scripts/metal-profile/). Raw evidence
(powermetrics log, run timelines/logs): `results/raw/m4max-metal-profile/`. The `.trace`
bundles (~1.5 GB) are kept offline; regenerate with the commands above.

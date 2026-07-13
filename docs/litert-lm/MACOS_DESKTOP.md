# macOS desktop — ~~LiteRT-LM is CPU-only on Apple Silicon~~ RETRACTED (finding + Qwen3 scaling)

> **⚠️ RETRACTION (2026-07-13).** The headline finding of this page was a misreading. The
> `Failed to create OpenCL context` INFO line is a **benign probe inside GpuEnvironment** — it
> appears on the *working* GPU path too, immediately followed by `Created Metal device from
> provided device id`. Re-run on 2026-07-13, the current `yardstick` source build selects the
> **WebGPU/Dawn→Metal GPU backend** (adapter log: `Apple M4 Max … backend=Metal`) and decodes
> Gemma-4-E2B at 130 tok/s cold — clearly GPU, not XNNPACK. Evidence:
> [`../../results/raw/2026-07-13-e2b-mac-webgpu/`](../../results/raw/2026-07-13-e2b-mac-webgpu/).
> The Qwen3 scaling numbers below therefore need re-verification: they may have been GPU numbers
> all along (their ~50–68%-of-MLX ratios match the known WebGPU-vs-native-Metal kernel gap), and
> the "CPU" labels in RESULTS.md derived from this page are suspect until re-captured. Kept for
> provenance; do not cite the CPU-only claim.

> Part of the [LiteRT-LM package](README.md). Desktop tier (Apple M4 Max). Kept separate from the
> iPhone GPU tables because **the headline here is a compute-unit finding, not a like-for-like GPU
> comparison** — see below. Numbers are pulled from the raw JSONL in
> [`../../results/raw/`](../../results/raw/) (files listed at the bottom); decode is the median of
> n=3 cold runs, short-chat task, 128-token output cap, Release build of the `yardstick` macOS CLI.

## Finding: LiteRT-LM's GPU backend does not start on Apple-Silicon macOS → it runs CPU-only

On every macOS run, LiteRT-LM v0.13.x logs:

```
INFO: [gpu_environment.cc:367] Failed to create OpenCL context.
```

and then runs on the **CPU** (XNNPACK). On **iOS** the same engine and the same `.litertlm`
artifacts run on the **Metal GPU** (see the iPhone tables in [README.md](README.md)); the macOS
xcframework's GPU path is **OpenCL-based**, and Apple-Silicon macOS does not provide a usable
OpenCL GPU context — so the GPU backend is unavailable and execution falls back to CPU.

The CPU fallback is corroborated by the throughput: at 0.6B, LiteRT-CPU is ~48% of MLX's
Metal-GPU decode, settling to ~66% at 4B–8B (the larger, more bandwidth-bound models let the
CPU exploit M4 Max's high unified-memory bandwidth). **This is a real per-platform gap** — a
Metal GPU path on macOS is the obvious win for "performance across desktop."

A secondary observation (same root as the iPhone Qwen3 energy hang): a
`DEADLINE_EXCEEDED … callback_thread_pool` fires on **teardown** after each short-chat result is
captured. It's benign for short-chat (the result is recorded; the process exits 0), but it is the
same thread-pool timeout that makes sustained 600 s generation hang on-device.

## Qwen3 size-scaling — M4 Max, short-chat, 128-cap, median n=3

| Qwen3 | LiteRT-LM (**CPU**) tok/s | MLX-Swift (**GPU/Metal**) tok/s | LiteRT-CPU as % of MLX-GPU | LiteRT peak RAM | MLX peak RAM |
|---|---:|---:|---:|---:|---:|
| 0.6B | 271.9 | 561.4 🏆 | 48% | 801 MB | 652 MB |
| 4B   | 110.1 | 163.0 🏆 | 68% | 2382 MB | 2523 MB |
| 8B   | 65.2  | 98.3 🏆  | 66% | 3451 MB | 4757 MB |

- **Compute unit, not quant, is the dominant axis here.** Quantisation is each runtime's native
  4-bit (LiteRT mixed-INT4 blockwise gs32 / MLX Q4); the ~1.5–2× gap is GPU-vs-CPU, not bit-width.
  This row is **disclosed as CPU-vs-GPU on purpose** — it is the finding, not a hidden unfairness.
- **Memory crossover at 8B:** LiteRT (mmap'd weights) holds a *lower* peak `phys_footprint` than
  MLX (weights loaded into arrays) — 3451 vs 4757 MB — the same mmap effect seen on iPhone.
- LiteRT-CPU decode is extremely stable run-to-run (e.g. 8B: 65.2 ± <1 tok/s).

## Why this matters for LiteRT
The desktop story today is **CPU-only on Apple Silicon**. For a team optimising "performance and
memory across desktop, mobile, and web," the actionable items are (1) a Metal GPU backend on
macOS (the OpenCL path is a dead end on Apple Silicon), and (2) the `callback_thread_pool`
teardown/sustained-generation timeout (shared with the iPhone hang).

## Reproduce
```bash
cd ios/BenchmarkApp && ./scripts/bootstrap.sh            # clones Vendored/LiteRT-LM (v0.13.1)
cd ../.. && GIT_LFS_SKIP_SMUDGE=1 swift build -c release --product yardstick
YS=.build/release/yardstick
$YS run --task short-chat --runtime litert-lm --model litert-community/Qwen3-8B \
        --output results/raw/m4max-litert-lm-qwen3-8b-short-chat-run1.jsonl
$YS run --task short-chat --runtime mlx-swift --model mlx-community/Qwen3-8B-4bit \
        --output results/raw/m4max-mlx-qwen3-8b-short-chat-run1.jsonl
```
The SwiftPM `yardstick` CLI ships the MLX + LiteRT-LM adapters (llama.cpp / CoreML stay on the
xcodebuild target). The "Failed to create OpenCL context" line is emitted by LiteRT itself.

## Provenance
`results/raw/m4max-litert-lm-qwen3-{0.6b,4b,8b}-short-chat-run{1,2,3}.jsonl` and
`results/raw/m4max-mlx-qwen3-{0.6b,4b,8b}-short-chat-run{1,2,3}.jsonl`.

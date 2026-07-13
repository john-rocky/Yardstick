# Gemma-4-E2B on macOS — WebGPU GPU backend, M4 Max (2026-07-13)

Reconciliation against the official card numbers (7,835 prefill / 160 decode on "M4",
https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).

**Setup:** Mac Studio M4 Max (40-core GPU, 128 GB, macOS 27.0), LiteRT-LM v0.13.1 release
xcframework, WebGPU/Dawn→Metal backend, `litert-mac-verify <model> "<1,029-token prompt>"
--max-tokens 2048 --runs 5 --backend gpu` (the `--runs N` flag repeats generation in-process:
run 1 pays any remaining one-time cost, later runs are warm; each run is a fresh conversation).

## Result — `e2b_webgpu_1029tok_runs5_clean.log`

| run | decode tok/s | prefill tok/s |
|---|---:|---:|
| 1 | 151.0 | 7,008 |
| 2 | 149.3 | 7,471 |
| 3 | 152.5 | 7,560 |
| 4 | 152.4 | 7,559 |
| 5 | 152.4 | 7,552 |

**Steady state: decode ~152 (−4.7% vs card 160), prefill ~7,555 (−3.6% vs card 7,835).**
The card reproduces; remaining deltas are within chip/protocol labeling (base M4 vs M4 Max,
prompt length).

## Cold vs warm — the earlier confusion

- `e2b_webgpu_1029tok_single_cachebuild.log`: first invocation after ML Drift cache build
  read **prefill 2,882 / decode 125.8** — the one-time cache/compile cost, not steady state.
- `e2b_webgpu_1029tok_runs4_contended.log`: 4-run capture taken while unrelated GPU work ran
  concurrently (browser); run 1 cold 751/75.6, warm 5,785–7,314 / 108–148. Kept for provenance.
- Same warm-vs-cold split as the iPhone numbers: the card is a warm protocol.

## Source-build (yardstick) accelerator evidence

`yardstick_e2b_shortchat_console.log` shows the SwiftPM source-vendored v0.13.1 build also
selects WebGPU→Metal (`Selected adapter: Apple M4 Max ... backend=Metal`, `Initializing
WebGPU-based API`), decoding E2B at 130.4 tok/s cold (`yardstick_e2b_shortchat.jsonl`).
The `Failed to create OpenCL context` INFO line appears on this working GPU path too — it is
a benign probe inside GpuEnvironment (immediately followed by `Created Metal device from
provided device id`), not a fallback to CPU. This retracts the earlier "macOS is CPU-only /
OpenCL fallback" reading in this repo's docs.

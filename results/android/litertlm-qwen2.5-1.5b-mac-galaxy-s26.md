# LiteRT-LM: the official ungated Qwen2.5-1.5B-Instruct bundle on a Mac and on an Android phone

Measured 2026-09-05. Bundle: `litert-community/Qwen2.5-1.5B-Instruct` / `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` (1,597,931,520 B; int8 weights, float activations; KV cache 4096; snapshot 19edb84c). Ungated (no token needed).

## Mac

MacBook Pro M4 Max (128 GB), macOS 27.0. `litert-lm` 0.16.1 from PyPI (`pip install litert-lm==0.16.1`, Python 3.12.13). Command: `litert-lm benchmark <file> --backend {cpu,gpu} -p 512 -d 256 --runs 3` (one warm-up, then three iterations averaged by the CLI; max tokens 4096; compiled-artifact cache on disk; speculative decoding auto).

| backend | prefill tok/s (512) | decode tok/s (256) | init | time to first token |
|---|---:|---:|---:|---:|
| cpu | 839.52 | 48.70 | 0.2894 s | 0.6914 s |
| gpu (Metal via ML Drift) | 2870.98 | 115.87 | 1.2314 s | 0.1983 s |

## Android: Galaxy S26 (SM-S942Q, Snapdragon SM8850, 8 cores, 12 GB, Android 16, security patch 2026-06-05)

Binary: my own `litert_lm_main` built from the v0.16.0 tree (2026-08-25) in `/data/local/tmp/llmbench` with the prebuilt GPU/OpenCL sampler libraries next to it (the GitHub release ships `litert_lm_main` for macOS only). This build ignores `--benchmark*` flags (it ran the built-in 39-token prompt regardless), so the phone rows use a real prompt: 499 raw tokens (529 after the chat template) via `--input_prompt_file`, `--max_output_tokens=256`. Three runs per backend, 45 s idle between runs; medians. The model reached EOS early (108 tokens cpu, 75 tokens gpu), so decode is measured over those lengths. Thermal status after the loop: 1.

| backend | prefill tok/s (529 tokens) | decode tok/s | runs (prefill / decode) | init executor |
|---|---:|---:|---|---:|
| cpu (XNNPACK, 1731/1794 prefill nodes delegated) | 341.8 | 27.76 | 341.83 / 398.88 / 260.58 ; 27.76 / 28.06 / 22.17 | 0.31 s |
| gpu (LITERT_CL, 1409/1409 nodes delegated) | 945.7 | 21.78 | 944.88 / 945.66 / 959.10 ; 22.00 / 21.61 / 21.78 | 3.32 s (program cache built in an earlier run) |

Default built-in prompt (39 tokens), same binary, for reference: cpu decode 27.47 / 27.48 tok/s (51 tokens), gpu decode 21.15 / 19.60 tok/s (97 tokens). NPU: `npu_registry.cc: NPU accelerator could not be loaded` (this bundle has no NPU variant).

## Notes

- Mac and phone protocols differ (CLI synthetic 512/256 vs real prompt, EOS-limited); stated in any post.
- On this q8 bundle the phone GPU prefills 2.8x faster than CPU but decodes slower (21.8 vs 27.8 tok/s). Not investigated; an int4 or GPU-targeted bundle was not tested.
- Raw logs: `litertlm-qwen2.5-1.5b-raw/` (`mac_{cpu,gpu}.txt`, `s26_{cpu,gpu}_p512_{1,2,3}.txt`, `s26_cpu_varA_full.txt` for the ignored-flags run, the prompt file and the loop script).

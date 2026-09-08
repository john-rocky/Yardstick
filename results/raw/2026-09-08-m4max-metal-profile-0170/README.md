# 2026-09-08 — Metal-level decode decomposition, LiteRT-LM 0.17.0 (PyPI) vs MLX, M4 Max

Primary records for [`docs/metal-profile-m4max-0170.md`](../../../docs/metal-profile-m4max-0170.md),
the re-capture of the 2026-07-07 study ([`docs/metal-profile-m4max.md`](../../../docs/metal-profile-m4max.md))
on LiteRT-LM **0.17.0**.

## Instrument

- **LiteRT-LM 0.17.0 has no GitHub release** (latest tagged release is v0.16.1, 2026-08-18); the
  only shipped macOS runtime is the PyPI wheel pair `litert-lm==0.17.0` (CLI) + `litert-lm-api==0.17.0`
  (uploaded 2026-09-04). The executable is `litert_lm/liblitert-lm.dylib` (66 MB, arm64) loaded via
  ctypes; its GPU accelerator is the **statically linked WebGPU (Dawn → Metal) delegate**
  (`RegisterAccelerator … name=GPU WebGPU`, `Selected adapter: Apple M4 Max … backend=Metal`), i.e. the
  same architecture as the v0.13.1 xcframework profiled on 07-07. The dylib also links the ml_drift
  Metal backend symbols, but it is not registered on macOS. Version pin: LiteRT-LM commit before the
  0.18.0 bump (`38fb04e8`, 2026-09-04), LiteRT dependency `9fe5be45` (2026-08-27); the 07-07 v0.13.1
  build pinned LiteRT `a412f505` (2026-06-01).
- Driver: [`scripts/metal-profile/litert_decode_driver.py`](../../../scripts/metal-profile/litert_decode_driver.py)
  (Python API `Engine` → `create_session(top_k=1, temperature=0)` → `run_prefill` → `run_decode_async`;
  per-streamed-chunk timestamps; steady-state rate excludes the first 32 tokens). MLX arm:
  [`mlx_decode_driver.py`](../../../scripts/metal-profile/mlx_decode_driver.py) (mlx-lm 0.31.3, mlx 0.32.2,
  chat template, `temp=0`, `stream_generate`). Same prompt as 07-07.
- Trace: [`profile_cell.sh`](../../../scripts/metal-profile/profile_cell.sh) attaches
  `xctrace record --template 'Metal System Trace'` 3 s after the first decoded token for 6 s, exports
  the `metal-gpu-intervals` table, analysed with `analyze_cell.py` / `gap_structure.py`.
  `.trace` bundles and the XML exports are kept offline (regenerate with the commands in the doc).
- Blocks: [`run_block.sh`](../../../scripts/metal-profile/run_block.sh) — cells interleaved ABAB per
  round, 15 s cooldown + `pmset -g therm` nominal gate + GPU-idle gate before every launch; round 1
  traced, rounds 2–3 untraced. `*.env.txt` records GPU utilization, top CPU consumers and power state
  before each launch (contamination gate).

## Files

- `<cell>_r<N>.json` — one decode run: runtime version, artifact (`hfRepoId`, revision, bytes,
  head-64MB sha), sampler, context/budget, per-token latency series, steady/overall tok/s.
- `<cell>_r<N>.log` / `.env.txt` — driver stdout+stderr / machine state around the run.
- `analysis/*.txt` — `analyze_cell.py` + `gap_structure.py` output for every traced cell.
- `mtl-dump/` — per-pipeline sizes and kernel-source hashes from the `DYLD_INSERT_LIBRARIES`
  shim (`scripts/metal-profile/mtl_dump.m`), the inputs of the hash-keyed ablation. Dispatch
  geometry and the extracted sources stay offline.
- `ablation/runs.{json,md}` — dispatch-skip ablation runs (300 tokens, 512 context; use
  differences only): `gemv_kernel_hashes.txt` lists the kernel-source hashes skipped per
  model/version.
- `gemv_variants.txt` — the K-sliced decode-GEMV baseline and one-variable variants (two runs, both models);
  `gemv_bench.txt` — the earlier two-pass (dequant → fp16 GEMV) vs fused benchmark, which turned
  out to model the prefill/first-token path, kept for the record.
- `summary.{txt,md}` — `summarize_runs.py` over every cell.

Known outlier: an early DeepSeek-int4 0.17.0 ablation baseline read 107 tok/s (7.97 ms) once
against 125–126 in every other run of the same setting; it is in `ablation/runs.json`
(`ds_i4_base`) and was not used. Instrument offset: the 0.13.1 wheel reads 2–7% faster than the
07-07 xcframework numbers on the same cells (`*_v0131` rows vs the 07-07 table).

Cells: `litert_qwen3_4b_int4`, `mlx_qwen3_4b_4bit`, `litert_ds_q8`, `litert_ds_int4`, `mlx_ds_4bit`
(all LiteRT-LM 0.17.0), plus the same-day version ladder `*_v0131`, `*_v0140`, `*_v0150`, `*_v0161`
(litert-lm-api 0.13.1 / 0.14.0 / 0.15.0 / 0.16.1, same driver, same artifacts).

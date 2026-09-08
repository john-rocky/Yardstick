# Metal-level profiling of LLM decode on macOS — method + tooling gotchas

How the [Metal decomposition study](../docs/metal-profile-m4max.md) was captured, and every
tooling trap hit on the way (macOS 27.0 / Instruments 16.0, Apple Silicon). Everything here is
headless-friendly except where noted.

## Capture

**Attach, never launch.** `xcrun xctrace record --template 'Metal System Trace' --launch -- …`
deadlocks LiteRT-LM's WebGPU (Dawn→Metal) runtime: weights upload normally (~590 staging blits),
then the process goes fully idle — no CPU samples, no GPU work — until the time limit SIGKILLs
it. Attaching to a running process mid-decode works:

```bash
litert-mac-verify model.litertlm "<long prompt>" --max-tokens 900 --backend gpu &
sleep 5   # past engine init + prefill
xcrun xctrace record --template 'Metal System Trace' --time-limit 6s \
  --output cell.trace --attach $!
```

Perturbation (traced vs untraced decode tok/s): ≤3% on LiteRT-LM, ~11% on MLX small models —
use the traced-run tok/s for trace-derived metrics. Recording start after `--attach` lags ~2–3 s;
aim the window at the middle of a long decode. Standard measurement hygiene applies
([fairness rules](fairness-rules.md)): AC power, `caffeinate -dims`, browser quit.

## Extract

Export tables headlessly (GUI not needed):

```bash
xcrun xctrace export --input cell.trace --toc          # list tables
xcrun xctrace export --input cell.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output cell_gpu.xml
```

Useful schemas: `metal-gpu-intervals` (per-encoder GPU execution: start, duration, channel,
label, command-buffer id — the workhorse), `time-profile` (CPU samples; proves the launch-mode
hang), `graphics-compiler-activity-intervals` (shader compilation windows),
`gpu-performance-state-intervals` (event-driven, sparse — not a frequency axis).

The XML is id/ref-compressed (`<start-time id="7">…` then `<start-time ref="7"/>`); dereference
before use — `scripts/metal-profile/parse_intervals.py` does this and
`scripts/metal-profile/analyze_cell.py` turns one export into the study's per-cell row
(busy%, encoders/token, gap analysis, in-kernel GB/s).

Metrics derived per cell: **GPU busy** = union of the process's GPU intervals over the captured
span; **idle** = complement, with gap locations read from neighbor-interval labels;
**in-kernel GB/s** = weight bytes / busy-per-token (weights-only lower bound);
**cadence** = encoders / distinct command buffers / blits per token.

## Counters (the missing axis) and what stands in for them

- Headless GPU performance counters are effectively unavailable: `--instrument 'Metal GPU
  Counters'` records only "RT Unit Active" (all zeros for compute); the counter-set picker
  (DRAM bytes, limiters, occupancy) is GUI-only; `--show-recording-options` exposes nothing.
  Consequence: DRAM-byte traffic accounting (read amplification) can't be measured headlessly —
  state it as a caveat rather than guessing.
- `sudo powermetrics --samplers gpu_power -i 500` gives GPU frequency, HW-active residency and
  GPU power — the frequency axis (is the deficit a downclock?) and an energy axis (mJ/token =
  W / tok/s). It needs a real TTY for sudo; run it in a separate terminal alongside a scripted
  run sequence and align by wall-clock (`scripts/metal-profile/pm_driver.sh` writes an epoch
  timeline; trim the first ~40% of each cell window to skip init).
- No-sudo GPU-busy spot check: `ioreg -r -d 1 -c IOAccelerator | grep 'Device Utilization'`
  (validated 86–99% during MLX decode, 0 idle; near-realtime).

## Contamination gate

Bandwidth-bound decode is exquisitely sensitive to *other* work on unified memory: a concurrent
14-core CPU job + an MPS generation job depressed every cell 15–30% while GPU clock and busy%
looked normal. Before any cell: check `ioreg` GPU utilization ≈ 0 **and** no process >300% CPU,
sustained for minutes; re-verify a known cell against its prior number (±10%) before trusting a
session's data.

## Misc

- powermetrics "HW active residency" (500 ms windows, any-engine) reads ~100% where the trace's
  channel-interval union gives 91–93% — different definitions; the trace number is the precise one.
- LiteRT-LM v0.13.1's mac release xcframework runs GPU via WebGPU/Dawn→Metal (accelerator log
  `GPU WebGPU`) under the Engine API. This is distinct from the OpenCL-based path that fails on
  Apple-Silicon macOS in the vendored yardstick build (see
  [MACOS_DESKTOP](../docs/litert-lm/MACOS_DESKTOP.md)) — check the accelerator log line, not
  assumptions, when attributing a backend.
- Weight-GB convention: artifact file size, as in the main report. It over-counts per-token
  reads for models with row-lookup structures (gemma PLE) — flag those cells instead of
  computing a %-roof from them.

## 2026-09-08 additions (LiteRT-LM 0.17.0 re-capture)

Tooling used for [`docs/metal-profile-m4max-0170.md`](../docs/metal-profile-m4max-0170.md);
everything below is headless.

- **Instrument when there is no GitHub release:** LiteRT-LM 0.17.0 shipped only as PyPI wheels
  (`litert-lm` CLI + `litert-lm-api`, the latter carrying `litert_lm/liblitert-lm.dylib`). The
  Python API (`Engine` → `create_session(sampler_config, max_output_tokens)` → `run_prefill` →
  `run_decode_async`) is the same Engine path as the Swift wrapper; the accelerator log line still
  says `GPU WebGPU` (Dawn → Metal). `scripts/metal-profile/litert_decode_driver.py` drives it,
  timestamps every streamed chunk and writes the JSON record; `mlx_decode_driver.py` is the MLX
  twin; `profile_cell.sh` attaches the trace; `run_block.sh` interleaves cells ABAB with cooldown,
  thermal and GPU-idle gates; `summarize_runs.py` tabulates. Check the same-day instrument
  offset before comparing with the 07-07 xcframework numbers: the 0.13.1 wheel measured 2–7%
  faster than the 07-07 xcframework on the same cells (same OS build family, two months apart).
- **Streamed bytes from the flatbuffer, not file arithmetic:**
  `scripts/metal-profile/litertlm_weight_bytes.py` (needs the `tflite` schema package) locates the
  TFLite flatbuffer inside the `.litertlm`, picks the subgraph that references the most constant
  bytes (the decode graph, which also streams the output head) and subtracts gather-only tables
  (`EMBEDDING_LOOKUP`/`GATHER` inputs). Pass that number to `analyze_cell.py`.
- **What the GPU actually runs — shader/dispatch dump:** `scripts/metal-profile/mtl_dump.m` is a
  `DYLD_INSERT_LIBRARIES` shim (Homebrew python is ad-hoc signed, so injection works) that
  swizzles `newLibraryWithSource:` on the concrete `MTLDevice` class and `dispatchThreadgroups:` /
  `dispatchThreads:` / `setComputePipelineState:` on the concrete compute-encoder class. It writes
  every MSL source Dawn/Tint hands to Metal (`lib_NNNN.metal`), logs pipeline creation and every
  dispatch with its grid, and — the useful part — can **skip dispatches by kernel source hash**
  after N dispatches (`MTL_DUMP_SKIP_HASHES`, `MTL_DUMP_SKIP_AFTER`). Skipping a kernel family
  and re-measuring tok/s is a per-family cost ablation that needs no counters; output equality
  against an unmodified run tells whether the skipped kernels' results were consumed.
  `mtl_dump_summary.py` maps pipelines to sources and finds the repeating per-token /
  per-layer dispatch cycle.
  ```bash
  clang -fobjc-arc -dynamiclib -framework Metal -framework Foundation scripts/metal-profile/mtl_dump.m -o libmtl_dump.dylib
  MTL_DUMP_DIR=out MTL_DUMP_DISPATCH_LIMIT=40000 DYLD_INSERT_LIBRARIES=$PWD/libmtl_dump.dylib \
    python scripts/metal-profile/litert_decode_driver.py model.litertlm --prompt ... --max-output-tokens 40 --out x.json
  ```
  Do not commit the extracted MSL (it is the vendor's shader code) nor per-kernel geometry
  tables; commit only the per-pipeline sizes and hashes needed to reproduce the ablation, which
  is what `results/raw/2026-09-08-m4max-metal-profile-0170/mtl-dump/` holds.
- **Trap — the first token is not the steady state.** LiteRT-LM compiles a second set of
  pipelines after the first decode step: prefill and decode token 1 run a longer path (40
  dispatches per layer) than steady decode (29 per layer, different kernels). A dump limited to
  ~3000 dispatches captures only the first path and describes the wrong kernels. Capture
  ≥ 20 000 dispatches (≥ 10 tokens for a 30-layer model) and detect the period on the tail.
- **Trap — short-run tok/s is not the table's tok/s.** 300-token runs at a 512-token context
  decode 5–10% faster than the 1200–1500-token protocol runs (smaller KV cache); use short runs
  only for within-run differences (ablation deltas), never as cell values.
- **Trap — the fp16-intermediate microbenchmark.** A two-pass model (dequantize to an fp16
  buffer, then fp16 GEMV; `experiments/int4-dequant-fusion/gemv_bench.swift`) costs 30+ ms/token
  on M4 Max because the fp16 round trip goes to DRAM; steady decode does not run that path, so
  it says nothing about the decode number. The decode baseline is `gemv_variants.swift`,
  validated against the ablation.
- `sudo powermetrics` needs a TTY password in an unattended session; the frequency axis was not
  re-captured on 2026-09-08 (the 07-07 result — top DVFS state in every cell — was not re-tested).

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

# int8-vs-int4 same-session pair capture — runbook (prepared 2026-07-20, awaiting device)

**Purpose.** The all-int4 answer for Shuangfeng needs int4/int8 uplift *ratios*, and
cross-session ratios are invalid on this device (build/state drift; see
`results/raw/2026-07-13-mlx-variance/README.md`). The 2026-07-20 own-int4 cells
(commit 613f8f2) and the 07-13/14 official-int8 cells are different sessions AND
different iOS builds (24A5355q → 24A5380h), so both sides are re-captured in ONE
session: 4 pairs × (official int8 + own BOCTAV4 int4), interleaved per model.

**Driver.** `scripts/bench_int4_int8_pair_iphone.sh` — one shot:
0. Backs up the **device-only** DeepSeek own-int4 to the Mac (no Mac copy exists).
1. Pushes any zeroed int8 file (all four were tiny-overwritten on device; Mac sources
   are pre-downloaded into the HF cache — see below). **USB required** for the pushes.
2. Waits for the cable to be pulled (charging → thermal fair within ~2 cells;
   battery-temp telemetry lags, so no temp gate — the wait is on ExternalConnected),
   settles 300 s, then runs 8 cells over the WiFi devicectl tunnel, 180 s gaps,
   `--runs 4` each (~45 min total). `SKIP_PHI=1` drops the 3.8B pair (4 GB push).

**Validation & import (after the run).**
- Every run JSON must show `initialThermalState == nominal` and
  `batteryState == unplugged`; re-run any cell that fails (the app JSON is the
  ground truth — this is the 07-20 lesson).
- Ratios: quote warm medians (r2-4) from adjacent cells of the same pair.
- Import policy decision at import time: the int8 cells can either (a) stay
  session-local (ratio evidence only; RESULTS.md keeps the 07-14 warm-matrix int8
  rows) or (b) supersede the 07-14 int8 rows — pick (b) only if re-rendering the
  whole litert-community cluster story on the new build is wanted.
- Post format for the Doc: extend the existing "LiteRT int8 vs int4 — measured
  (supplementary)" pattern (per-model table, parity caveats: DeepSeek −8 pt GSM8K /
  VibeThinker degenerate → int8 stays the parity row; TinySwallow + Phi-4-mini are
  8/8 int4-clean).

**Mac-side sources (pre-downloaded 2026-07-20, `hf download` cache):**
- litert-community/DeepSeek-R1-Distill-Qwen-1.5B — `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm`
- litert-community/TinySwallow-1.5B-Instruct — `TinySwallow-1.5B-Instruct.litertlm`
- litert-community/VibeThinker-1.5B — `VibeThinker-1.5B.litertlm`
- litert-community/Phi-4-mini-instruct — `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm`

int4 Mac sources: tinyswallow/vibethinker `-v2` + phi4 under `~/code/litertlm-convert/out/`;
DeepSeek int4 = device-only until phase 0 backs it up.

# 2026-07-28/29 iPhone protocol re-capture (Gemma-4-E2B, iPhone 17 Pro, iOS 27 24A5380h)

The fairness re-capture of `methodology/CAMPAIGN-gemma4-e2b-full-recapture.md`, iPhone half.
Harness `2026-07-28-agreed-protocol-r3`, ctx forced 2048, unplugged, one block per sitting,
LiteRT-LM as the cross-block anchor, positional rule (runs 2–3 of each launch), RUNS=3 ×
LAUNCHES=4. Spans two raw dirs (the date rolled mid-campaign):
`2026-07-28-gemma4-e2b-protocol/` holds warmup + PROBE proof + blocks A1/A2 (console +
JSON); this dir holds blocks C/N/E. Session boundaries (UTC, for `--since`):
A1/A2 measured cells ≥ `2026-07-28T07:08:53Z` (that dir's `.session-start`); C/E measured
cells ≥ `2026-07-28T16:56:13Z` (this campaign's re-stamp; the one cactus warmup cell at
16:52:12Z is warm-up, not data).

## The prefill proof (defect 1 closed on device)

`console_PROBE_prefill_capped.txt` / `_eos.txt` (07-28 dir): the capped
`long-context-1024-gen256` run reports `promptTokenCount=1106` — identical to the Mac count
on the same tokenizer, non-zero for the first time under a cap — and its warm prefill
(3,550 tok/s) agrees with the EOS task's (3,429) within 3.5%; <!-- derived: 3550/3429 --> counts differ by exactly the
tail-length delta (1106 vs 1081). One instrument now yields the cross-arm prefill column.
<!-- derived: 3550/3429; both traced above -->

## Blocks (all cells thermal-nominal unless stated)

| block | task | warm medians (n) |
|---|---|---|
| A1 depth | decode | litert **58.5** (8) · mlx-PTQ 47.4 (8) · optiq 34.9 (8) |
| A1 depth | prefill (task path) | litert **3,513** · mlx 3,274 · optiq 2,646; prompt_toks 1106/1107/1107 |
| A2 chat | decode | litert **61.1** (8) · mlx 49.1 (**6**) · optiq 36.0 (**6**) |
| C chat | decode | litert **59.4** (8) · cactus-shipped 50.0 (8) · core-ai 47.1 (8) |
| N native | prefill | **3,889** median (n=7, 2,827–3,927) @ctx2048; decode ~60; in-run median 664 MB, teardown 103 MB (separate fields) |
| E energy | J/tok (battery-delta, 600 s sustain, nominal-gated) | see below |

Anchor drift across sittings: litert chat 61.1 (A2) vs 59.4 (C) — 2.8%; <!-- derived: 61.1/59.4 --> cross-block ratios
go through it. <!-- derived: 61.1/59.4 -->

## Energy (defect 5 closed: one session, nominal start enforced, one 600 s window)

n=1 per arm (battery budget; round 2 not captured — disclosed):

| arm | J/tok | avg W | sustained tok/s | Δbatt |
|---|--:|--:|--:|--:|
| mlx-PTQ | **0.164** | 4.83 | 29.4 | 5% |
| cactus-shipped | 0.164 | 4.89 | 29.8 | 5% |
| cactus-uncal | 0.183 | 4.88 | 26.7 | 5% |
| llama.cpp | 0.213 | 4.72 | 22.2 | 5% |
| litert-lm | 0.232 | 9.51 | 41.0 | 10% |
| mlx-OptiQ | 0.415 | 9.06 | 21.8 | 10% |
| core-ai | — | — | — | structurally impossible (below) |

- **The published energy ranking does not survive nominal**: the 07-19 thread numbers
  (LiteRT 0.122 < MLX 0.151) came from cells that started `fair`; <!-- external: 2026-07-19 capture, audited in the campaign doc --> at enforced-nominal MLX
  reads 0.164 vs LiteRT 0.232 (E table above). <!-- derived: rounded from the E table rows --> Same direction as the Mac GPU-energy column. LiteRT still
  holds the highest sustained rate (41 tok/s) — it spends more power for more speed.
- **The thermal gate fired 4 times** (mlx, optiq, llamacpp, coreai cells all deferred at
  `fair` after the preceding 600 s sustain and re-ran at nominal after a 600 s cool) —
  the exact contamination the 07-19 cells published.
- **Core AI cannot run the equalized energy protocol on iOS**: per-call maxTokens 2048
  exceeds its upstream KV cap (1024); the app dies at setup with signal 9 (jetsam),
  `console_E_coreai_1*.txt`. The 07-19 Core AI row existed only because that session gave
  it maxTokens=192 while others ran 2048 — the asymmetry this block removed. Finding, not
  failure (`arm_can_energy`), same class as the depth exclusion.
- Battery-delta quantization at 1% steps means single-cell J/tok carries ~±10-20% error at
  Δ5%; ranks of adjacent rows (mlx vs cactus-shipped) are not resolved at n=1.
- **The E block was interrupted for ~3.9 h between llama.cpp (21:32:50Z) and the two cactus
  cells (01:24:03Z / 01:39:17Z)** — the device left for a recharge (and the aborted Core AI
  attempt sat in between). Every cell, both sides of the gap, was taken after passing the
  nominal thermal gate; the six J/tok cells all record `initialThermalState: nominal`.

## Disclosures

- **A2 mlx/optiq n=6** (< the n≥7 floor, which is our own convention): the 4th launch was
  lost when the device left WiFi range mid-block (console and on-device store agree: 9 runs
  each). litert's anchor cell is n=8. Options: publish at n=6 disclosed, or re-run A2 whole
  in one sitting; do NOT top up cross-sitting.
- **E is n=1 per arm** (was n=2 historically): round 2 was cut on battery budget after the
  gate-cooling doubled the block's wall clock. All seven arms did land round 1 at nominal
  in one session with one window.
- llama.cpp chat/depth were not re-run on iPhone (already n=10–12 on-protocol from
  2026-07-27; per the campaign doc, re-running buys nothing).
- MLX arm lineage: `mlx-community/gemma-4-e2b-it-4bit` @ `2387675…` (July re-upload),
  recorded per-row in `modelRevision`; a checkpoint change vs the 2c3e507 cells the older
  published table used. OptiQ @ `ed4ba3d8…`.
- Cactus bundles sideloaded at the app's `Cactus-Compute__gemma-4-E2B-it/` layout (the
  first push of 07-28 used a wrong nested path and reached no cell; caught in warmup).

# 2026-07-30 pull — E round-2 (iPhone 17 Pro, v0.13, r3 harness)

Round 2 of the energy block, run 2026-07-29 (JST 07-30 session), one cell per arm,
nominal-gated, one 600 s window, ctx 2048, unplugged, battery kept inside 50–90%.
`app-path/` holds the full r3 set of this pull (128 cells); the 12 `task == "energy"`
cells are the complete n=2 energy record (round 1 = 2026-07-28T20:12Z…, round 2 =
2026-07-29T08:29Z… and …T18:25Z…). Round-1 provenance: `../2026-07-29-gemma4-e2b-protocol/`.

## The instrument finding that dominates this data

**On this iOS 27 beta, the battery gauge reports in 5% steps.** Evidence, in-row: across
all 12 cells `batteryDeltaPercent` is exactly 0, 5, or 10 — never anything between —
while sustained decode rates reproduce across rounds within a few percent. A ~600 s cell
whose true drain is ~6–8% therefore quantizes to 5% or 10% depending on phase, and J/tok
(derived from the delta) swings ×~2 between identical runs:

| arm | r1 J/tok (Δ) | r2 J/tok (Δ) | r1→r2 sustained tok/s |
|---|--:|--:|--:|
| litert-lm | 0.232 (10) | 0.125 (5) | 41.0 → 38.1 |
| mlx-PTQ | 0.164 (5) | 0.304 (10) | 29.4 → 29.0 |
| mlx-OptiQ | 0.415 (10) | 0.210 (5) | 21.8 → 22.0 |
| llama.cpp | 0.213 (5) | **nil (0)** | 22.2 → 22.7 |
| cactus-uncal | 0.183 (5) | 0.323 (10) | 26.7 → 27.7 |
| cactus-shipped | 0.164 (5) | 0.160 (5) | 29.8 → 30.4 |

- The only round-pair that agrees (cactus-shipped, 2.7%) is the only one whose two
  deltas landed on the same step. llama.cpp's r2 read Δ0 → the app correctly recorded
  `energyJoules: null` rather than fabricating a figure.
- The per-cell wattages are artifacts of the same quantization (they derive from the
  delta), so they cannot arbitrate.
- Consequence: **n=2 medians do not resolve a cross-arm iPhone J/tok ranking on this
  instrument.** What the data DOES establish at n=2: sustained decode rates (stable,
  ≤4% across rounds, all nominal-gated) and the retention story built on them. Options
  for a rankable J/tok — longer windows (1,200 s+ halves the relative step error),
  many-cell averaging, or dropping the iPhone J/tok column in favor of "tokens per 5%
  step" — are an audit/methodology decision, not made here.
- The Mac GPU-only energy column (powermetrics) is unaffected by this finding.

Core AI: structurally excluded (`arm_can_energy`; KV 1024 vs maxTokens 2048, signal 9 —
see the 07-29 README).

## Bookkeeping disclosures

- Round 2 ran as `LAUNCHES=1 run-block E`, which reuses round-1 console labels
  (`E_<arm>_1`): the three round-1 console files for litert/mlx/optiq in the 07-29 dir
  were overwritten by round-2 content. Round-1 consoles are preserved in git
  (commit 810ab8d); JSON records are timestamped and unaffected. The remaining-arm
  cells used `_2` labels (no overwrite).
- Round 2 was split across two sittings (headline three arms, then a recharge break at
  9%, then llamacpp + cactus ×2) — every cell nominal-gated on both sides of the break.
- The thermal gate deferred 3 cells this round (mlx fair, optiq serious, cactus-shipped
  fair); all re-ran nominal after 600 s cools.

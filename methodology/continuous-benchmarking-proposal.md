# Continuous competitive benchmarking — a strawman to iterate on together

Status: PROPOSAL (2026-07-20). Nothing here is built; it exists so the design conversation
starts from something concrete. The measurement layer it schedules is real and shipped
(see README + `results/raw/*/SUMMARY.txt` audit trails).

## What exists today vs. what "continuous" adds

Today this repo is a **reproducible measurement harness with an audit trail**: campaign
scripts drive devices headlessly, raw JSON lands in `results/raw/`, `render_results.py` /
`generate_charts.py` regenerate every table and figure, CI keeps them in sync, and
per-campaign SUMMARY.txt files record the audit decisions. What it is NOT yet: nobody
runs it unless a human launches it, and nothing watches for regressions.

Continuous = a scheduler + a regression gate + publishing. The three sections below.

## 1. Scheduler (the part that must respect physics)

A benchmark run is not a CI job — the device's physical state is part of the protocol.
The scheduler's job is to encode the rules we currently enforce by hand:

- **Thermal gating**: capture only at `initialThermalState == nominal`; poll and re-run
  otherwise (the loop `bench_mlx_nominal.sh` already implements — promote it from script
  to service behavior).
- **Battery/energy prerequisites**: unplugged, ≤90 %, Auto-Lock off for battery-delta
  runs; these need either a human check-in step or a smart-plug + battery-state probe.
- **Session anchoring** (the key idea, see §2): every scheduled session re-runs one
  fixed anchor cell (e.g. LiteRT Gemma-4-E2B short-chat ×3 nominal) before its payload
  cells. Cross-session ratios are only ever computed through anchors.
- **Serialized device access**: one queue per device; deep-context / sustained tasks
  block the queue for their full duration (the LiteRT teardown-hang and jetsam-spin
  lessons say never overlap).
- Cadence strawman: anchors + a small rotating payload nightly per device; the full
  7-arm × 5-axis matrix weekly; energy (battery-limited) weekly on a rotation.

Runner hardware: one Mac (self-hosted runner) per device pool, devices on Wi-Fi
`devicectl` with USB fallback. Everything the runner executes is the scripts already in
`scripts/` — the scheduler only sequences them.

## 2. Regression detection (institutionalize the anchor method)

Session-to-session device drift is real and large: the same binary and pins measured
LiteRT at 52.7 (07-18) and 60.9 (07-20) — 1.16× — and June→July drifted up to 25 %
(`iphone-session-variance`). Naive time-series alerting on absolute tok/s would fire
constantly and teach everyone to ignore it. Instead:

- **Anchor-normalize**: report each cell as (cell ÷ same-session anchor). Alert when the
  *normalized* value moves > N σ of its history (start: 3 trial-σ, tune on data).
- **Quality gates are absolute**: GSM8K on pinned `evaldata/gsm8k_test.jsonl` with the
  in-repo harness; alert on any drop > 2 points at n=100. Cheap tripwire tier: the 9-item
  on-device quality task per nightly run (it caught the shipped-CQ4 collapse in one run).
- **Artifact identity is part of the record**: HF revision hashes + engine commit pins
  per cell (the Cactus 07-09 silent-bundle-swap and the MLX 07-06 re-upload both changed
  results under unchanged names — sha256 the artifacts, alert on hash change even when
  numbers look stable).
- Failed runs stay in the table with reasons (fairness rule 4) — a new failure IS a
  regression signal (the official-QAT-gguf unloadable row is the model case).

## 3. Publishing

- RESULTS.md + charts regenerate per run (already automated); a static dashboard page
  (per-cell history sparklines from the raw JSONL) is a small addition — no service
  needed, GitHub Pages over committed JSON.
- Weekly digest: the delta table (anchor-normalized moves + new failures + artifact-hash
  changes) — that digest, not the full matrix, is the thing a human reads.

## Android (scoping honestly — currently 0 %)

This repo is Apple-silicon end-to-end. Of the five engines, LiteRT-LM, llama.cpp and
Cactus ship Android support; MLX and Core AI are Apple-only — so the Android matrix is a
*different column set*, not a port of this one. The measurement concepts (anchors,
thermal gating, battery-delta energy, pinned artifacts, one-harness quality) transfer;
the driver layer (`devicectl`, `phys_footprint`, battery API) does not — Android needs
`adb`-based equivalents and its own memory/energy semantics. Proposal: keep this repo
Apple-native, stand up a sibling harness for Android sharing the JSONL schema +
render/chart layer, and let the dashboard merge the two. Sizing that is its own
conversation — flagging it now so the gap is a decision, not a surprise.

## Open questions (the actual agenda)

1. Where does the runner live — our hardware, Google's lab, or both (cross-lab anchors
   would also measure lab-to-lab variance)?
2. Cadence/cost envelope per device per week (battery cycles are the scarce resource).
3. Alert routing: who gets the weekly digest; what threshold pages a human.
4. Android: sibling harness owner and its first engine trio.
5. Model set governance: who decides which models/builds enter the matrix (the
   best-usable-build rule needs an owner once vendors start shipping swaps like the
   07-09 CQ4 replacement).

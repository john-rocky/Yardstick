# Operations — running this repo as a standing benchmark

The runbooks a team needs to operate this repo continuously: release
regressions, adding models/arms/devices, and the parts that stay manual. The
measurement rules live in `methodology/fairness-rules.md` (11 rules + the
working-rule slug table); this file is only *how to run the machine*.

## The core loop

```
./bench release-watch        # upstream releases vs environment.lock.json pins
./bench matrix  <cells>      # standing matrix -> summary -> RESULTS -> LEADERBOARD
./bench regress <cells> --engine <arm> --version <v> --baseline <selector>
```

- Cell files: `matrices/*.cells` (grammar: `matrices/README.md`). CI validates
  them (`cells-valid`).
- Everything a run produces is a schema-v1 JSON under `results/raw/<campaign>/`
  plus its console log — a number without a stored report is not a measurement
  (stored-report-rule).
- `results/summary/*.csv` is the derived accumulation layer (CI: `summary-fresh`);
  `LEADERBOARD.md` renders from it (CI: `leaderboard-check`).
- Regression verdicts persist under `results/regression-reports/<date>-<engine>-<ver>/`
  (report.md + verdicts.json + invocation.txt) and flatten into
  `results/summary/history.csv`.

## Runbook: an engine shipped a release (worked example: LiteRT-LM v0.16.0)

1. `./bench release-watch` — confirms the drift (v0.15.0 pinned, v0.16.0 out).
2. Bump the pin where the arm is acquired:
   - iOS/Mac: `LITERTLM_TAG` in `ios/BenchmarkApp/scripts/bootstrap.sh`, then
     rebuild the app (`bootstrap.sh` + the xcodebuild command in `CLAUDE.md`)
     and the Mac CLI (`scripts/build_yardstick_mac.sh`) **on real hardware** —
     there is no CI build (hosted runners lack the required Xcode).
   - Android: `LITERTLM_TAG=v0.16.0 android/scripts/build_litert_lm_main.sh`,
     push per `android/README.md`.
   - Record artifact sha256s in `environment.lock.json` (the registry); the
     build stamps the *observed* pins into every row (the witness) — if they
     disagree, the row is telling you the truth.
3. Speed/regression capture:
   `./bench regress matrices/release-regression-litert.cells --engine litert-lm --version v0.16.0 --baseline campaign:<last-litert-campaign>`
   - Anchors run first; cross-session verdicts are anchor-normalized
     (`--anchors matrices/anchors.cells` is applied automatically). An anchor
     whose runtime is the engine under test is marked CONFOUNDED — that is why
     `anchors.cells` carries a non-litert anchor per platform.
   - Exit 1 = REGRESSION somewhere; read the report dir before believing or
     disbelieving it (UNRELIABLE = spread too wide, throw out and re-run).
4. Quality gate (absolute, not anchor-relative):
   bump the instrument pin in `tools/litert-mac-verify/Package.swift`, then
   `./reproduce mac gsm8k-e2b-yardstick --regress` (>2-3 pt drop at n=100 is
   real; budget/mode mismatches refuse to score — budget-mode-rule).
5. Commit the new campaign dir + regression report + regenerated summary/
   RESULTS/LEADERBOARD. CI holds it all together.

## Runbook: add a model

1. Speed axis: add a `ModelInfo` per runtime to
   `ios/BenchmarkApp/Sources/Models/ModelCatalog.swift` (id, hfRepoId,
   quantization label — quant-per-arm-rule; the label is part of the result).
   Side-loaded artifacts (`hfRepoId: ""`) need a staging step and `local=1` in
   cells.
2. Display identity: if several orgs ship the same logical model, add a
   pattern to `LOGICAL_MODELS` in `scripts/bench_common.py` (order matters —
   builds that must not pool, like QAT-vs-PTQ, get distinct patterns).
3. Cells: add rows to the relevant `matrices/*.cells`; interleave arms within
   the model block; give >=3B models `cooldown=300`. Run
   `python3 scripts/validate_cells.py` (CI does too).
4. Quality axis (optional per model): run `scripts/parity_gsm8k.py` per arm
   with `GSM8K_MODEL_ID=<catalog id>` and `GSM8K_QUANT=<label>` so the report
   carries the join key (quality<->speed joins on model_id + runtime +
   thinking). Validate answer extraction on a few items first — chat-template
   and thinking-mode defaults differ per arm (budget-mode-rule; checklist in
   the knowledge repo).
5. Android: the cells line IS the registration (HF repo id + `file=` for GGUF
   quant choice). Nothing else to edit.

## Runbook: add an engine arm

1. Swift adapter (`LLMRuntime` + `RuntimeKind` case) — see
   `.github/ISSUE_TEMPLATE/wire-new-runtime.md` and `ios/BenchmarkApp/README.md`.
2. Pin it: `environment.lock.json` arms entry + vendored acquisition in
   `bootstrap.sh` + `stamp_engine_pins.sh` picks it up for row stamping.
3. Cells + `matrices/README.md` runtime list + `scripts/validate_cells.py`
   RUNTIMES set.
4. External-binary arms (own CLI, own timing): copy the Core AI pattern —
   wrapper script + importer emitting schema v1 with the comparability caveat
   in provenance (`scripts/coreai_mac_wrapper.sh` /
   `scripts/import_coreai_llm_benchmark.py`).
5. Quality arm: a `--which` case in `scripts/parity_gsm8k.py`.

## Runbook: add a device

- `devices/<name>.md` (soc, RAM, access, power/screen policy).
- iPhone: `BENCH_UDID` env for the runner; the app id needs the
  increased-memory entitlements (multi-GB models do not fit without them).
- Android: adb-authorize; `BENCH_ANDROID_SERIAL` selects among several.
- `DEVICE_DISPLAY` in `scripts/bench_common.py` for rendering.
- Never average across device classes (same-device-class rule); a new device
  is a new row space, and its first sessions establish its own anchors.

## What stays manual, and why

| step | why it cannot be automated |
|---|---|
| energy cells (`manual=1`) | unplug discipline: battery-delta needs the cable out, <=90% charge, Auto-Lock off — physically a human step |
| iOS/Mac builds | Xcode 27 beta is not on hosted runners; owner hardware builds + signs |
| device staging (side-loaded bundles) | artifacts live outside HF; `copy_to` steps per campaign |
| pin bumps | a version bump is a decision, not an event — release-watch tells you, you decide |

## Known limits an operator must not discover the hard way

- **Core AI arm is best-effort**: PLE models (Gemma-4 E2B/E4B) need the
  unpublished `COREAI_STATIC_INPUTS` engine patch; a clean clone reports
  `unsupported`. The Mac external `llm-benchmark` rows are not
  protocol-identical (own timing, no context budget) and say so in provenance.
- **GSM8K beyond the LiteRT arm still shells to external checkouts**
  (`environment.lock.json` → `external_instruments_not_yet_in_repo`).
- **Cross-session device drift is 16-25%** (iphone-session-variance): never
  compare absolute numbers across sittings without anchors; the differ
  enforces this (INFO-ONLY / anchor-normalized).
- **litert cells can stall ~10 min at teardown** — the iPhone runner's
  gtimeout is load-bearing; keep `CELL_TIMEOUT=3600` for litert cells.
- **The SPM-built Mac yardstick silently lacks four runtimes** — matrix
  runners refuse the `spm-lite` flavor; always build via
  `scripts/build_yardstick_mac.sh`.
- **Android has no warm regime in v1** and no TTFT on llama-cli; every
  deviation is listed in `methodology/android.md`.

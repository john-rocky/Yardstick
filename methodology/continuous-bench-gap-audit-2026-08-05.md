# Continuous-bench gap audit — ① one-command reproduction / ② machine-readable results

2026-08-05. Context: promoting this repo from manual measurement campaigns to a standing
benchmark a team can adopt as infrastructure (the direction Lu proposed in the doc:
continuous, automated competitive benchmarking — LiteRT-LM / llama.cpp / Core AI / MLX /
Cactus × Android / iOS / macOS). This audit covers conditions ① and ② of six
(③ release regression tracking, ④ gate codification, ⑤ Android, ⑥ leaderboard output
depend on these two).

## ① One-command reproduction — verdict: NOT met (4 blocking gaps; pins are the strong part)

Already in place (credit where due):
- **Version pins exist and are argued**: llama.cpp `b8999` release zip, LiteRT-LM tag via
  `LITERTLM_TAG` (bootstrap.sh:24), coreai-models `0.2.0`, mlx-swift-lm pinned **by
  revision** `60bd0d78` with the floating-branch incident documented (project.yml:21–27),
  cactus pinned by commit and built from source (bootstrap.sh:118–134).
- `bootstrap.sh` is one command for the iOS app's vendored deps.
- Per-campaign protocol drivers exist (`bench_gemma4_e2b_protocol_{iphone,mac}.sh`).

Blocking gaps:
1. **The quality axis lives outside the repo.** `parity_gsm8k.py` shells out to four
   user-local checkouts: `~/code/litert-mac-verify/.build/release/litert-mac-verify`
   (line 50), `~/code/coreai/coreai-models/.build/release/llm-runner` (line 173),
   `~/code/litertlm-convert/src_models/...` bf16 weights (line 190), `~/code/cactus`
   (line 240). A clean clone cannot reproduce ANY GSM8K number — including the
   published 86.0 row and the 8/4 thinking pair (89.0/92.0). The Mac instrument
   (litert-mac-verify + the swift-litert-lm fork it builds against) must be vendored,
   submoduled, or published.
   **→ LiteRT yardstick part FIXED 2026-08-05**: litert-mac-verify vendored at
   `tools/litert-mac-verify/`, its engine dependency pinned by exact revision to the
   now-published fork state (`john-rocky/swift-litert-lm@e4e48d9d`, tag
   `yardstick-2026-08-04` — the previously-uncommitted v0.15.0 + ThinkingConfig
   instrument), `VERIFY` made repo-relative (`LITERT_MAC_VERIFY` to override), and
   `--thinking` threaded through `run_litertlm` (the capture-time modification the
   raw README describes had never landed in the script). Clean-clone verified.
   The coreai / bf16-tokenizer / cactus arms remain external
   (`environment.lock.json` → `external_instruments_not_yet_in_repo`).
2. **No Python environment spec.** No requirements.txt / pyproject anywhere; the venvs
   used for published numbers (`.venv-litert015` = litert-lm-api 0.15.0, the 0.14
   contrast venv) are ad-hoc and undocumented in-tree.
   **→ FIXED 2026-08-05**: observed `pip freeze` snapshots committed at
   `tools/python-envs/` for all three venvs behind published rows (`.venv-litert015`,
   `~/venvs/lt0150run`, `~/venvs/lt092run` — the last turned out to be the 0.14
   contrast AND the litert-torch-0.9.1 conversion env in one), each registered in
   `environment.lock.json → python_instruments` with python version + spec path;
   recreate commands in the README there.
3. **User-specific absolute paths in drivers**: HF-cache snapshot paths hardcoded
   (`gsm8k_litert_pip_thinking.py`, `bench_litert_015_mac_speed.sh`), device UDID and
   `DEVELOPMENT_TEAM` baked into scripts. Nothing a second machine can run unedited.
   **→ PARTIALLY FIXED 2026-08-05** (the two named drivers): HF-cache path honors
   `HF_HOME` with `LITERTLM_BUNDLE` override, venv CLIs overridable
   (`LITERT_CLI_014/015`), and the speed driver's OUT no longer points at the
   historical `~/code/apple-silicon-llm-bench` clone — it defaults to a date-stamped
   dir in THIS repo (published campaign dirs can't be clobbered by a re-run).
   **→ iPhone drivers FIXED 2026-08-11** (the three `reproduce`-registered ones):
   `BENCH_UDID` env (or arg 1) with an existence check against `devicectl list`,
   `DEVELOPMENT_TEAM` env-overridable in the protocol build step, and the 0150 /
   resident-ab drivers' OUT date-stamped in THIS repo (resident-ab's had still
   pointed at the historical clone; both had pointed INTO their published campaign
   dirs, so any re-run would have clobbered published raw records). The protocol
   driver's `DEV`/`ECID` were already env-overridable. Older campaign scripts keep
   their hardcodes — they are historical provenance, not entry points.
4. **Pinned defaults drift from published rows.** bootstrap defaults to `v0.13.1` while
   the 8/4 rows were captured at `v0.15.0` via env override — the override is recorded
   only in prose READMEs. There is no lockfile stating "these exact versions produced
   RESULTS.md".
5. (Known, documented) Core AI needs the unpublished `COREAI_STATIC_INPUTS` engine patch —
   that arm cannot be independently reproduced at all today.
6. **No top-level entrypoint.** 20+ campaign scripts; mapping "published table → command"
   requires reading campaign READMEs. The report's "one driver script per platform"
   claim is only loosely true.

## ② Machine-readable results — verdict: half met (per-run records are strong; there is no accumulation layer)

Already in place:
- Device runs emit a rich per-run JSON (device / thermal / battery / memory /
  harnessStamp / per-cell provenance), and rule 5 (hfRepoId + quantization per run) is
  enforced by the app, not by discipline.
- Raw logs are force-included by .gitignore (`!results/raw/**/*.log`) — the "a stored
  log IS the measurement" rule is structural.

Blocking gaps:
1. **Six incompatible schemas in `results/quality/` alone** (field-set census, n=32):
   17× `{tag,n,correct,acc,max_tokens}`, 5× `{mode,n,ok,results}`, 4× rescored variant,
   2× pip015-style (`runtime_build`, `backend`, medians), 2× bundle-style, 1× with
   `correction`. No `schemaVersion` field exists anywhere in the tree.
   **→ Schema v1 LANDED 2026-08-05** (`schema/result.v1.json`, e67ac99); historical
   variants are normalized by `build_summary.py`, not rewritten. Since 2026-08-11
   `parity_gsm8k.py` emits v1 natively (pre-v1 keys kept for old readers).
2. **Engine version is not recorded in results.** Device JSON records the runtime *name*
   and the harness contract (`harnessStamp`) but not the engine build (tag/commit/
   checksum) that produced the row. The v0.13.1→v0.15.0 re-measure had to reconstruct
   build identity from prose. This structurally blocks ③ (regression tracking).
   **→ FIXED 2026-08-05** for rows from new builds — see fix 5 below. Pre-stamp rows
   stay `nil` (honest absence; never backfilled by guessing).
3. **No accumulation layer.** `results/summary/` exists and is EMPTY. RESULTS.md is
   semi-generated markdown (`import_warm_campaign.py`); campaigns are keyed by
   prose READMEs; there is no queryable all-runs table (csv/parquet/sqlite).
   **→ FIXED 2026-08-05** (e67ac99): `build_summary.py` normalizes everything into
   `results/summary/{quality,device-runs}.csv`; regression diffing runs over it.
4. **Charts hardcode their numbers.** `generate_charts.py` (ROWS + GSM8K constants),
   `x_hero_e2b_iphone_chart.py`, `chart_deepcontext_e2b_iphone.py`, the Mac chart —
   all carry literals. Chart↔data consistency is manual discipline; the 8/4 session
   proved it works and that it costs hours per update.
   **→ PARTIALLY FIXED 2026-08-11** (the living README charts): `generate_charts.py`
   no longer transcribes — GSM8K bars read `results/quality/gsm8k_<tag>.json` and the
   iPhone J/tok panel reads the battery-1pct energy runs (all 17 swapped literals
   verified equal to their sources first; the only PNG diff is float-precision bar
   length). Still literal: `chart_deepcontext()`'s memory/decode rows (per-sitting
   curation, needs the campaign audit tables to look up honestly) and the X-card
   scripts — those are frozen renderings of PUBLISHED posts; naive re-derivation
   gives different numbers (e.g. pooled median 61.1 vs the audited 62.1), so they
   stay fixed and each new outward campaign re-derives per the raw-audit rule.
5. **Filename-as-metadata.** `runtime_model_task_timestamp.json` naming varies across
   campaigns; any join is a regex.

## Smallest fix set that flips ① and ② (proposed order)

1. **Lockfile**: `environment.lock` (machine-readable) — every arm's tag/commit/artifact
   checksum, the Python deps, and the env overrides behind each published table row.
   Generated by bootstrap, committed per capture.
2. **Vendor the quality instrument**: bring litert-mac-verify + the swift-litert-lm fork
   into the repo (submodule or `tools/` subtree), make `parity_gsm8k.py` paths
   repo-relative. (Core AI stays best-effort until the engine patch can be published.)
3. **Schema v1**: `schema/result.v1.json`; add `schemaVersion`, `engineVersion`,
   `engineArtifact` to the app's BenchmarkResult and to quality reports; one migration
   script normalizes the 32 quality files + campaign JSONs into
   `results/summary/all-runs.csv` (the empty dir finally earns its name).
4. **Entrypoint**: `./reproduce <platform> <table>` mapping each published table to the
   exact pinned command (starts as a thin wrapper over the existing protocol scripts).
   **→ LANDED 2026-08-05** (`./reproduce` at repo root): 6 tables registered
   (mac gsm8k-e2b-yardstick / mac+iphone e2b-protocol / mac litert-015-speed /
   iphone litert-0150 / iphone litert-resident-ab), each with prerequisite checks and
   the pinned command list; `--run` executes (yardstick) or forwards to the protocol
   script (staged captures). Known gaps are surfaced in the entries themselves —
   e.g. litert-015-speed still hardcodes `~/venvs/lt{092,0150}run` and an OUT dir
   under the historical `~/code/apple-silicon-llm-bench` clone (gaps ①-2/①-3).
5. **Engine version at runtime**: stamp the vendored tag into the app at build time
   (Info.plist → result JSON) so every future row carries its engine identity.
   **→ LANDED 2026-08-05**: `ios/BenchmarkApp/scripts/stamp_engine_pins.sh` (a build
   phase on both targets + bootstrap step 8) reads the **observed** vendored state —
   `git describe` of each Vendored/ clone, the CLiteRTLM binaryTarget zip+checksum
   from LiteRT-LM's Package.swift, a sidecar tag written next to llama.xcframework at
   download — into the `BenchEnginePins` Info.plist key; `BenchmarkResult` records the
   arm-under-test's pin as `engineVersion`/`engineArtifact` (schema v1) and
   `build_summary.py` surfaces both as columns. Observed state, not env defaults,
   because both drift the moment you look: the local v0.13.1 LiteRT clone ships
   **v0.13.0** engine zips (repo tag ≠ binary), and the local coreai-models symlink
   sat at `0.2.1-zoo+static-inputs-patch` while the lockfile said 0.2.0. Mac CLI has
   no Info.plist: set `BENCH_ENGINE_PINS_FILE=ios/BenchmarkApp/Vendored/engine-pins.json`
   (written by the same script). Unreadable pin ⇒ omitted ⇒ row records nil.
   **Verified**: script standalone + build-phase plist mode (on a copied Info.plist),
   `swiftc -typecheck` of the Models layer, `plutil -lint` of the regenerated pbxproj.
   **Build-side verified 2026-08-11**: a real `xcodebuild` Release device build
   (the documented CLAUDE.md invocation) succeeds, the 'Stamp engine pins' phase fires
   in Xcode's own environment before codesign, and the built app's Info.plist carries
   `BenchEnginePins` with the expected observed content (litert-lm repo `v0.13.1` /
   engine zip `v0.13.0`, core-ai `0.2.0+static-inputs-patch`, llama `b8999`,
   mlx `60bd0d78`, cactus honestly `unrecorded`); `codesign --verify` passes on the
   stamped app. **Still unverified**: that a captured device row emits
   `engineVersion` in its JSON (the EnginePins→BenchmarkResult path is typechecked,
   not yet executed on device — the bench iPhone was unavailable 2026-08-11).
   Confirm on the next device session before quoting engine identity from a
   captured row.

Items 1+3 are pure additions (no re-measurement); 2 is a repo-surgery task; 4–5 are
small. ③ (regression automation) becomes a loop over `reproduce` + schema diffing once
these land.
**→ ③ LANDED 2026-08-11**: `scripts/regression_diff.py` diffs two capture sets over
the accumulation layer with the fairness rules as code — quality pairs join on tag
(rule-3 budget/mode mismatches are NOT-COMPARABLE, never scored), device cells join
on (device, runtime, model_id, task, cold/warm) with rule-4 spread gating and
cross-session pairs demoted to INFO-ONLY (June→July drift), exit 1 on a REGRESSION
verdict. The capture side: `./reproduce <platform> <table> --regress` re-runs an
exec-style table into `results/quality/regression/<date>-<table>/` (published
reports can't be clobbered) and diffs against the published rows. Typical release
flow: bump the instrument pin in `tools/litert-mac-verify/Package.swift`, run
`./reproduce mac gsm8k-e2b-yardstick --regress`. `parity_gsm8k.py` now emits
schema v1 natively, stamping the OBSERVED instrument pin (the Package.resolved
revision behind the VERIFY binary + the fork's mac engine zip version/checksum —
verified identical to the lockfile registry) as `engineVersion`/`engineArtifact`;
pre-v1 device rows all carry nil engine_version, so device-side version splits use
`campaign:` selectors until stamped rows accumulate. Verified against real data:
the 86.0→88.0 yardstick pair scores OK (+2.0 pts < 3-pt threshold), a synthetic
80.0 re-capture exits 1, and the 07-27→07-28 campaign diff correctly demotes every
cross-session cell.

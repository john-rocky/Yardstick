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
3. **User-specific absolute paths in drivers**: HF-cache snapshot paths hardcoded
   (`gsm8k_litert_pip_thinking.py`, `bench_litert_015_mac_speed.sh`), device UDID and
   `DEVELOPMENT_TEAM` baked into scripts. Nothing a second machine can run unedited.
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
2. **Engine version is not recorded in results.** Device JSON records the runtime *name*
   and the harness contract (`harnessStamp`) but not the engine build (tag/commit/
   checksum) that produced the row. The v0.13.1→v0.15.0 re-measure had to reconstruct
   build identity from prose. This structurally blocks ③ (regression tracking).
3. **No accumulation layer.** `results/summary/` exists and is EMPTY. RESULTS.md is
   semi-generated markdown (`import_warm_campaign.py`); campaigns are keyed by
   prose READMEs; there is no queryable all-runs table (csv/parquet/sqlite).
4. **Charts hardcode their numbers.** `generate_charts.py` (ROWS + GSM8K constants),
   `x_hero_e2b_iphone_chart.py`, `chart_deepcontext_e2b_iphone.py`, the Mac chart —
   all carry literals. Chart↔data consistency is manual discipline; the 8/4 session
   proved it works and that it costs hours per update.
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
5. **Engine version at runtime**: stamp the vendored tag into the app at build time
   (Info.plist → result JSON) so every future row carries its engine identity.

Items 1+3 are pure additions (no re-measurement); 2 is a repo-surgery task; 4–5 are
small. ③ (regression automation) becomes a loop over `reproduce` + schema diffing once
these land.

# Next-session brief — warm re-capture campaign

> **STATUS: EXECUTED 2026-07-13 (evening session).** All three blockers resolved
> (driver: `scripts/bench_warm_matrix_iphone.sh`; variance: device-state change, June binary
> exonerated — `results/raw/2026-07-13-mlx-variance/`; bundles: LiteRT staged, Core AI 0.6B/1.7B
> blocked by `methodology/coreai-build-regression-2026-07.md`). Warm matrix captured for the
> Qwen3 ladder + E2B on iPhone (`results/raw/2026-07-13-iphone-warm/`) and Mac
> (`results/raw/2026-07-13-mac-litert-warm/`); tables updated (README, RESULTS, cross-framework
> doc). Residual follow-ups: MLX-E2B re-capture after a loader update; CoreML-LLM 0.6B
> re-conversion; the 7 remaining Mac LiteRT artifacts; Core AI 0.6B/1.7B assemble after the
> next macOS/MetalToolchain drop. The "within ~10%" claim below was corrected (0.76×).

Context: the published iPhone/Mac benchmark tables report **cold (first-launch) medians**, which
are not comparable with vendor model cards (warm/steady-state). This was confirmed a real
methodology mistake on 2026-07-13 and disclosed to the LiteRT team in the shared Google doc
(re-run promised, cc fengwuyao/marissaw). This session's job: **re-capture the full matrix warm,
persist it properly, and update the tables to show warm alongside cold.** Do NOT publish any
warm number until the three blockers below are cleared — the last session got burned repeatedly
by claiming from insufficient data.

## Definitions (already written into methodology/fairness-rules.md §2)
- **first-ever**: first launch after install (caches build during the run) — report separately.
- **cold**: fresh process, caches on disk, first generation = the historical tables.
- **warm**: in-process steady state = `--runs 4`, **discard run 1, report median of runs 2-4**.
  This is the vendor-card convention and the headline for cross-vendor comparison.
- Thermal gate: ≥100 s cooldown between cells; verify `initialThermalState == nominal` per run.

## Blocker 1 — driver does not persist jsonl (FIX FIRST)
`scratchpad/.../warm_matrix_tier1.sh` (copied to
`results/raw/2026-07-13-iphone-warm-partial/driver_v1_NO_JSONL_COPYBACK.sh`) captured console
only. The Yardstick app writes device-jsonl to its app container; the driver must copy it back.
Use the pattern already in `scripts/bench_cactus_parity_iphone.sh` (`devicectl device copy from
--domain-type appDataContainer`). Then re-import to `results/raw/` and add an initialThermalState
!= nominal re-run gate. Write a proper `scripts/bench_warm_matrix_iphone.sh`.

## Blocker 2 — unexplained MLX session variance (RESOLVE BEFORE TRUSTING ANY WARM NUMBER)
MLX Qwen3-0.6B, same Release/nominal/cold conditions: **125.8 (June jsonl) vs 177.9 (2026-07-13
console)** = 1.4x apart. Until this is explained (MLX-swift version? model revision? device
thermal history? Debug-vs-Release mislabel?) the warm numbers are not reproducible. Pin
mlx-swift version, re-run 0.6B MLX cold+warm x2 on a cooled device, and diff `device.systemVersion`
/ package.resolved against the June run.

## Blocker 3 — missing model bundles (device prep)
- Core AI: app expects `Documents/CoreAIModels/qwen3_0_6b_{ane,gpu}/` (metadata.json + .aimodel +
  tokenizer/). Only `~/code/coreai/_artifact_archive/qwen3_0_6b_{ios,dynamic}` exist — wrong names,
  not the ane/gpu split. Assemble the ane/gpu bundles (coreai-matrix-completion-task.md) and
  `devicectl copy to` them. Without this, no 3-way iPhone warm table.
- `litert-local/qwen3-1.7b-int4`: not on HF (download error). Side-load the .litertlm, or switch
  the 1.7B LiteRT cell to a litert-community id that downloads.

## Then, and only then
1. Run the full warm matrix (10 models x {Core AI ane/gpu, MLX, LiteRT}) on iPhone with the fixed
   driver; repeat the Mac side via `litert-mac-verify --runs 5` / a warm yardstick path
   (**Mac GPU must be idle — no other cmux session doing GPU work, no browser automation**;
   check with `ps` for mlx/python/coreml first).
2. Rebuild README.md / RESULTS.md / docs/litert-community-vs-mlx-coreai.md tables as **warm
   (in-process, median of runs 2-4) alongside cold**, per fairness-rules §2. Flag any ranking
   flip vs the old cold tables.
3. Update the Google doc's iPhone/Mac tables the same way (browser: the tables live in the
   "Core AI Comparison" tab; the E2B card reconciliation and the CPU-only retraction are already
   posted there).
4. Post ONE consolidated correction comment on the doc's 185.7 / Fengwu status thread with the
   warm numbers and any ranking changes (cc fengwuyao/marissaw). Fix the specific stale claim in
   my earlier "showcase models" reply (Qwen3-0.6B "within ~10% of MLX" — likely wrong: warm is
   ~0.67x if MLX 179 holds).
5. Also still open (non-warm): create the HF collections Lu asked for (13:00 today — GenAI /
   Vision models / Audio, directly on litert-community) and reply to that thread.

## Provenance / what already shipped today (do not redo)
- Mac E2B warm reproduces the card (152/7.5k vs 160/7,835): `results/raw/2026-07-13-e2b-mac-webgpu/`.
- Repo retractions pushed (commit f876a14): cactus-vs-litert{,.ja}.md, MACOS_DESKTOP.md,
  fairness-rules.md §2. Google doc body caveat corrected in place + comment cc yuhuic.
- Partial warm console + this session's blockers: `results/raw/2026-07-13-iphone-warm-partial/`.
- 12 doc-comment replies posted; see memory [[warm-cold-recapture]].

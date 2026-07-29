# 2026-07-28 Mac protocol capture (Gemma-4-E2B, M4 Max)

First protocol-compliant Mac capture: ctx forced 2048, harness `2026-07-28-agreed-protocol-r3`,
xcodebuild yardstick (mlx-swift-lm pinned 60bd0d78 — same loader as the iPhone app), arms
litert / mlx(PTQ) / optiq / llamacpp, n=8 warm per chat/depth cell (positional rule).

- `CHAT_*` / `DEPTH_*`: speed + memory + the cross-arm prefill column (one instrument, every
  arm — the LiteRT prefill-under-cap fix is live; capped runs report promptTokenCount=1106).
- `NATIVE_litert_*.jsonl`: raw `YARDSTICK_NATIVE_OK` console lines from LiteRT's own
  `benchmark()` (card row). Imported to `native_NATIVE_litert_*.json` by
  `scripts/import_native_benchmark.py` — the analyzer reads only the imported files.
- `ENERGY_*.jsonl` (no `_PM_`): **sustained-decode captures, NOT energy cells.** Captured
  through bare `yardstick run --task energy`; macOS has no battery to read, so
  `energyJoules`/`energyJoulesPerToken` are null. Valid for burst-retention (600 s sustained
  decode rate, thermal all-nominal). llama.cpp's two cells abort-trapped in teardown AFTER
  writing their rows — data valid, crash disclosed.
- `ENERGY_PM_*.jsonl`: the actual Mac energy cells, captured through
  `scripts/measure_energy.py` (co-running powermetrics, `energySource: "powermetrics"`,
  ctx 2048 passthrough added 2026-07-28). J/tok comes from these rows only.
  **Scope caveat (measured in the raw powermetrics log, 2026-07-28):** on this macOS 27
  beta, powermetrics reports `CPU Power: 0 mW` on every sample (288/288; no cluster-level
  power lines either), while the GPU sampler reads real values. These J/tok are therefore
  **GPU+ANE energy only**. The bias is not symmetric across arms — llama.cpp does more
  host-side work than the Metal-decode arms, so its row is a lower bound with a larger
  missing share. A cross-arm Mac energy RANKING must disclose this; prefer "GPU energy per
  token" as the column name. Also not directly comparable with pre-beta powermetrics rows
  (whether their combined figure included a live CPU sampler is not recorded).
  **Worse than off: the CPU sampler is flaky per-cell** — it fired on 2 of 8 cells
  (mlx_2 at 16.8 W, optiq_1 at 23.5 W) and read 0 on the other six, so the recorded
  combined `energyJoulesPerToken` mixes two bases across cells of the same arm (mlx read
  0.116 vs 0.215 across its two rounds). The ranked column is therefore
  `energyJoulesPerTokenGPU`, derived uniformly by `scripts/derive_gpu_energy.py`; on that
  basis the two rounds agree to ≤1.5% per arm:
  mlx 0.1164/0.1155 · optiq 0.1369/0.1366 · litert 0.1557/0.1533 · llamacpp 0.1887/0.1879
  J/tok(GPU). Raw powermetrics power lines preserved under `powermetrics-logs/`.
- `COREAI_llm-benchmark.json`: Apple's own CLI (separate harness, not protocol-identical),
  fleet_exports tbl bundle + `--raw-dir` PLE dump + `COREAI_CHUNK_THRESHOLD=1` — the route
  behind the published 82.4/75.9; this session read 81.7/74.0 (n=5).

MLX lineage note (for the write-up): this campaign's MLX arm is the hub PTQ model
`mlx-community/gemma-4-e2b-it-4bit` @ `2387675…` (the July re-upload; recorded per-row in
`modelRevision`). The published Mac 8,505/171.3 row came from a same-ckpt int4-g32 conversion
AND the 2c3e507-era checkpoint — different lineage; do not mix the two in one cell.

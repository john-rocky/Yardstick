# Full warm-matrix prep runbook (Mac now / iPhone on device return) — 2026-07-13

**Hard rules for this campaign (user directive, absolute):**
1. Only verified numbers leave this repo: raw jsonl cross-checked, protocol-conformant
   (initialThermalState==nominal every run, warm n=3, one session window). Any deviation →
   the number is withheld or the deviation is disclosed AND the user decides.
2. Harness verdicts are law: THERMAL_FAIL / EXCLUDED are never overridden.
3. Nothing is posted anywhere (doc/comment/HF/mail/push) without the user's explicit GO on
   the final text.

## Environment repairs applied (2026-07-13 evening — record for reproducibility)

- `~/code/coreai/coreai-models/.venv`: coreai-torch 0.4.0 → **0.4.1** (odiec fix). Use
  `.venv/bin/coreai.llm.export` directly — `uv run` silently resyncs back to the 0.4.0 pin.
- `llm-benchmark` rebuilt with `DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.3.app swift build
  -c release --product llm-benchmark` (June binary ABI-broken vs updated FoundationModels).
- `~/clipconv` (litertlm conversion env) had drifted: transformers 4.46.1 → **5.6.2** (fork
  needs `AttentionInterface` + `cache_utils.LinearAttentionCacheLayerMixin`; 5.1-5.4 lack the
  latter), protobuf → **>=5.27** (`runtime_version` import). litert-torch editable checkout
  itself unchanged since Jun 19 (115a136).
- Qwen3-1.7B has NO registered short-name in coreai export registry — use HF id `Qwen/Qwen3-1.7B`.

## Status of the Core AI unblock (probe running)

Root cause identified for the odiec abort: the updated macOS beta (26A5378j) ships an odiec
that only accepts IR with "AICode versioned locations", which **coreai-torch 0.4.1** emits and
0.4.0 does not. Evidence: `~/code/coreai/leaderboard` exported gemma4-e2b fp16 with
coreai-torch 0.4.1 today and eval-driver compiled + ran it on this build (cache
`~/Library/Caches/coreai-cache/26A5378j/eval-driver/*`, 13:06). Probe chain in flight:
0.4.1 upgrade → re-export qwen3-0.6b dynamic → `coreai-build compile --platform iOS h18p`.
- If PASS → re-export the fleet with 0.4.1: iPhone bundles (qwen3 0.6b ane/gpu, 1.7b gpu)
  AND Mac IRs become compilable again. Also rebuild `llm-benchmark` with
  `DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.3.app` (June binary is ABI-broken vs the
  updated FoundationModels; Beta.3 build launches fine).
- If FAIL → Core AI stays blocked; matrix runs MLX+LiteRT (+June-compiled on-device 4B pair).

## Mac GPU warm bench — artifact checklist

Protocol: same as June Mac table + `--runs 5` warm (litert-mac-verify, greedy, 512 tok,
short-chat prompt; GPU idle: `ps aux | grep -Ei 'mlx|coreml|litert'` clean, no browser
automation during capture). Qwen3 0.6B/1.7B-int4mixed/4B + E2B already done
(`results/raw/2026-07-13-mac-litert-warm/`, `2026-07-13-e2b-mac-webgpu/`).

| Model | LiteRT artifact | source | status |
|---|---|---|---|
| DeepSeek-R1-1.5B | `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | HF litert-community | downloading |
| Phi-4-mini | `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` | HF | downloading |
| TinySwallow-1.5B | `TinySwallow-1.5B-Instruct.litertlm` | HF | downloading |
| VibeThinker-1.5B | `VibeThinker-1.5B.litertlm` | HF | downloading |
| Gemma3-1B | official `gemma3-1b-it-int4.litertlm` (auto-gated) — NOTE June 185.7 was a SELF-conversion; official file is a different artifact → label as new baseline, do not equate | downloading |
| OLMo-2-1B | **HF `mlboydaisuke/OLMo-2-1B-Instruct-LiteRT`** (June-published artifact — preferred over reconversion for fidelity; a fresh local rebuild also exists at `out/olmo2-1b-boctav4`, 888M, env-drift lineage) | HF | downloading |
| SmolLM3-3B | **HF `mlboydaisuke/SmolLM3-3B-LiteRT`** (reconversion cancelled in favour of the June artifact) | HF | downloading |
| Llama-3.2-3B | **HF `mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT`** | HF | downloading |
| Ministral-3-3B | **HF `mlboydaisuke/Ministral-3-3B-Instruct-2512-LiteRT`** | HF | downloading |
| DeepSeek (ours) | HF `mlboydaisuke/DeepSeek-R1-Distill-Qwen-1.5B-LiteRT` (in addition to the litert-community q8 — June table used q8; keep both, label clearly) | HF | downloading |
| Qwen3-1.7B (June row artifact) | June Mac row provenance TBC (int8 vs BOCTAV4) — resolve BEFORE re-running so we re-verify the SAME artifact class | convert | provenance TBC |

**Inventory lesson (user):** before re-converting/re-exporting ANYTHING, check (1) HF
`mlboydaisuke/*` + `litert-community/*`, (2) the iPhone app containers on device return, (3)
`~/Library/Application Support/CoreAIKit/Models/` (ModelStore: gemma-4-E2B-CoreAI,
qwen3-0.6b-CoreAI-official, qwen3.5-0.8B-CoreAI present locally). Only Core AI *IRs* must be
re-made regardless (0.4.0-era IRs don't compile on 26A5378j).

**Source-weight retention (user):** the downloaded fp16 originals in the HF cache are KEPT
(not deleted after exports) — needed if a conversion turns out broken or an iPhone/Mac
variant has to be regenerated.

**Measured-artifact retention (user, 2026-07-13):** any artifact that produced a measurement
is NEVER deleted — it is the provenance of the number, and where a version break made the old
generation unusable (Core AI 0.4.0 → 0.4.1), the NEW-generation artifacts (ct041 IRs +
assembled bundles in `~/code/coreai/coreai-models/exports/*_ct041*`) are upload candidates
for HF (replacing the now-uncompilable June IRs on the `-CoreAI-official` repos). Any HF
upload is an external action → user GO required first. Deletions are limited to: never-measured
intermediates that are provably regenerable AND unusable (e.g., the 0.4.0 re-exports removed
earlier today — the June-measured originals remain in `_artifact_archive/` and on HF).

**Local sweep finds (2026-07-13 late):**
- `~/code/coreai-kit/Examples/SiriAsk/App/Model/qwen_ane/` = complete June-lineage 0.6B ANE
  bundle (mixed-4/8 static, compiled h18p, metadata+tokenizer included, 732M) — stage as-is.
  Gives TWO ANE lineages to measure: June mixed-4/8 vs ct041 pure4bit.
- `~/code/coreai/leaderboard/models/device/gemma4_e2b_..._aotc_h18p` — compiled E2B decode core.
- `~/Documents/Models/litert-lm/` — staged litert dirs incl. LFM2.5-350M / MiniCPM5-1B
  (Lu-model era) + E2B/0.6B/4B/8B community dirs.
- `~/Downloads/CoreML-LLM/conversion/build_qwen3_0_6b_stateful_chunks.py` — the CoreML 0.6B
  chunks builder exists; rebuild is possible locally (optional queue item).

**Core AI on HF — what works on this OS:** pre-compiled `.aimodelc` bypasses odiec entirely →
`mlboydaisuke/qwen3-1.7b-CoreAI-official/ios-gpu/qwen3_1_7b_dynamic.h18p.aimodelc` is the
EXACT June iPhone-GPU artifact and runs as-is (downloaded; PREFER it over the ct041 rebuild
for June comparability — measure both, label lineage). `gemma-4-E2B-CoreAI` also carries an
h18p aotc (unused this campaign). `qwen3-0.6b/-4b-CoreAI-official` hold 0.4.0-era IRs only →
unusable here; ct041 re-exports required (0.6B done; 4B Mac export queued after P2). Fleet
iPhone Core AI compiled bundles are NOT on HF — check device containers on return.

MLX column: `mlx_lm` (June protocol, already steady-state) — re-run for the same models where
an mlx-community repo exists. Core AI column: rebuilt `llm-benchmark --model <IR dir>` after
0.4.1 re-exports (h16c JIT).

**Conversion artifacts are never committed** (repo rule); paths + sha256 recorded in the
results dir README instead.

## iPhone full-matrix prep (execute before device return)

1. **Staging dir** `~/bench-staging/litert-lm/`: every .litertlm above + Qwen3 ladder files
   (0.6B/1.7B-int4/4B already on device) — one `devicectl copy to` per repo-dir on return.
2. **MLX caches**: pre-download to Mac HF cache, then copy blobs/refs into the app container
   (pattern: bench_coreai_iphone.sh `sideload()`): Qwen3 ladder ✓, DeepSeek-R1-1.5B-4bit,
   gemma-3-1b-it-4bit(=Gemma3-1B), Phi-4-mini 4bit, TinySwallow?, Llama-3.2-3B-4bit,
   SmolLM3-3B-4bit (verify repo ids before download).
3. **Gemma-4-E2B MLX**: pre-7/6 revision of `mlx-community/gemma-4-e2b-it-4bit` (pin by
   commit hash via HF API) so the June loader works; stage that snapshot. Warm E2B litert
   n=3 re-run cell also queued (current warm is n=2).
4. **Core AI bundles** (if probe passes): re-export with 0.4.1 + AOT h18p + assemble:
   qwen3_0_6b_{ane,gpu}, qwen3_1_7b_gpu (+ fleet if time). Engine-verify ANE regions per
   `_matrix_assemble.sh::ane_regions` BEFORE staging.
5. **CoreML-LLM 0.6B**: stateful-chunks rebuild — locate conversion pipeline (CoreML-LLM
   repo); if >1 session of work, defer and keep the cell marked pending.
6. **Driver settings**: `BASE_COOLDOWN=180` default, `COOLDOWN_3B=300` for ≥3B cells
   (add per-cell override), litert cells `CELL_TIMEOUT=3600`, `RUNS=4`.
7. **Session checklist (run before first cell)**: device plugged, ≤90% batt, Auto-Lock
   Never, `initialThermalState=nominal`, record iOS build number manually
   (`devicectl device info details | grep Build`), Mac clock synced.
8. **Post-session audit (before ANY table edit)**: for every cell — all-nominal? n(warm)=3?
   single session? THERMAL_FAIL.txt empty or cells excluded? EXCLUDED.txt reconciled?
   Only then import → render → show the user → wait for GO.

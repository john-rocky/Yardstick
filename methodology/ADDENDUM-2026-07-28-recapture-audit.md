# Addendum to NEXT-SESSION-fairness-recapture.md — independent audit, 2026-07-28 ~07:00 JST

A second session audited the campaign/handoff docs against raw data and code, then audited the
execution state (Mac capture in flight, iPhone not started). **Verdict: keep going — do not
restart.** Every Mac cell captured so far is protocol-valid (spot-checked in raw). The items
below correct the handoff where it is stale or wrong. Ordered by urgency.

## 1. ACTIVE — the Mac energy cells being captured right now carry NO energy numbers

`cmd_energy` calls bare `yardstick run --task energy`. On macOS there is no battery to read, so
`ENERGY_*.jsonl` land with `energyJoulesPerTokenDecode = null`, `averagePowerWatts = null`
(verified in `ENERGY_litert_1.jsonl`). Every historical Mac energy row got its joules from
**`scripts/measure_energy.py`** — a `sudo powermetrics` co-running wrapper that stamps
`energySource: "powermetrics"` (verified in `m4max-mlx-swift-…-sustained-energy.jsonl`).

- The running loop (`bench_gemma4_e2b_protocol_mac.sh energy && … coreai`, started 06:40) will
  finish ~08:10. Let it finish or kill it — **do not edit the .sh while it runs** (bash reads
  incrementally). If killed, note `coreai` is chained with `&&` and must be re-invoked.
- Today's ENERGY files are still valid **sustained-decode** captures (usable for Mac
  burst-retention). They are not energy cells. Label or discard accordingly.
- To get Mac J/tok: re-run through `measure_energy.py` (needs an interactive `sudo -v` — ask
  the user). It does **not** yet pass `--context-tokens`; add a passthrough (safe to edit —
  it is not the running script) so the cells carry `contextTokensConfigured: 2048`.
- Cost of the re-run: 4 arms × 2 rounds × 600 s + cooldowns ≈ **90–100 min of Mac wall
  clock**, mains-powered, no thermal/battery budget spent. Schedule it AFTER the current
  `energy && coreai` chain exits, keep the desktop idle (powermetrics measures the whole
  system, and `guard_gpu_only` applies), write outputs into the same campaign dir so
  `energySource: "powermetrics"` distinguishes them from today's decode-only cells. Keep
  today's `ENERGY_*.jsonl` relabeled as sustained-decode (retention) captures, not energy.

## 2. Step 1 is already done except the iPhone proof — do not re-implement

The handoff's "Not done" list is stale. Already in the working tree (uncommitted): the
`MediaPipeRuntime` prefill split (`bench` unconditional, `decodeBench = capped ? nil : bench`),
the app-side energy thermal gate (`YARDSTICK_THERMAL_DEFER` + exit 7 before model load) plus
driver `launch_gated`, `modelRevision` recording, and the `verify_published_numbers.py`
MODEL_HINTS fix. **The fix is also proven working on Mac**: every capped
`DEPTH_litert_*.jsonl` run today reports `promptTokenCount = 1106` with prefill ≈7.4–7.9k
(engine) / 6.6–7.4k (wall) tok/s, stamp `-r3`, ctx 2048 — under the old code these were all 0
(cf. the 24 litert deep runs of 2026-07-27, all `promptTok=0`).

The load-bearing assumption is no longer "inferred": vendored
`ios/BenchmarkApp/Vendored/LiteRT-LM/runtime/core/tasks.cc:439-441` calls
`TimePrefillTurnEnd()` immediately after `executor.Prefill()` returns, before Decode.

**Corrected pass criterion for the one-launch iPhone proof.** The handoff says the capped
gen256 count "must match what an EOS-terminating `long-context-1024` run reports for the same
prompt" — but the two tasks do not share a prompt (same 18 lorem blocks, different tail
instruction), so exact equality is structurally impossible. Pass =
(a) capped `long-context-1024-gen256` reports `promptTokenCount > 0`, ≈1,10x (Mac reads 1106
on the same tokenizer), and (b) its prefill tok/s is consistent with the native row's order of
magnitude. Do not fail the fix on a count mismatch against the other task.

## 3. `install` does not wipe the ML Drift kernel cache

`cmd_install` runs `devicectl device install app` with no uninstall — an update, and the
07-27 README's *measured* finding is that the cache in `tmp/` **survives app updates**. So the
handoff's "install is what makes the next runs first-ever" is wrong, and its "cold-on-disk"
NOTE line is too. Consequences: keep `warmup` mandatory (harmless either way); staged models
also survive install, so stage only what `inventory` reports missing; if a true first-ever
regime is ever wanted, uninstall explicitly first.

## 4. Defect 5's "battery window" bullet is a misdiagnosis — fix it before it reaches the doc

Raw refutes the mechanism: all six 07-19 energy cells ran the same fixed ~600 s sustain
(durations 601–667 s; llama.cpp 608 s vs LiteRT 605 s). `EnergyMonitor` computes
J = packWh × Δbattery% × 3600 at 1% resolution; the 5% vs 10% deltas are an **outcome of
power draw** (llama.cpp ≈9.8 W vs LiteRT ≈4.9 W), not a per-arm window setting. Nothing "ran
twice as long". The only real per-arm asymmetry on 07-19 was Core AI's per-call
`maxTokens=192` (others 2048), which the one-driver re-capture already equalizes.

Also circular: "decode 40.3 vs its own 52.7" is not fair-start evidence — 40.3/52.7 = 0.765 is
literally the published 76% retention ratio (sustained vs burst), and 2 of the 3 short-chat
runs behind 52.7 started `fair` themselves. The fair-start defect stands on the
`initialThermalState` field alone, which is sufficient.

Keep: nominal-gate + one session + re-measure (the gate is already built). Drop the
"10% window ran longer" narrative from the campaign doc §5 and from the driver comment above
the energy branch (edit only after the driver is not running).

## 5. Post-proof doc errata (write-up stage, not measurement stage)

- `agreed-protocol-gemma4.md` L70-73 ("LiteRT-LM reports no prompt-token count on the app
  path … no configuration in which one instrument gives a prefill row for all five arms") is
  false once the fix is proven. Rewrite it; it currently contradicts the campaign doc.
- "a protocol that specifies n ≥ 7" — n≥7 is **our** convention (protocol doc's operational
  rule + docx correction note), not part of Marissa's agreed spec. Phrase it as ours.
- "nine days apart" is 8 days (depth cells) to 10 (OptiQ chat cells are 07-17).
- llama.cpp's two failed energy attempts recorded `null` J/tok with 0% delta, not "0.000".
- v0.15.0 re-checked 2026-07-28: still does not exist; newest is v0.14.0. Thinking row stands.

## 6. Small operational reminders

- `NATIVE_litert_*.jsonl` (Mac) contain raw `YARDSTICK_NATIVE_OK` lines, not JSON — run
  `python3 scripts/import_native_benchmark.py results/raw/2026-07-28-gemma4-e2b-protocol-mac/NATIVE_litert_*.jsonl`
  before analysis, or both audit scripts will not see the native row. (Mac native reads
  8,101 tok/s prefill at ctx 2048, n=7 — vs 7,305 published at the old hardcoded ctx 1056.)
- `results/raw/2026-07-28-gemma4-e2b-protocol/.session-start` was stamped 2026-07-27T21:08Z
  (before any capture). Harmless — the `-r3` harness-stamp filter is what actually isolates
  this campaign — leave it.
- The Mac `cmd_coreai` route (fleet_exports bundle + `--raw-dir` PLE dump +
  `COREAI_CHUNK_THRESHOLD=1`) already answers the handoff's Step 5 "bundle question"; its
  guards refuse to run without the exact bundle and PLE dump. The 9.6 TB `ple_table` error
  was a missing `--raw-dir`, not a broken bundle.
- iPhone 17 Pro currently shows `unavailable` in devicectl — reconnect before Step 2.

## 7. CP1 audit result (supervisor session, 2026-07-28 09:3x) — Mac phase

**PASS, spot-checked in raw:** chat/depth n=8 per arm (positional rule), all `nominal`, all
`-r3`, ctx 2048, engine-vs-wall within ~1%, prefill fix live (`promptTok` 1106/1107/1098 at
depth), `modelRevision` recorded on every row (MLX = `2387675…`, the July re-upload — a
checkpoint change vs the published 2c3e507 cells; label it, as the stage comment already
says). `COREAI_llm-benchmark.json` = 81.7 prompt / 74.0 gen over n=5, tight trials —
consistent with the published 82.4/75.9 lineage and the 0.2.0 re-measure 84.5/74.4.
`analyze_comparability.py` changes (jsonl layout + PTQ/OptiQ arm disambiguation) reviewed and
approved. `phone_chain.sh` reviewed: safe — it stops after the two PROBE cells and re-stamps
`.session-start`; it does not auto-run blocks.

**Still open on the Mac side (do before calling the Mac phase done):**
1. Energy J/tok — §1 above. Not started as of 09:25 (`measure_energy.py` untouched, no
   `energySource: "powermetrics"` rows). Today's 8 ENERGY_*.jsonl are decode-only.
2. `import_native_benchmark.py` over `NATIVE_litert_*.jsonl` — the analyzer skips raw
   `YARDSTICK_NATIVE_OK` text lines, so the native row is invisible until imported.

**When the phone comes back (PROBE judgment):**
- Expect `harness=2026-07-28-agreed-protocol-r3` in the PROBE consoles (app was rebuilt 05:58,
  after the fix). If the stamp reads `-r2`, stop: the device is running the old binary.
- Judge the capped PROBE by §2's corrected criterion: `promptTok > 0` and ≈1,10x (Mac reads
  1106 on the same tokenizer), prefill rate plausible vs the native row. The EOS probe's
  count will be smaller by the tail-length difference — a mismatch there is EXPECTED, not a
  failed proof.

**CP1-b (11:3x): Mac supplementary run audited — PASS, Mac phase now complete.**
`ENERGY_PM_*` verified in raw: `energySource: powermetrics`, ctx 2048, `-r3`, sequential
10:05–11:30 JST, per-domain W recorded per row. The flaky-CPU-sampler finding is confirmed
arithmetically in-row (mlx_2: GPU 19.50 + CPU 16.84 ≈ package 36.35 W; optiq_1 likewise), and
the uniform `energyJoulesPerTokenGPU` column re-derives from in-row fields
(19.5038 W × 602.01 s = 11,742 J ✓). Cross-round agreement ≤1.5%/arm. The GPU-only scope
caveat and `energyBasisNote` must survive into any published Mac energy cell. Note for CP5:
on this basis the desktop inversion claim SURVIVES (MLX 0.116 < OptiQ 0.137 < LiteRT 0.154 <
llama.cpp 0.188 J/tok-GPU), and old untraceable LiteRT 0.154 reproduces at 0.153–0.156;
old MLX 0.090 is superseded, basis unknown — do not reconcile it.

**Phone chain is DEAD without running its probes** (no `console_PROBE_*`, `.session-start`
not re-stamped) and the device is back on USB (`wired`, build 24A5380h — unchanged from
07-27). Before anything else: re-verify the six staged artifacts NOW over USB (cheap; rules
out the chain having died on a staging failure), then unplug, settle, and re-run the chain or
its steps manually. Judge the probes by §7's criterion, including the `-r3` stamp check.

**For the write-up (CP5), noted now:** today's Mac MLX arm is the hub PTQ model
(`mlx-community/gemma-4-e2b-it-4bit`), not the same-ckpt int4-g32 conversion behind the
published Mac 8,505 (handoff §2). Post 4/6's number must come from one regime or the other,
stated — do not mix the two MLX lineages in one cell.

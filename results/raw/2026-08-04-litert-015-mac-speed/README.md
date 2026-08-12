# 2026-08-04 LiteRT-LM 0.14.0 vs 0.15.0 Mac GPU speed delta (Gemma-4-E2B)

**Instrument:** pip CLI `litert-lm benchmark` from both venvs (`~/venvs/lt092run` = 0.14.0,
`~/venvs/lt0150run` = 0.15.0) — the only instrument that runs both sides: v0.15.0 has no
GitHub release assets (no CLiteRTLM xcframework), so the xcodebuild yardstick cannot load it.
This is a **version-delta capture, not a table re-capture** — absolute numbers here are
CLI-instrument numbers and must not be pasted over the xcodebuild table cells.

**Protocol mapping of the 2026-07-28 Mac capture:** official bundle
`litert-community/gemma-4-E2B-it-litert-lm` @ `9262660` unchanged · GPU backend ·
`--max-num-tokens 2048` · `--cache no` · one launch = built-in warmup + 1 measured run ·
**n=8 launches per cell** · 0.14/0.15 interleaved ABAB per round (thermal-drift
cancellation) · 15 s cooldown + `pmset -g therm` nominal gate between launches · strictly
serial, quiet machine. Thread conditions identical across versions (the "4 CPU threads"
commit 7568c3011 is ASR-only and not in 0.15.0 — verified in the 0.15 verification session).

Runner: `scripts/bench_litert_015_mac_speed.sh` · raw: `run_*.log` (48) + `summary.csv`.

## Results (median of n=8, [min..max])

| cell | 0.14.0 | 0.15.0 | Δ | verdict |
|---|---|---|---|---|
| decode tok/s (p1024, g256) | 154.6 [153.9..155.7] | 159.1 [158.8..159.4] | **+2.9%** | real (tight spreads, consistent across all three bands: +3.2/+3.1/+2.9%) |
| decode tok/s (p256, g256) | 155.8 [155.0..156.5] | 160.7 [156.8..161.8] | +3.1% | real |
| decode tok/s (p64, g256) | 156.1 [155.4..157.4] | 161.1 [160.4..162.1] | +3.2% | real |
| prefill tok/s p=1024 | 8,088 [7,742..8,209] | 8,162 [8,045..8,212] | +0.9% | noise — **published p1024 prefill column not invalidated** |
| prefill tok/s p=256 | 2,252 [2,186..2,270] | **5,417** [4,911..5,739] | **+141%** | real — the short-prompt band was overhead-bound on ≤0.14 (p64≈p256≈2.3k flat) and 0.15 unblocks it |
| prefill tok/s p=64 | 2,381 [2,220..2,569] | 2,435 [2,367..2,632] | +2.3% | within spread |
| TTFT p=256 | 0.120 s | **0.053 s** | **−55%** | real (follows the p256 prefill fix) |

Notes:
- 0.14 CLI decode at depth (154.6) ≈ the published xcodebuild cell (154.0) — the two
  instruments agree at the protocol band, which supports quoting the delta.
- The 8/4 quick A/B's "0.14 p256 unstable (475→2290)" did **not** reproduce under n=8
  (0.14 p256 spread 3.7%); treat that as a cold/contended first-run artifact.
- Energy re-verification trigger (handoff item 3: "only if speed moved a lot") **not met**:
  decode +3% and no change at the energy protocol's sustained-decode operating point.
- Cross-check with the same-day released-artifact quality pair (thinking axis):
  OFF 89.0 / ON 92.0 GSM8K n=100 on litert-lm-api 0.15.0 GPU (Python API instrument),
  reports `gsm8k_litert-gemma4-e2b-pip015gpu-{off,thinking}.json`, raw log
  `2026-08-04-litert-pip015-thinking/run.log`.

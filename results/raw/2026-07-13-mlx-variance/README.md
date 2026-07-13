# MLX Qwen3-0.6B session-variance investigation — RESOLVED (2026-07-13)

**Question** (next-session brief, blocker 2): June Release rows show 125.8-133.1 tok/s decode;
2026-07-13 sessions show 166-179 — 1.3-1.4x apart under apparently identical conditions
(Release, initialThermalState=nominal, plugged, iOS "27.0", same model revision). Which is real?

## What was ruled out (evidence)

| factor | verdict | evidence |
|---|---|---|
| build config / thermal / battery | identical | device dict in all jsonl: Release, nominal, charging/full |
| model weights | identical | mlx-community/Qwen3-0.6B-4bit last HF commit 2025-04-28; same 350 MB; identical greedy output |
| mlx-swift / mlx-swift-lm versions | identical | workspace Package.resolved untouched since Jun 10 19:29 (mlx-swift 0.31.4, mlx-swift-lm b95dc780); both June and July builds pin from it |
| app measurement code | identical effect | diff e60aa26→HEAD in runner is metadata-only; and see A/B/C below |
| **app binary (decisive A/B)** | **exonerated** | e60aa26 (Jun 17 tree) rebuilt 2026-07-13 with the June pins and run on the same device: **164.8 cold / 169.1 warm** — matches today's band, not June's |

Sessions on 2026-07-13 (all `--runs 4`, cell = one process, warm = median runs 2-4, nominal):
- morning driver session: 177.9 cold / 179.5 warm
- session A (current tree): 167.7 / 166.4
- session B (current tree, +100 s): 170.3 / 168.5
- session C (**June e60aa26 binary**): 164.8 / 169.1

## Conclusion

- The June 125-133 level is **not reproducible today with any binary** → the cause is device
  state that changed between Jun 19 and Jul 8-13. Prime suspect: an iOS 27.0 **beta build
  update** (current: 24A5355q; the June build number was never recorded — the app logs only
  `systemVersion`, which stayed "27.0"). Consistent micro-signature: June MLX ITL p50 8.20 ms →
  today 5.6-6.0 ms (dispatch-level speedup on a tiny-kernel model), while LiteRT 0.6B
  (~120 tok/s, own GPU stack) and MLX 1.7B (bigger kernels) barely moved.
- Reproducibility TODAY: within-session ±1.5%; across today's sessions ±6% (166-180).
  Cross-runtime comparisons must therefore come from the **same session window**, and the
  re-captured tables must re-measure cold AND warm now rather than mixing eras.
- Consequence for published tables: the June MLX-0.6B cold rows (and the "118.6 vs 119.6 tie",
  already known to be Release-vs-Debug) are superseded; the full matrix is re-captured
  2026-07-13+ with `scripts/bench_warm_matrix_iphone.sh` (device-jsonl retained per cell).

Raw device-jsonl for sessions A/B/C: `device-jsonl/` (session C files are the four
`*T07-5*Z`/`*T08-0*Z` timestamps; June-binary rows lack `streamedChunkCount`, which
distinguishes them). Console logs: `console_*.txt`.

# Mac M4 Max — LiteRT-LM warm verification (2026-07-13)

Companion to the iPhone warm re-capture (`../2026-07-13-iphone-warm/`): the same Qwen3 ladder,
measured warm on the Mac. `litert-mac-verify <model> "<short-chat prompt>" --max-tokens 512
--runs 5 --backend gpu --greedy` — LiteRT-LM v0.13.1 xcframework, WebGPU/Dawn→Metal, engine
kept across runs (run 1 = cold, runs 2-5 = warm; fresh conversation per run). Mac GPU idle
(no other GPU work, no browser automation during capture).

| Model (.litertlm) | cold (run 1) | warm (med r2-5) | prefill warm |
|---|--:|--:|--:|
| Qwen3-0.6B `qwen3_0_6b_mixed_int4` | 247.2 | **267.1** | 563 |
| Qwen3-1.7B int4-mixed (ours, re-export 2026-07-13) | 166.8 | **172.6** | 407 |
| Qwen3-4B `qwen3_4b_mixed_int4` | 106.6 | **109.2** | 111 |

Notes:
- Cold→warm is +2-8% (engine ramp) — the Mac LiteRT numbers were already close to steady state,
  so the June Mac table's protocol was not materially cold-biased (unlike the iPhone tables).
- The June Mac table's "Qwen3-1.7B 115.8" row was the **int8** BOCTAV4 conversion; today's
  172.6 is the **int4-mixed** artifact (the same one as the iPhone 1.7B row) — different
  artifact, not a discrepancy.
- Gemma-4-E2B Mac warm was verified separately the same day against the official card
  (152 decode / 7.5k prefill at 1,029-token prompt): `../2026-07-13-e2b-mac-webgpu/`.
- The other 7 models of the June Mac LiteRT column could not be re-verified tonight — their
  `.litertlm` artifacts are no longer on this Mac (deleted post-capture); re-download /
  re-conversion is queued as follow-up. Their rows stay labelled with the June protocol.

Raw logs: `qwen3_{0_6b,1_7b_int4,4b}_runs5.txt`.

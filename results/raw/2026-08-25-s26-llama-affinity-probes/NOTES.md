# Galaxy S26 × llama.cpp — why the first-session numbers were wrong, with proof

The 2026-08-25-s26-first-session campaign put llama.cpp Qwen3-0.6B at
16.7→15.7→12.8 tok/s (declining, nominal thermal) — slower than the Pixel 8a
(30.1) on the same official b8999 arm64 binary, while every litert-lm cell on
the device gained 2-4x. Six single-run probes, same binary / model
(`unsloth Qwen3-0.6B Q4_K_M`) / prompt / flags (`-c 4096 -n 128 --temp 0
--top-p 1 -st`), varying only CPU affinity and thread count:

| probe | mask (cpus) | -t | Generation tok/s |
|---|---|---|---|
| P1 | f0 (4,5 perf + 6,7 prime) — the arm's standing mask | 4 | 15.0 |
| P2 | none | 4 | 105.6 |
| P3 | c0 (6,7 prime only) | 2 | 89.5 |
| P4 | 3f (0-5 perf only) | 6 | 96.7 |
| P5 | ff (all 8) | 8 | 39.6 |
| P6 | f0 repeat | 4 | 14.4 |

Reading: masks that span the perf/prime cluster boundary collapse ggml's
thread sync (f0 worst at ~1/7th, ff at ~1/3rd); single-cluster masks and no
mask are healthy. `taskset f0` was tuned on the Pixel 8a, where cpu4-7 are
four contiguous mid cores — it is not device-neutral. The SM8850 has no
efficiency cores at all (cpu0-5 @3.63 GHz + cpu6-7 @4.74 GHz), so the
"pin to the big cores" rationale does not apply here; this device runs
unmasked (`BENCH_CPU_MASK=`), recorded per run in conditions.cpuAffinity.

Control probe (litert insensitivity): `litert_lm_main --backend=cpu` on the
same Qwen3-0.6B artifact decodes 25.8 tok/s unmasked vs 27.7 in the masked
campaign — litert-lm manages its own thread pool and does not hit the
cross-cluster pathology, so only the llama.cpp rows needed re-measuring.

Probe consoles: `probe_P*.txt` in this directory. Probes are diagnostics
(one run each, varied conditions) — they are not benchmark rows; the
quotable numbers come from the 2026-08-25-s26-llama-nomask campaign.
Device state during probes: plugged (USB), screen on, post-campaign
(~30 min idle), thermal not gated — the P1/P6 agreement with the campaign's
gated runs (12.8-16.7) shows the effect is the mask, not heat.

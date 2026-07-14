# 2026-07-14 iPhone full warm matrix — audited import set

Single measurement session (iPhone 17 Pro, iOS 27.0 24A5355q, Release,
`--runs 4` protocol: run 1 = cold, runs 2-4 = warm; thermal gate = all runs
nominal). Assembled from three sub-campaigns (raw kept alongside):

- `2026-07-14-iphone-full-warm-r2/` — main 45-cell sweep
- `2026-07-14-iphone-redo/` — thermal re-runs (900 s cooldowns)
- `2026-07-14-iphone-coldpass/` — RUNS=1 true-cold pass for freshly-pushed
  Core AI bundles whose in-campaign run 1 was FIRST-EVER (on-device
  compile/cache build; e.g. deepseek ANE load 117 s, gemma3 GPU run-1 decode
  37% below warm). first-ever ≠ cold per fairness-rules §2.
- (`2026-07-14-iphone-full-warm/` = r1, ABORTED — do not import)

Selection rules applied (scripted): last all-nominal attempt per cell;
first-attempt run 1 dropped for fresh-push cells (cold taken from the nominal
coldpass run instead); litert Qwen3-4B / VibeThinker-1.5B import nominal
run-1 cold samples only (warm never held nominal across 4 attempts each —
runs 3-4 self-heat to fair even after 900 s cooldowns; 2026-07-13 captured
warm for 4B, so this is ambient/session-dependent).

Core AI artifact lineage: June-compiled bundles (ct041 re-exports SIGSEGV on
this device: Mac 26A5378j AICode vs device MPSGraph;
`--min-deployment-version 26.0` rejected "Model requires OS 27.0").
`*-june` control cells = byte-identical June artifacts re-measured for the
cross-session anchor.

NOT imported (no valid data today):
- core-ai/qwen3-4b-ane — all 8 runs fair (r2); post-08:00 re-attempts hit the
  ANECompilerService failure below. June row left in place.
- core-ai/smollm3-3b-ane — first-ever ANE compile crashed 08:10 and every
  retry (incl. post-reboot, post-space-freeing) fails: silent death or ENOENT
  from a poisoned/evicted aned cache entry.
- core-ai/qwen3-0.6b-gpu, core-ai/qwen3-1.7b-ane — no June-lineage bundle
  exists; ct041 incompatible (above). June rows left in place.
- mlx-community/gemma-4-e2b-it-4bit — pre-7/6 snapshot no longer obtainable
  (CDN refuses; device cache purged); post-7/6 revision incompatible with the
  pinned loader (weight-key mismatch).
- core-ai/llama-3.2-3b-{ane,gpu} cold — run 1 was first-ever and the coldpass
  attempt failed (ANE: compile crash; GPU: fair both tries at midday ambient).
  Warm rows imported; cold pending a cooler/ANEC-recovered session.

Device incidents this session (raw evidence in sub-campaign dirs + .ips):
1. ANECompilerService hit cpu/diskwrites resource-limit violations during the
   llama32-3b ANE first-ever compile (07:48); ALL heavy ANE compiles since
   fail (device reboot and freeing ~15 GB did not recover it — suspected
   persistent resource budget). Cached ANE models keep loading fine.
2. Storage pressure (device was near-full during staging) caused iOS to purge
   aned compile caches repeatedly: qwen3-4b (June cache purged -> 201 s
   recompile at 05:44), later vibethinker/llama32 ANE caches evicted between
   their r2 cells and the coldpass.
3. Device rebooted 10:44 (aned reset attempt) — reboot boundary recorded;
   only coldpass/recovery cells ran after it.

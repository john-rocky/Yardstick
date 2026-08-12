# 2026-08-04 same-session resident-memory A/B — LiteRT-LM 0.14.0 vs 0.15.0 (iPhone 17 Pro)

**Question:** is the +17% deep-context resident (published 849 MB → 996 MB in today's
protocol run) a v0.15 regression or cross-session drift? (X-post 2/4 claims LiteRT-LM is
"the only runtime under a gigabyte on both counts".)

**Design:** two installs side by side — `…llmbench014` (vendored v0.14.0, upstream tag +
the two binaryTarget checksum fixes, thinking wiring compiled out) and `…llmbench`
(v0.15.0). Phone freshly REBOOTED, unplugged, long-context-1024 cells, ABAB interleave
n=4 per side, 180 s cooldowns, one launch per cell. Driver:
`scripts/bench_litert_resident_ab_iphone.sh`; raw in `pulled-*/` (A/B window
07:50Z–08:25Z; one pre-A/B 014 warmup cell excluded; the 015 container also carries the
earlier protocol files — excluded by timestamp).

## Answer: cross-session drift, and v0.15 is strictly BETTER

| deep-context (p≈1.1k) | v0.14.0 | v0.15.0 | Δ |
|---|---:|---:|---|
| median resident MB | **1,541** [1,534..1,566] | **984** [980..993] | **−36%** |
| median footprint MB | 812 [760..851] | 704 [691..751] | −13% |

- Today's device state inflates resident across the board (0.14 reads 1,541 vs the July
  session's 849-era numbers; 0.15 reads 984 vs 996 in this morning's protocol run —
  self-consistent). The metric is session-state-sensitive by construction (it counts
  resident pages, a function of system memory pressure).
- Version-wise, v0.15.0 keeps ~560 MB LESS resident than v0.14.0 in the same state, and
  ~110 MB less footprint. The "under a gigabyte on both counts" claim holds on the
  current release in today's state (984/704) — and would have FAILED on 0.14 (1,541).

## Contamination disclosure (why speed from this session is NOT quotable)

The reboot that fixed the memory baseline dirtied the compute baseline: post-boot
background work (indexing etc.) makes prefill bimodal (~1.9k vs ~3.5k tok/s on BOTH
versions) and swings 0.14 decode 49–58 tok/s. Speed quotes stay with the pre-reboot
protocol session (`../2026-08-04-litert-0150-iphone/`, decode spread 0.6%). Memory in
this session is stable (per-version spread <2%) — the A/B's purpose — and the ABAB
interleave makes the version comparison state-symmetric.

Operational lesson: reboot equalizes memory state but contaminates CPU/IO for tens of
minutes — schedule speed cells before a reboot, memory A/Bs after.

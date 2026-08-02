# Pair session 2026-07-23 — resume state (6/8 cells captured)

**Captured (run_ok=4, unplugged, consoles on Mac; run JSONs still ON DEVICE):**

| cell | cold | warm med (r2-4) | when |
|---|--:|--:|---|
| DeepSeek q8 | 29.0 | 29.1 | 18:22 |
| TinySwallow q8 | 29.2 | 29.4 | 19:04 |
| TinySwallow int4 | 46.8 | 45.6 | 19:09 |
| VibeThinker q8 | 30.6 | 29.3 | 19:13 |
| VibeThinker int4 | 46.8 | 45.7 | 19:18 |
| Phi-4-mini q8 | 11.5 | 10.8 | 19:22 |

Adjacent-pair warm ratios so far: TinySwallow **1.55×**, VibeThinker **1.56×**.
Phi q8 = first-ever valid iPhone cell for that model (June OOM was pre-entitlements).

**Remaining 2 cells:** own/Phi-4-mini-int4-BOCTAV4-128 + own/DeepSeek-R1-1.5B-int4-BOCTAV4
(re-run; its 18:27 attempt died to CoreDeviceError 3). Driver:
`run_cells_final.sh` (same dir) — 3 attempts/cell, NO killall.

**To resume when the device is back** (wake screen or replug; if replugged, unplug
again before cells + 300 s settle):
1. `xcrun devicectl device info details --device A6F3E849-…` nudges the tunnel once awake.
2. `run_cells_final.sh`
3. Pull ALL of today's run JSONs from `Documents/results` (24 so far + 8 final) into
   `device-jsonl/`, validate init-thermal=nominal + unplugged per run, compute pair
   ratios (adjacent cells; DeepSeek pair = 18:22 q8 vs the final int4 — same session).
4. Import per `scripts/int4_int8_pair_runbook.md` ((a) session-local vs (b) supersede),
   then the Doc post.

**Tunnel lessons (2026-07-23):** the WiFi devicectl tunnel dies when the phone's
screen sleeps unplugged (Auto-Lock must be set to Never BEFORE unplugging);
`killall CoreDeviceService` while on WiFi makes it WORSE (>5 min outage, caused the
19:26 abort); `devicectl device info details` re-establishes the tunnel only if the
phone is awake. CoreDeviceError 3 / 4016 / 10002 are all tunnel-degradation faces.

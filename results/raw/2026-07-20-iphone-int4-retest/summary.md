# 2026-07-20 iPhone own-int4 retest (single session, all 4 cells)

Re-capture of the four own-int4 BOCTAV4 cells after the device's iOS beta
updated 24A5355q → 24A5380h (cross-session pooling invalid → full same-session
retake). First-ever iPhone capture for Phi-4-mini (its 07-14 cell failed on
devicectl CoreDeviceError 4016 + a truncated push left a 0-byte file on
device; re-pushed 2.43 GiB over USB this session).

Imported cells = pass 3: **unplugged**, WiFi-tunnel devicectl, 120–180 s idle
gaps, init+peak thermal nominal on every imported run (run JSONs record it).
Same binary as 07-14 (Release, `com.example.CoreMLLLMChat` 0.2.0), `--runs 4`,
short-chat, cold = run 1 (first-ever: mldrift caches absent, as on 07-14).

| runtime | model | task | cold (run1) | warm (med r2-4) | n | thermal |
|---|---|---|---|---|---|---|
| litert-lm | own/DeepSeek-R1-1.5B-int4-BOCTAV4 | short-chat | 47.6 | 46.6 | 4 | nominal (run4 peak fair) |
| litert-lm | own/TinySwallow-1.5B-int4-BOCTAV4 | short-chat | 47.2 | 47.0 | 4 | nominal |
| litert-lm | own/VibeThinker-1.5B-int4-BOCTAV4 | short-chat | 46.7 | 45.8 | 4 | nominal |
| litert-lm | own/Phi-4-mini-int4-BOCTAV4-128 | short-chat | 17.0 | 17.4 | 4 | nominal |

Vs the retired 07-14 rows (24A5355q): DeepSeek 41.2/45.4, TinySwallow
42.5/41.9, VibeThinker 45.2/44.9 — the 1.5B cluster sits ~+3–12% higher on
24A5380h; treat as device-state/OS-build shift, not a model change
(see results/raw/2026-07-13-mlx-variance/README.md).

Charging-heat lesson (passes 1–2, EXCLUDED.txt): benching while **charging**
drives the device to thermal fair within ~2 cells even at 60 s gaps
(VibeThinker decode collapsed 46→33 in-cell); `AverageBatt*Temp` from
pymobiledevice3 is a lagging long-window average (sat at 33 °C for 30+ min
after actual state was back to nominal) — gate on the app-recorded
initial/peakThermalState, not on battery-temp telemetry, and bench unplugged.

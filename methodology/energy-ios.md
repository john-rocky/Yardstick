# Energy methodology — iPhone (battery tick-window, `-r4`)

How the iPhone battery-efficiency rows (J/token, average watts) are produced.
This is the on-device counterpart to the Mac `powermetrics` flow in
[`energy.md`](energy.md); the two use **different instruments** and are not
directly comparable in absolute joules — compare runtimes *within a device*,
never one device to another.

> **Two earlier instruments are retired — do not quote or pool their rows.**
> 1. *Fair-start battery-delta* (07-19 headline cells, e.g. LiteRT 0.122 /
>    MLX 0.151): the two headline cells started in thermal state `fair`,
>    violating the nominal-start protocol.
> 2. *Nominal start/end battery-delta* (`-r3`, 07-28/29 rounds 1–2): protocol-
>    clean, but the instrument itself turned out to be broken — see the finding
>    below. Its apparent MLX-first ranking was a quantization artifact.
>
> Rows from the three generations carry different `energySource` /
> harness-stamp values and **must never be pooled or ranked against each
> other**. The audited record of the retirement evidence and the first full
> `-r4` capture is `results/raw/2026-07-30-gemma4-e2b-protocol/README.md`.

## The instrument finding that forced the rebuild

On iOS 27 beta the battery gauge visible to apps reports in **5% steps** —
not the 1% steps this document originally assumed. Evidence (in-row, 12
cells): every `batteryDeltaPercent` came back exactly 0, 5, or 10, while
sustained decode rates reproduced across rounds within a few percent. A ~600 s
run whose true drain is ~6–8% quantizes to 5% or 10% depending on where in a
step it *starts*, so start/end-delta J/tok swings ×~2 between identical runs.
No number derived from a start/end delta on this OS is rankable.

## The tick-window instrument

Instead of differencing start and end, measure **between gauge transitions**:
the moment the level ticks down twice bounds a window whose true energy is
exactly known (each tick = 5% of the pack), independent of where the run
started inside a step.

Implementation ([`EnergyMonitor.swift`](../ios/BenchmarkApp/Sources/Benchmark/EnergyMonitor.swift),
[`BenchmarkRunner.swift`](../ios/BenchmarkApp/Sources/Benchmark/BenchmarkRunner.swift)):

- A 1 Hz poll task records every downward 0.05-step transition as
  `(timestamp, level)`. Any charging event invalidates the window.
- The energy task re-prompts the runtime continuously; the sustain loop exits
  early once **2 complete ticks** are captured (`windowComplete()`), with a
  **1,800 s cap** so a slow-draining arm has room to finish one window.
- `joules = ticks × 0.05 × packWh × 3600`; J/token divides by the tokens
  generated **inside the window** (not the whole run).
- The JSONL is self-auditing: `energySource: "battery-tick-window"`,
  `energyTickCount`, `energyWindowTokenCount`, and `batteryTickTimestamps`
  (transition times, seconds from generation start) — the J/tok figure can be
  re-derived from the record.
- **A cell with fewer than 2 ticks is not a measurement.** Publish no number
  from an incomplete window; re-run the cell.

`packWh` is a per-device constant in `EnergyMonitor.estimatedBatteryWh()`.
For the measured **iPhone 17 Pro (`iPhone18,1`, US eSIM-only)**: **16.5 Wh**
(4,252 mAh at the ~3.88 V nominal implied by the Pro Max teardown). The
global/physical-SIM Pro ships ≈15.5 Wh; absolute joules scale by 15.5/16.5
(≈ −6%) on that variant, re-derivable from the recorded fields.

## Error bar and rank policy (audited 2026-07-30)

Tick spacing within one cell varies ~18% — the gauge's 5% steps are not
equidistant in true energy — so a 2-tick cell carries **±~10%** on J/tok.
Consequences:

- Adjacent arms whose values sit within ±10% of each other are **one
  unresolved cluster**: report the values, refuse the ordering (e.g. the
  Gemma-4-E2B middle trio 0.222 / 0.226 / 0.231).
- Differences outside ±10% are resolved (e.g. LiteRT 0.147 vs MLX 0.182,
  24% apart).
- More ticks per window (longer cap) or repeated cells tighten this; a single
  2-tick cell never resolves a <10% gap.

## Thermal gate (agreed protocol)

Energy cells must start `nominal`. The gate lives **in the app**: at launch it
checks `ProcessInfo.thermalState` and, if not nominal, prints
`YARDSTICK_THERMAL_DEFER` and exits code 7 **before loading a model** —
battery-temperature telemetry lags ~30 min (07-20 lesson), so the app's own
thermal state is the only trustworthy gate. The driver
(`bench_gemma4_e2b_protocol_iphone.sh`, `launch_gated`) retries a deferred
cell up to 3× with 600 s cools; three deferrals = record the skip, never
substitute a number. In production the gate fired repeatedly (4 of 6 first
attempts in the 07-30 block) — cells that would previously have been recorded
throttled.

## ⚠️ Run unplugged — the make-or-break

**USB power charges the phone: no discharge, no ticks, and `chargingDetected`
invalidates the window.** Beyond the instrument, charging heat drives the
thermal state to `fair` within ~2 cells. The driver refuses to launch cells
while `Transport Type: wired` (`require_unplugged`). Two ways to drive:

1. **Wireless `devicectl` (recommended).** Pair over the network
   (`…coredevice.local`), unplug USB, drive with `--device <hostname>`.
   **Wi-Fi must stay on** — treat its idle draw as a small constant shared by
   every runtime. Cellular off.
2. **Launch-then-unplug.** Launch over USB *without* `--console` (returns
   immediately), pull the cable; the app keeps the screen awake and saves the
   JSONL to `Documents/results/`. Reconnect and `copy from` afterwards.

A populated `energyJoules` is itself proof the run discharged on battery;
still verify `device.batteryState == "unplugged"` per JSONL.

## Pre-flight checklist (hold constant across compared runtimes)

- [ ] **Unplugged**, on battery (see above); Auto-Lock = Never.
- [ ] **Low Power Mode OFF** — it throttles and would confound runtimes.
- [ ] **Brightness fixed**, Auto-Brightness OFF.
- [ ] **Battery inside 50–90%.** Start ≤90% (the gauge is sticky near full)
      and stay ≥50%; recharge breaks between cells are fine — the tick window
      is per-cell, and the thermal gate re-qualifies each cell after a break.
- [ ] **Thermal `nominal` at cell start** — enforced by the in-app gate, not
      by operator judgment or battery-temp telemetry.
- [ ] **No other foreground apps**, notifications quiet.
- [ ] Same model **file** and prompt as every other runtime in the comparison.

## What the number is and isn't

- **Whole-system, not chip-only.** Display, Wi-Fi idle, background OS work —
  everything the battery powered during the window. Good for ranking runtimes
  under identical conditions; not an "inference-only" joule count. (The Mac
  `powermetrics` rows share this caveat.)
- **The ±10% cell error dominates**; it swamps the ~1% pack-voltage
  uncertainty, so chasing a more precise Wh constant is not worth it.
- **Pack capacity drifts with battery health.** The JSONL records the
  hardware id and levels so degraded-pack captures can be flagged.
- **The answer can differ by OS.** On this data the iPhone crown (LiteRT) and
  the Mac GPU-only crown (MLX) disagree — neither result generalizes to the
  other device.

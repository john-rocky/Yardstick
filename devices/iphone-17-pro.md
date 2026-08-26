# iPhone 17 Pro

Primary iPhone reference. The newest A-series silicon at the time of
writing; sets the ceiling for what is possible on iPhone today.

| Field | Value |
|-------|-------|
| Chip | Apple A19 Pro (TBD — confirm at first measurement) |
| Neural Engine | 16-core ANE |
| RAM | TBD |
| iOS version tested | iOS 26 |
| Storage class | Internal NVMe |
| Power | Plugged in, brightness fixed (see methodology) |

## Notes

- Newest A-series silicon — sets the ceiling for what is possible on iPhone today.
- Largest RAM budget in the iPhone lineup at release — most permissive jetsam threshold.
- Apple Intelligence-capable: `PrivateFoundationModelsApple` / Apple FM
  runs through the same harness once iOS 26.x ships the relevant API.

## Build

```sh
cd ios/BenchmarkApp
./scripts/bootstrap.sh
open BenchmarkApp.xcodeproj
# set Team in Signing & Capabilities, select iPhone 17 Pro, ⌘R
```

## Results

See the runtime/model rows in [`../RESULTS.md`](../RESULTS.md) filtered to `iPhone 17 Pro`.

## Thermal admission for charts (2026-08-26)

Plugged + warm ambient pins this device's reported thermal state at "fair"
regardless of load — on 2026-08-26 it would not report nominal even powered
off, so the state label cannot discriminate throttling there. The chart
filter therefore admits, besides nominal starts, fair-start sessions whose
in-session mlx anchor (Qwen3-0.6B warm) matches the newest all-nominal
anchor within 5% (`ios_admissible_campaigns` in scripts/generate_charts.py).
Evidence, same day: fair-at-full-speed (anchor 177 vs nominal-era 171.6;
litert 122 and LFM 68.3 reproducing their nominal-session values exactly)
and fair-with-throttling (anchor runs at 127-159 — rejected by the gate).
Sessions without an anchor stay nominal-only. The LEADERBOARD is unfiltered
as before (its rows carry thermal states per run).

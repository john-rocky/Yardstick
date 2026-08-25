# Galaxy S26 (SM-S942Q)

- SoC: Qualcomm Snapdragon SM8850 (Adreno GPU) — the flagship counterpart to
  the Pixel 8a's mid-range Tensor G3. Never pool or average rows across the
  two devices (same-device-class rule); each carries its own anchors.
- CPU topology (read on-device 2026-08-25): cpu0-5 @ 3.63 GHz (performance)
  + cpu6-7 @ 4.74 GHz (prime). The lane's standard `taskset f0` mask lands on
  cpu4-7 = 2 performance + 2 prime here (on Pixel 8a it means the 4 mid
  cores) — the mask is recorded per run in `conditions.cpuAffinity`, so rows
  stay comparable within the device either way.
- RAM: 12 GB
- First measured: 2026-08-25 (this repo's first flagship Android row space);
  Android version + security patch are recorded per run in the JSON
  (`device.systemVersion` / `device.securityPatch`).
- Access: adb over USB, serial RFGL80R6A6H. USB debugging authorized.
- Power/screen policy for speed cells: USB attached, screen on
  (`conditions.screen: "on-usb"`) — same protocol as pixel-8a.md. Energy
  cells: manual, unplugged (methodology/android.md).
- NPU: physically present (Hexagon) and known to work via a separately built
  litert_lm_main with the Qualcomm dispatch library — but that binary is not
  this repo's pinned artifact, and the LiteRT NPU path is Early Access
  Program only, so LiteRT rows here are cpu/gpu and the NPU row stays n/a
  with that reason.
- ⚠ llama.cpp arm, first-session caveat (2026-08-25): do not quote this
  device's llama.cpp numbers until diagnosed. Qwen3-0.6B declined
  16.7→15.7→12.8 tok/s across three back-to-back nominal runs (Pixel 8a:
  30.1 — a flagship losing to a mid-range on the same official b8999 arm64
  binary), and DeepSeek-R1-1.5B landed at Pixel parity (9.1 vs 9.2) while
  every litert-lm cell on this device gains 2-4x. Suspects: scheduler /
  taskset interaction on this core layout, or ggml CPU-variant dispatch.
  The rows stay (stored-report-rule) with their conditions recorded.
- Build/run: `android/README.md` (engine acquisition, driver, campaign runner).

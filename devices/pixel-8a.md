# Pixel 8a (akita)

- SoC: Google Tensor G3 (1× Cortex-X3 + 4× A715 + 4× A510; Mali-G715 GPU)
- RAM: 8 GB
- First measured: 2026-08 (Android lane v1); Android version + security patch
  are recorded per run in the JSON (`device.systemVersion` /
  `device.securityPatch`) — the OS moves under a long campaign, the records
  carry the truth.
- Access: adb over USB. USB debugging must be authorized (the on-device dialog
  reappears after revocation); keep the screen unlocked for the first connect.
- Power/screen policy for speed cells: USB attached, screen on
  (`conditions.screen: "on-usb"`) — the Android counterpart of the iPhone
  plugged-speed protocol. Energy cells: manual, unplugged (methodology/android.md).
- NPU: not reachable (LiteRT NPU path is Early Access Program only) — LiteRT
  rows are cpu/gpu; the NPU row stays n/a with that reason.
- Build/run: `android/README.md` (engine acquisition, driver, campaign runner).

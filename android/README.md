# Android lane — adb-driven CLI benchmarks (Pixel 8a first)

Engines with Android support: **LiteRT-LM** (cpu/gpu) and **llama.cpp** (cpu).
MLX and Core AI are Apple-only (n/a rows); Cactus is a phase-2 slot.
Measurement semantics and every disclosed difference from the Apple lane:
`methodology/android.md`. Device notes: `devices/pixel-8a.md`.

## Prerequisites (one-time, host Mac)

- `adb` (Android platform-tools), device with USB debugging authorized
- `bazelisk` (`brew install bazelisk`) — respects the checkout's `.bazelversion`
- Android NDK r28b+ (Android Studio SDK manager); auto-detected under
  `~/Library/Android/sdk/ndk`, or set `ANDROID_NDK_HOME`
- `python3 -m pip install huggingface_hub` (model downloads; gated repos need
  `huggingface-cli login`)

## Engine acquisition (per release)

```bash
LITERTLM_TAG=v0.16.0 android/scripts/build_litert_lm_main.sh   # bazel source build, 10-20 min first time
android/scripts/fetch_llama_android.sh                         # official b8999 android tar
```

- LiteRT-LM releases ship **no Android binary** (verified v0.14–v0.16), so this
  is a per-release source build into `android/bin/<tag>/` (binary + GPU `.so`
  set). On a macOS host the build needs `--enable_platform_specific_config`
  (encoded in the script; upstream google-ai-edge/LiteRT-LM#3247).
- llama.cpp ships an official `android-arm64` artifact at the same `b8999` tag
  the Apple arm pins — no build (official-sdk rule). CPU-only.
- Both record sha256 pins in `android/engine-pins.json`; every result row
  carries `engineVersion`/`engineArtifact` from there.

## Push runtime to the device

```bash
adb shell mkdir -p /data/local/tmp/llmbench
adb push android/bin/v0.16.0/litert_lm_main /data/local/tmp/llmbench/
adb push android/bin/v0.16.0/*.so /data/local/tmp/llmbench/        # GPU backend
adb push android/bin/llama-b8999/llama-cli /data/local/tmp/llmbench/
adb push android/bin/llama-b8999/llama-bench /data/local/tmp/llmbench/
adb push android/bin/llama-b8999/*.so /data/local/tmp/llmbench/    # shared-lib build
adb shell chmod +x /data/local/tmp/llmbench/litert_lm_main /data/local/tmp/llmbench/llama-*
```

Models are HF-downloaded on the host and pushed on first use by the driver.

## Run

```bash
# one cell
python3 android/bench/run_cell.py --runtime litert-lm --backend gpu \
    --model-id litert-community/Qwen3-0.6B --task short-chat --runs 3 \
    --out results/raw/$(date +%F)-android-smoke/app-path-android

# a cells file (anchors first, interleaved rounds, thermal gate)
CAMPAIGN=$(date +%F)-android python3 android/bench/run_campaign.py \
    matrices/release-regression-litert.cells

# or the top-level driver (runs mac/iphone/android lanes that appear in the file)
./bench matrix matrices/release-regression-litert.cells --platform android
```

Records land as schema-v1 JSON under `results/raw/<campaign>/app-path-android/`
(the `app-path*` glob build_summary.py already reads) with the raw console log
next to each record (stored-report-rule). `python3 scripts/build_summary.py`
then folds them into `results/summary/device-runs.csv` with `platform=android`,
and the leaderboard renders the android section automatically.

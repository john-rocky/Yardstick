#!/usr/bin/env bash
# Build litert_lm_main for android_arm64 at a pinned tag.
#
# LiteRT-LM releases ship no Android binary (verified v0.14.0/v0.15.0/v0.16.0:
# assets are two xcframeworks + litert_lm_main.macos_arm64 only), so the Android
# arm is a per-release source build. The Maven litertlm-android AAR is the
# official artifact but exposes only the Kotlin API — no CLI — so using it would
# mean an instrumented APK harness (recorded as a phase-2 option in
# methodology/android.md).
#
# Prerequisites (owner machine, one-time):
#   - bazelisk (brew install bazelisk) — respects the checkout's .bazelversion
#   - Android NDK r28b+ (Android Studio SDK manager)
#   - ~10-20 min and a few GB of bazel cache on first build per tag
#
# Env:
#   LITERTLM_TAG      tag to build (default: v0.16.0)
#   ANDROID_NDK_HOME  NDK path (default: newest under ~/Library/Android/sdk/ndk)
#   LITERTLM_SRC      scratch clone dir (default: ~/.cache/apple-silicon-llm-bench/litert-lm-<tag>)
#
# NOTE: this intentionally does NOT build inside ios/BenchmarkApp/Vendored/LiteRT-LM.
# That checkout is the iOS pin; bazel output there would pollute the vendored state.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${LITERTLM_TAG:-v0.16.0}"
SRC="${LITERTLM_SRC:-$HOME/.cache/apple-silicon-llm-bench/litert-lm-$TAG}"
OUT="$REPO_ROOT/android/bin/$TAG"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  ANDROID_NDK_HOME="$(ls -d "$HOME"/Library/Android/sdk/ndk/* 2>/dev/null | sort -V | tail -1 || true)"
fi
[[ -n "$ANDROID_NDK_HOME" && -d "$ANDROID_NDK_HOME" ]] || {
  echo "ERROR: ANDROID_NDK_HOME not set and no NDK under ~/Library/Android/sdk/ndk" >&2; exit 1; }
export ANDROID_NDK_HOME
echo "== NDK: $ANDROID_NDK_HOME"

command -v bazelisk >/dev/null || { echo "ERROR: bazelisk not installed (brew install bazelisk)" >&2; exit 1; }

if [[ ! -d "$SRC/.git" ]]; then
  echo "== cloning LiteRT-LM $TAG -> $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 --branch "$TAG" https://github.com/google-ai-edge/LiteRT-LM.git "$SRC"
fi
OBSERVED="$(git -C "$SRC" describe --tags --always)"
echo "== source at: $OBSERVED"
[[ "$OBSERVED" == "$TAG" ]] || echo "WARN: observed tag $OBSERVED != requested $TAG (recorded as-is in pins)"

echo "== bazel build (android_arm64) — first build takes 10-20 min"
# --enable_platform_specific_config: required on a macOS host — without it the
# rules_rust tool crates (thiserror etc.) fail with E0463 "can't find crate for
# thiserror_impl" because .bazelrc's `build:android` disables the host's
# build:macos config. Upstream: google-ai-edge/LiteRT-LM#3247.
(cd "$SRC" && bazelisk build --config=android_arm64 --enable_platform_specific_config \
  //runtime/engine:litert_lm_main)

mkdir -p "$OUT"
cp -f "$SRC/bazel-bin/runtime/engine/litert_lm_main" "$OUT/litert_lm_main"
chmod +x "$OUT/litert_lm_main"

# GPU backend needs the accelerator .so set next to the binary on device
# (LD_LIBRARY_PATH=.). Ship whatever this tag provides.
SO_SRC="$SRC/prebuilt/android_arm64"
if [[ -d "$SO_SRC" ]]; then
  cp -f "$SO_SRC"/*.so "$OUT/" 2>/dev/null || echo "WARN: no .so files in $SO_SRC (GPU backend unavailable)"
else
  echo "WARN: $SO_SRC missing — GPU backend unavailable for this tag"
fi

(cd "$OUT" && shasum -a 256 litert_lm_main *.so 2>/dev/null | tee SHA256SUMS)

# Record the observed pin (registry = requested tag, witness = observed describe + sha256),
# same registry/witness convention as environment.lock.json engine_version_stamping.
python3 - "$REPO_ROOT" "$TAG" "$OBSERVED" "$OUT" <<'PY'
import json, os, sys, hashlib
root, tag, observed, out = sys.argv[1:5]
pins_path = os.path.join(root, "android", "engine-pins.json")
pins = json.load(open(pins_path)) if os.path.exists(pins_path) else {}
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
entry = {"requested_tag": tag, "observed": observed,
         "litert_lm_main_sha256": sha(os.path.join(out, "litert_lm_main")),
         "so_files": {f: sha(os.path.join(out, f))
                       for f in sorted(os.listdir(out)) if f.endswith(".so")}}
pins.setdefault("litert-lm", {})[tag] = entry
json.dump(pins, open(pins_path, "w"), indent=2)
print(f"pins updated: {pins_path}")
PY

echo "== done: $OUT"

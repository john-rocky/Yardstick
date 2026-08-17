#!/usr/bin/env bash
# Fetch the OFFICIAL llama.cpp Android binary at the same tag the Apple arm pins.
#
# b8999 (environment.lock.json -> arms."llama.cpp".tag) ships
# llama-<tag>-bin-android-arm64.tar.gz in its GitHub release — an official
# artifact (fairness rule: prefer the official runtime SDK), so no NDK build.
# The official Android build is CPU-only (no OpenCL/Vulkan): llama.cpp Android
# GPU rows are n/a in v1, disclosed in methodology/android.md.
#
# Env:
#   LLAMA_TAG  release tag (default: b8999 — keep in lockstep with the Apple arm)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${LLAMA_TAG:-b8999}"
OUT="$REPO_ROOT/android/bin/llama-$TAG"
TAR="llama-$TAG-bin-android-arm64.tar.gz"
URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG/$TAR"

mkdir -p "$OUT"
if [[ ! -x "$OUT/llama-bench" ]]; then
  echo "== fetching $URL"
  curl -fL --retry 3 -o "$OUT/$TAR" "$URL"
  tar -xzf "$OUT/$TAR" -C "$OUT" --strip-components=1
  rm -f "$OUT/$TAR"
fi
[[ -x "$OUT/llama-bench" && -x "$OUT/llama-cli" ]] || {
  echo "ERROR: llama-bench / llama-cli not found after extraction — inspect $OUT" >&2; exit 1; }

(cd "$OUT" && shasum -a 256 llama-bench llama-cli | tee SHA256SUMS)

python3 - "$REPO_ROOT" "$TAG" "$OUT" <<'PY'
import json, os, sys, hashlib
root, tag, out = sys.argv[1:4]
pins_path = os.path.join(root, "android", "engine-pins.json")
pins = json.load(open(pins_path)) if os.path.exists(pins_path) else {}
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
pins.setdefault("llama.cpp", {})[tag] = {
    "artifact": f"llama-{tag}-bin-android-arm64.tar.gz",
    "llama_bench_sha256": sha(os.path.join(out, "llama-bench")),
    "llama_cli_sha256": sha(os.path.join(out, "llama-cli")),
}
json.dump(pins, open(pins_path, "w"), indent=2)
print(f"pins updated: {pins_path}")
PY

echo "== done: $OUT"

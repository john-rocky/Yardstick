#!/usr/bin/env bash
# Record the OBSERVED engine pins — what is actually vendored, not what env vars asked
# for — so every benchmark row can carry the engine build that produced it
# (schema/result.v1.json engineVersion / engineArtifact; continuous-bench condition 3).
#
# Two outputs, recomputed on every invocation:
#   1. Vendored/engine-pins.json          — for the Mac yardstick CLI
#                                           (BENCH_ENGINE_PINS_FILE points here)
#   2. built product's Info.plist         — key `BenchEnginePins`, when invoked as an
#                                           Xcode build phase (TARGET_BUILD_DIR +
#                                           INFOPLIST_PATH set; app target only)
#
# Why observed state: a Vendored/ clone survives a changed LITERTLM_TAG (bootstrap skips
# existing dirs), and the binaryTarget zip inside LiteRT-LM's Package.swift can lag the
# repo tag (the v0.13.1 checkout ships v0.13.0 engine zips) — so the repo tag and the
# engine artifact are recorded separately, both read from the artifacts themselves.
# A pin that cannot be read is omitted (the row records nil), never guessed.
set -euo pipefail

cd "$(dirname "$0")/.."   # ios/BenchmarkApp
V="Vendored"
mkdir -p "${V}"

ENTRIES=""
add_pin() {  # add_pin <runtime-id (RuntimeKind rawValue)> <version> [artifact]
    local id="$1" ver="$2" art="${3:-}"
    local e="\"${id}\": {\"version\": \"${ver}\""
    if [ -n "${art}" ]; then
        e="${e}, \"artifact\": \"${art}\""
    fi
    e="${e}}"
    if [ -n "${ENTRIES}" ]; then
        ENTRIES="${ENTRIES}, ${e}"
    else
        ENTRIES="${e}"
    fi
}

# litert-lm — repo pin from the vendored clone's git state; the engine binary is the
# CLiteRTLM binaryTarget zip its Package.swift resolves (version + checksum from there).
if [ -d "${V}/LiteRT-LM/.git" ]; then
    LITERT_VER="$(git -C "${V}/LiteRT-LM" describe --tags --always 2>/dev/null || true)"
    LITERT_URL="$(grep -F 'CLiteRTLM.xcframework.zip' "${V}/LiteRT-LM/Package.swift" 2>/dev/null | grep -o 'https[^"]*' | head -1 || true)"
    LITERT_SUM="$(grep -F -A1 'CLiteRTLM.xcframework.zip' "${V}/LiteRT-LM/Package.swift" 2>/dev/null | sed -n 's/.*checksum: *"\([0-9a-f]*\)".*/\1/p' | head -1 || true)"
    LITERT_ART=""
    if [ -n "${LITERT_URL}" ]; then
        ZIPVER="$(printf '%s' "${LITERT_URL}" | sed -n 's#.*/download/\([^/]*\)/.*#\1#p')"
        LITERT_ART="CLiteRTLM.xcframework.zip@${ZIPVER}"
        if [ -n "${LITERT_SUM}" ]; then
            LITERT_ART="${LITERT_ART} sha256:${LITERT_SUM}"
        fi
    fi
    if [ -n "${LITERT_VER}" ]; then
        add_pin "litert-lm" "${LITERT_VER}" "${LITERT_ART}"
    fi
fi

# llama.cpp — a release xcframework carries no version marker inside; bootstrap writes a
# sidecar tag file at download time. No sidecar -> "unrecorded", never the env default.
if [ -d "${V}/llama.xcframework" ]; then
    if [ -f "${V}/llama.xcframework.tag" ]; then
        LLAMA_VER="$(tr -d '[:space:]' < "${V}/llama.xcframework.tag")"
        add_pin "llama.cpp" "${LLAMA_VER}" "llama-${LLAMA_VER}-xcframework.zip"
    else
        add_pin "llama.cpp" "unrecorded (framework predates the tag sidecar)"
    fi
fi

# mlx-swift — the actual SwiftPM resolution when one exists (Package.resolved appears
# after the first build); the project.yml pin otherwise (clean clone, pre-build).
MLX_REV=""
RESOLVED="BenchmarkApp.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
if [ -f "${RESOLVED}" ]; then
    MLX_REV="$(/usr/bin/python3 - "${RESOLVED}" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
pins = d.get("pins") or d.get("object", {}).get("pins", [])
for p in pins:
    if str(p.get("identity") or p.get("package", "")).lower() == "mlx-swift-lm":
        print(p.get("state", {}).get("revision", ""))
        break
PY
)"
fi
if [ -z "${MLX_REV}" ]; then
    MLX_REV="$(awk '/mlx-swift-lm:/{f=1} f && /revision:/{print $2; exit}' project.yml 2>/dev/null || true)"
fi
if [ -n "${MLX_REV}" ]; then
    add_pin "mlx-swift" "${MLX_REV}"
fi

# core-ai — git state of the (possibly symlinked) checkout, plus whether it carries the
# unpublished StaticInputBuffer engine patch (the PLE-model arm is meaningless without
# knowing which of the two engines ran).
if [ -e "${V}/coreai-models" ]; then
    COREAI_VER="$(git -C "${V}/coreai-models" describe --tags --always 2>/dev/null || true)"
    if [ -n "${COREAI_VER}" ]; then
        if grep -qrs "public struct StaticInputBuffer" "${V}/coreai-models/swift/Sources" 2>/dev/null; then
            COREAI_VER="${COREAI_VER}+static-inputs-patch"
        fi
        add_pin "core-ai" "${COREAI_VER}"
    fi
fi

# cactus — built from a pinned source clone; the clone's HEAD is the version.
if [ -d "${V}/cactus/.git" ]; then
    add_pin "cactus" "$(git -C "${V}/cactus" rev-parse --short=8 HEAD)" "cactus-ios.xcframework (built from source)"
elif [ -d "${V}/cactus-ios.xcframework" ]; then
    add_pin "cactus" "unrecorded (framework present without its source clone)"
fi

PINS_JSON="{${ENTRIES}}"

# json.tool both pretty-prints and validates; a malformed pin string fails the build here
# rather than producing rows with silently broken provenance.
printf '%s\n' "${PINS_JSON}" | /usr/bin/python3 -m json.tool > "${V}/engine-pins.json"
echo "engine pins -> ${V}/engine-pins.json"

# Build-phase mode: stamp the processed Info.plist of the app target. Tool targets
# (yardstick) have no Info.plist and only refresh the json above.
if [ -n "${TARGET_BUILD_DIR:-}" ] && [ -n "${INFOPLIST_PATH:-}" ]; then
    PLIST="${TARGET_BUILD_DIR}/${INFOPLIST_PATH}"
    if [ -f "${PLIST}" ]; then
        plutil -replace BenchEnginePins -json "${PINS_JSON}" "${PLIST}"
        echo "stamped BenchEnginePins -> ${PLIST}"
    fi
fi

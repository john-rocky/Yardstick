#!/usr/bin/env bash
# Fetch the binary + cloned dependencies the BenchmarkApp project needs:
#
#   1. Vendored/llama.xcframework  (llama.cpp Metal build, ~168 MB)
#   2. Vendored/Anemll             (cloned source for AnemllCore SwiftPM target)
#   3. Vendored/LiteRT-LM          (cloned source for the LiteRTLM SwiftPM target)
#   4. Vendored/CoreML-LLM         (cloned source for the CoreMLLLM SwiftPM target)
#   5. Vendored/coreai-models      (cloned source for the CoreAILM SwiftPM target)
#
# Optional: if `xcodegen` is installed and you've edited project.yml, this
# script also regenerates BenchmarkApp.xcodeproj. Most users do NOT need
# xcodegen — the .xcodeproj is committed.
set -euo pipefail

cd "$(dirname "$0")/.."

LLAMA_TAG="${LLAMA_TAG:-b8999}"
LLAMA_ZIP_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-xcframework.zip"
VENDORED_DIR="Vendored"
LLAMA_FRAMEWORK="${VENDORED_DIR}/llama.xcframework"
ANEMLL_DIR="${VENDORED_DIR}/Anemll"
ANEMLL_PKG="${ANEMLL_DIR}/anemll-swift-cli/Package.swift"
LITERTLM_DIR="${VENDORED_DIR}/LiteRT-LM"
LITERTLM_TAG="${LITERTLM_TAG:-v0.13.1}"

mkdir -p "${VENDORED_DIR}"

# 1. llama.xcframework
if [ ! -d "${LLAMA_FRAMEWORK}" ]; then
    echo "Downloading llama.xcframework (${LLAMA_TAG}) …"
    TMP_ZIP="$(mktemp -t llama-xcf).zip"
    trap 'rm -f "${TMP_ZIP}"' EXIT
    curl -L --fail -o "${TMP_ZIP}" "${LLAMA_ZIP_URL}"
    UNPACK_DIR="$(mktemp -d -t llama-unpack)"
    unzip -q "${TMP_ZIP}" -d "${UNPACK_DIR}"
    if [ -d "${UNPACK_DIR}/llama.xcframework" ]; then
        mv "${UNPACK_DIR}/llama.xcframework" "${LLAMA_FRAMEWORK}"
    elif [ -d "${UNPACK_DIR}/build-apple/llama.xcframework" ]; then
        mv "${UNPACK_DIR}/build-apple/llama.xcframework" "${LLAMA_FRAMEWORK}"
    else
        echo "ERROR: Could not locate llama.xcframework inside the release zip." >&2
        find "${UNPACK_DIR}" -maxdepth 3 -name "llama.xcframework" -print
        exit 1
    fi
    rm -rf "${UNPACK_DIR}"
    echo "  -> ${LLAMA_FRAMEWORK}"
else
    echo "${LLAMA_FRAMEWORK} already present."
fi

# 2. Anemll (SwiftPM Package.swift lives in a subdirectory; SPM cannot resolve from a remote URL).
if [ ! -d "${ANEMLL_DIR}" ]; then
    echo "Cloning Anemll …"
    git clone --depth 1 https://github.com/Anemll/Anemll.git "${ANEMLL_DIR}"
else
    echo "${ANEMLL_DIR} already present."
fi

# Patch Anemll's swift-transformers pin from branch:main to a tagged 1.x range.
# Upstream branch:main HEAD removed the Tokenizers product (now target-only),
# which breaks john-rocky/CoreML-LLM. Pinning to 1.0.0..<2.0.0 keeps both happy.
if grep -q 'swift-transformers".*branch: "main"' "${ANEMLL_PKG}"; then
    echo "Patching Anemll Package.swift to pin swift-transformers 1.x …"
    sed -i '' 's|.package(url: "https://github.com/huggingface/swift-transformers", branch: "main").*|.package(url: "https://github.com/huggingface/swift-transformers", "1.0.0"..<"2.0.0"),|' "${ANEMLL_PKG}"
fi

# 3. LiteRT-LM (google-ai-edge). Vendored as a *local* SwiftPM package: the released
# package declares `.unsafeFlags(["-Xlinker","-all_load"])`, and SwiftPM rejects unsafe
# flags from versioned/remote products — local path packages are allowed. GIT_LFS_SKIP_SMUDGE=1
# skips a broken Android LFS object upstream; the iOS/macOS xcframework is a remote
# binaryTarget (release URL), so it's fetched by SwiftPM, not git-LFS.
if [ ! -d "${LITERTLM_DIR}" ]; then
    echo "Cloning LiteRT-LM (${LITERTLM_TAG}) …"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "${LITERTLM_TAG}" \
        https://github.com/google-ai-edge/LiteRT-LM.git "${LITERTLM_DIR}"
else
    echo "${LITERTLM_DIR} already present."
fi

# 4. CoreML-LLM (john-rocky). Local SwiftPM package (CoreMLLLM product) for the CoreML/ANE
#    runtime. Tracks the latest default branch; pin a commit if you need exact reproduction.
COREMLLLM_DIR="${VENDORED_DIR}/CoreML-LLM"
if [ ! -d "${COREMLLLM_DIR}" ]; then
    echo "Cloning CoreML-LLM …"
    git clone --depth 1 https://github.com/john-rocky/CoreML-LLM.git "${COREMLLLM_DIR}"
else
    echo "${COREMLLLM_DIR} already present."
fi

# 5. coreai-models (Apple Core AI, CoreAILM product). Ships CXGrammar.xcframework in-repo.
#    Required for the project to *compile* (project.yml links it) even though Core AI *runs*
#    additionally need a side-loaded .aimodel in Documents/CoreAIModels/<id>/.
COREAI_DIR="${VENDORED_DIR}/coreai-models"
COREAI_TAG="${COREAI_TAG:-0.2.0}"
# The Core AI arm needs StaticInputBuffer (EngineOptions.staticInputBuffers) to feed Gemma-4's
# per-layer-embedding table to the `_tbl` decode graph. That API does NOT exist in Apple's public
# coreai-models (checked: absent from both 0.1.0 and 0.2.0) — it comes from our engine patch,
# coreai-models-community/apps/coreai-pipelined-static-inputs.patch. So a plain clone of the
# public repo compiles everything EXCEPT CoreAIRuntime.swift, with a bare
# "cannot find type 'StaticInputBuffer' in scope".
# Prefer a local patched checkout (what project.yml's comment assumes); fall back to the public
# tag so the other arms still build, and say so loudly rather than failing 200 lines later.
COREAI_LOCAL="${COREAI_LOCAL:-$HOME/code/coreai/coreai-models}"
if [ ! -e "${COREAI_DIR}" ]; then
    if [ -d "${COREAI_LOCAL}" ] && grep -qrs "public struct StaticInputBuffer" "${COREAI_LOCAL}/swift/Sources"; then
        echo "Linking coreai-models -> ${COREAI_LOCAL} (patched: has StaticInputBuffer)"
        ln -s "${COREAI_LOCAL}" "${COREAI_DIR}"
    else
        echo "WARNING: no patched coreai-models at ${COREAI_LOCAL}."
        echo "         Cloning public ${COREAI_TAG}; the Core AI arm will NOT compile"
        echo "         (needs coreai-pipelined-static-inputs.patch). Other arms are fine."
        git clone --depth 1 --branch "${COREAI_TAG}" https://github.com/apple/coreai-models.git "${COREAI_DIR}"
    fi
else
    echo "${COREAI_DIR} already present."
fi

# 6. Cactus engine (cactus-compute/cactus) — build cactus-ios.xcframework from source.
#    The framework bundles cactus_engine.h + a module map, so `import cactus` works
#    directly; CactusRuntime.swift is canImport-guarded and stubs out if this step
#    is skipped (e.g. no cmake). Requires cmake >= 3.10 + Xcode iOS SDK.
CACTUS_DIR="${VENDORED_DIR}/cactus"
CACTUS_FRAMEWORK="${VENDORED_DIR}/cactus-ios.xcframework"
# Pinned: cactus main moves fast and artifacts change under it (their 07-09 CQ4 swap
# is a table-level finding in this repo). This is the commit the published cells used.
CACTUS_COMMIT="${CACTUS_COMMIT:-1ace6d78}"
if [ ! -d "${CACTUS_FRAMEWORK}" ]; then
    if command -v cmake >/dev/null 2>&1; then
        if [ ! -d "${CACTUS_DIR}" ]; then
            echo "Cloning cactus @ ${CACTUS_COMMIT} …"
            git clone https://github.com/cactus-compute/cactus.git "${CACTUS_DIR}"
            (cd "${CACTUS_DIR}" && git checkout --quiet "${CACTUS_COMMIT}")
        fi
        echo "Building cactus-ios.xcframework (this compiles the engine twice: device + simulator) …"
        if (cd "${CACTUS_DIR}" && bash apple/build.sh); then
            cp -R "${CACTUS_DIR}/apple/cactus-ios.xcframework" "${CACTUS_FRAMEWORK}"
            # Upstream bundles only cactus_engine.h, whose first include is the C++
            # cactus_graph.h — absent from the framework, so `import cactus` cannot
            # compile as shipped. The FFI declarations never use graph types; drop
            # the include from the bundled header (our copy only).
            for h in "${CACTUS_FRAMEWORK}"/*/cactus.framework/Headers/cactus_engine.h; do
                sed -i '' '/#include "cactus_graph.h"/d' "$h"
            done
            echo "  -> ${CACTUS_FRAMEWORK}"
        else
            echo "WARNING: cactus apple build failed; the Cactus arm will report unavailable."
        fi
    else
        echo "WARNING: cmake not found; skipping the Cactus arm (CactusRuntime stubs out)."
    fi
else
    echo "${CACTUS_FRAMEWORK} already present."
fi

# 7. Optional: regenerate the Xcode project (only needed if you edit project.yml).
if command -v xcodegen >/dev/null 2>&1; then
    if [ "${REGEN_XCODEPROJ:-0}" = "1" ] || [ ! -d "BenchmarkApp.xcodeproj" ]; then
        echo "Generating BenchmarkApp.xcodeproj …"
        xcodegen generate
    else
        echo "BenchmarkApp.xcodeproj already present (set REGEN_XCODEPROJ=1 to force regen)."
    fi
fi

cat <<'EOF'

Done. Open the project:
    open BenchmarkApp.xcodeproj

In Xcode:
  1. Set your Apple Developer Team in Signing & Capabilities.
  2. Select your iPhone as the run destination.
  3. ⌘R.

LiteRT-LM (Gemma .litertlm, GPU) is vendored above (Vendored/LiteRT-LM) and wired via
project.yml — no manual Xcode step. Its decode tok/s comes from LiteRT-LM's own counters.

First Xcode build resolves SPM (mlx-swift-lm, swift-huggingface, swift-transformers,
executorch, Anemll, CoreMLLLM). Takes several minutes.
EOF

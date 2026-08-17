#!/usr/bin/env python3
"""Compare upstream engine releases against this repo's pins and print the drift
table plus the exact next command (`./bench release-watch`).

Uses `gh` for the GitHub API (already a repo prerequisite). Read-only: bumping a
pin stays a human decision — the flow is documented in docs/OPERATIONS.md.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gh_json(args):
    try:
        out = subprocess.check_output(["gh", *args], text=True, stderr=subprocess.DEVNULL)
        return json.loads(out)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


def latest_release(repo):
    d = gh_json(["api", f"repos/{repo}/releases/latest"])
    return d.get("tag_name") if d else None


def main():
    lock = json.load(open(os.path.join(ROOT, "environment.lock.json")))
    arms = lock["arms"]
    rows = []

    litert = arms.get("litert-lm", {})
    litert_pinned = sorted({c.get("tag") for c in litert.get("captures", {}).values()
                            if c.get("tag")})[-1] if litert.get("captures") else litert.get("default_tag")
    rows.append(("litert-lm", "google-ai-edge/LiteRT-LM",
                 latest_release("google-ai-edge/LiteRT-LM"), litert_pinned,
                 "./bench regress matrices/release-regression-litert.cells "
                 "--engine litert-lm  (bump LITERTLM_TAG in bootstrap.sh first)"))

    rows.append(("llama.cpp", "ggml-org/llama.cpp",
                 latest_release("ggml-org/llama.cpp"),
                 arms.get("llama.cpp", {}).get("tag"),
                 "bump arms.llama.cpp in environment.lock.json + bootstrap; "
                 "android: android/scripts/fetch_llama_android.sh"))

    coreai = arms.get("coreai-models", {})
    rows.append(("core-ai", "apple/coreai-models",
                 latest_release("apple/coreai-models"),
                 coreai.get("tag"),
                 "best-effort arm: PLE models still need the unpublished "
                 "COREAI_STATIC_INPUTS patch"))

    mlx = arms.get("mlx-swift-lm", {})
    head = gh_json(["api", "repos/ml-explore/mlx-swift-examples/commits/main",
                    "--jq", "{sha: .sha}"])
    rows.append(("mlx-swift-lm", "ml-explore/mlx-swift-examples",
                 (head or {}).get("sha", "")[:8] or None,
                 (mlx.get("revision") or "")[:8],
                 "pinned BY REVISION in project.yml (floating-branch incident) — "
                 "bump deliberately, never track main"))

    print(f"{'arm':14} {'pinned':12} {'latest':12}  action")
    print("-" * 100)
    drift = False
    for arm, repo, latest, pinned, action in rows:
        mark = ""
        if latest is None:
            latest = "(unknown)"  # API miss is not drift
        elif pinned and str(latest) != str(pinned):
            mark = "  <-- DRIFT"
            drift = True
        print(f"{arm:14} {str(pinned):12} {str(latest):12}{mark}")
        if mark:
            print(f"{'':14} next: {action}")
    if not drift:
        print("\nall pins current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""bench doctor — preflight for the standing matrix.

Says what is missing BEFORE a run dies on it, with the exact fix command.
Born from the 2026-08-26 fresh-clone rehearsal, where every finding below was
hit in order: build attempted without bootstrap (opaque xcodegen error), the
runner silently picking up an out-of-repo yardstick, a bootstrap default tag
three releases behind the published rows, and `bench matrix` "succeeding"
with zero cells run.

  ./bench doctor                       # common + mac; device lanes best-effort
  ./bench doctor --platform iphone     # device absence is then a failure

Exit 1 if any required check fails.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "ios", "BenchmarkApp")
VENDORED = os.path.join(APP_DIR, "Vendored")

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
_RESULTS = []


def check(status, name, detail=""):
    _RESULTS.append((status, name, detail))


def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def which(tool):
    return shutil.which(tool)


# ---------- common ----------

def check_common():
    v = sys.version_info
    check(OK if v >= (3, 10) else FAIL, f"python {v.major}.{v.minor}",
          "" if v >= (3, 10) else "need >= 3.10")
    try:
        import matplotlib  # noqa: F401
        check(OK, "matplotlib (charts)")
    except ImportError:
        check(WARN, "matplotlib (charts)", "pip install matplotlib — only generate_charts.py needs it")
    tok = os.environ.get("HF_TOKEN") or os.path.exists(
        os.path.expanduser("~/.cache/huggingface/token"))
    check(OK if tok else WARN, "HF auth",
          "" if tok else "public models download anonymously; gated ones (Gemma) need `hf auth login`")
    st = os.statvfs(ROOT)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    check(OK if free_gb > 20 else WARN, f"disk free {free_gb:.0f} GB",
          "" if free_gb > 20 else "multi-GB models + a ~10 GB build tree land here")


# ---------- mac ----------

def newest_lock_litert_tag():
    try:
        lock = json.load(open(os.path.join(ROOT, "environment.lock.json")))
        tags = [c.get("tag") for c in lock["arms"]["litert-lm"]["captures"].values()
                if c.get("tag")]
        return max(tags, key=lambda t: [int(x) for x in re.findall(r"\d+", t)]) if tags else None
    except (OSError, KeyError, ValueError):
        return None


def check_mac(required):
    if not which("xcodebuild"):
        check(FAIL if required else WARN, "xcodebuild", "install Xcode (27 beta era; see docs/OPERATIONS.md)")
        return
    _, out = run(["xcodebuild", "-version"])
    check(OK, "xcodebuild", out.split("\n")[0].strip())
    check(OK if which("xcodegen") else FAIL if required else WARN, "xcodegen",
          "" if which("xcodegen") else "brew install xcodegen — build_yardstick_mac.sh needs it")

    missing = [d for d in ("llama.xcframework", "Anemll", "LiteRT-LM",
                           "CoreML-LLM", "coreai-models")
               if not os.path.exists(os.path.join(VENDORED, d))]
    if missing:
        check(FAIL if required else WARN, "Vendored engines",
              f"missing {', '.join(missing)} — run ios/BenchmarkApp/scripts/bootstrap.sh")
    else:
        check(OK, "Vendored engines")
        want = newest_lock_litert_tag()
        rc, out = run(["git", "-C", os.path.join(VENDORED, "LiteRT-LM"),
                       "describe", "--tags", "--always"])
        have = out.strip().split("\n")[0] if rc == 0 else None
        if want and have and not have.startswith(want):
            check(WARN, "LiteRT-LM vendored tag",
                  f"vendored {have}, newest lock capture {want} — rows will stamp what is "
                  f"vendored; LITERTLM_TAG={want} bootstrap.sh to match the published rows")
        elif have:
            check(OK, "LiteRT-LM vendored tag", have)

    dd = os.environ.get("DD_MAC", os.path.join(ROOT, ".build", "dd-mac"))
    ys = os.environ.get("YS_BIN", os.path.join(dd, "Build", "Products", "Release", "yardstick"))
    if not os.access(ys, os.X_OK):
        check(FAIL if required else WARN, "mac yardstick",
              f"no binary at {ys} — run scripts/build_yardstick_mac.sh (~15-30 min first time)")
        return
    rc, out = run([ys, "version"])
    flavor = re.search(r"flavor=(\S+)", out)
    flavor = flavor.group(1) if flavor else "unknown"
    if flavor == "full":
        check(OK, "mac yardstick", out.strip().split("\n")[0])
    else:
        check(FAIL if required else WARN, "mac yardstick",
              f"flavor={flavor} — llama-cpp etc. compiled out; rebuild via scripts/build_yardstick_mac.sh")


# ---------- iphone ----------

def check_iphone(required):
    lvl = FAIL if required else WARN
    check(OK if which("gtimeout") else lvl, "gtimeout (coreutils)",
          "" if which("gtimeout") else "brew install coreutils — litert teardown hangs need it")
    rc, out = run(["xcrun", "devicectl", "list", "devices"], timeout=30)
    if rc != 0:
        check(lvl, "devicectl", "xcrun devicectl failed — Xcode command line tools?")
        return
    devices = [ln for ln in out.strip().split("\n")[2:] if ln.strip()]
    if not devices:
        check(lvl, "iPhone device", "no device visible to devicectl — connect + trust it")
        return
    udid = os.environ.get("BENCH_UDID")
    if udid:
        check(OK, "iPhone device", f"BENCH_UDID={udid}")
    elif len(devices) == 1:
        check(OK, "iPhone device", devices[0].split()[0])
    else:
        check(lvl, "iPhone device", f"{len(devices)} devices visible — set BENCH_UDID")
    check(WARN, "BenchmarkApp install",
          "not probed — needs the increased-memory entitlements app; see docs/OPERATIONS.md 'add a device'")


# ---------- android ----------

def check_android(required):
    lvl = FAIL if required else WARN
    if not which("adb"):
        check(lvl, "adb", "install platform-tools")
        return
    rc, out = run(["adb", "devices"])
    devs = [ln.split()[0] for ln in out.strip().split("\n")[1:]
            if ln.strip() and ln.split()[-1] == "device"]
    if not devs:
        check(lvl, "Android device", "no authorized device — plug in + accept the RSA prompt")
        return
    serial = os.environ.get("BENCH_ANDROID_SERIAL")
    if not serial and len(devs) > 1:
        check(lvl, "Android device", f"{len(devs)} devices — set BENCH_ANDROID_SERIAL")
        return
    serial = serial or devs[0]
    check(OK, "Android device", serial)
    base = ["adb"] + (["-s", serial] if serial else [])
    rc, out = run(base + ["shell", "sha256sum", "/data/local/tmp/llmbench/litert_lm_main"])
    if rc != 0 or "No such file" in out:
        check(lvl, "litert_lm_main on device",
              "not pushed — android/README.md; binaries: android/bin/ or the GitHub release assets")
    else:
        sha = out.split()[0]
        try:
            pins = json.load(open(os.path.join(ROOT, "android", "engine-pins.json")))
            known = json.dumps(pins).find(sha) >= 0
        except OSError:
            known = False
        check(OK if known else WARN, "litert_lm_main on device",
              sha[:12] + ("" if known else " — sha not in android/engine-pins.json; rows will stamp 'unknown'"))


def main():
    ap = argparse.ArgumentParser(prog="bench doctor")
    ap.add_argument("--platform", action="append",
                    choices=["mac", "iphone", "android"],
                    help="lane(s) to require; default: check all, require none but common")
    args = ap.parse_args()
    req = set(args.platform or [])

    check_common()
    check_mac("mac" in req or not req and sys.platform == "darwin")
    check_iphone("iphone" in req)
    check_android("android" in req)

    width = max(len(n) for _, n, _ in _RESULTS)
    icon = {OK: "  ok ", WARN: "warn ", FAIL: "FAIL ", SKIP: "skip "}
    for status, name, detail in _RESULTS:
        print(f"{icon[status]} {name:<{width}}  {detail}".rstrip())
    fails = sum(1 for s, _, _ in _RESULTS if s == FAIL)
    print(f"\n{fails} failure(s)" if fails else "\nready")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

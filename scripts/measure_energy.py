#!/usr/bin/env python3
"""
Yardstick — measure system energy alongside a bench run, on Mac.

Wraps `yardstick run` with a co-running `powermetrics` subprocess so we
can attribute joules to a single bench invocation. Re-writes the JSONL
that yardstick produced, adding:

    metrics.energyJoules            # CPU + GPU + ANE × elapsed
    metrics.averagePackagePowerW    # arithmetic mean across samples
    metrics.energyJoulesPerToken    # if the run reported a token count
    metrics.energySource = "powermetrics"
    metrics.energyMeasurementWindowSeconds

Important caveats — read before quoting numbers:

  * powermetrics measures the whole system, not just our process. Run
    on an idle desktop, with no other heavyweight apps open.
  * The samples are estimated by Apple's tooling and may carry a few
    hundred mW of bias. Use them to compare runtimes on the *same*
    Mac, not to compare devices.
  * Sub-second runs sample too few intervals to be useful; the script
    aborts with a warning if fewer than 4 power samples land inside
    the bench window.

Usage:

    sudo python scripts/measure_energy.py run \\
        --task short-chat --runtime mlx-swift \\
        --model mlx-community/gemma-4-e2b-it-4bit \\
        --output results/raw/m4max-mlx-gemma-4-e2b-energy.jsonl

The `sudo` is for powermetrics only; the script preserves env so that
the user's HF cache + yardstick binary on PATH still resolve.

The wrapped yardstick binary is found in this order:

  1. $YARDSTICK_BIN  (if set)
  2. ./yardstick or ./build/yardstick under the repo
  3. /tmp/yardstick-dd/Build/Products/Release/yardstick (the path the
     repo's xcodebuild command produces)
  4. `which yardstick`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE_INTERVAL_MS = 100
POWER_FALLBACK_PATTERNS = {
    # `Combined Power (CPU + GPU + ANE): 1234 mW` is the line powermetrics
    # prints when the cpu_power / gpu_power / ane_power samplers are all
    # active. Different macOS releases sometimes drop the parens or change
    # the casing, so we match on a loose regex.
    "combined": re.compile(
        r"Combined Power.*?:\s*([\d.]+)\s*mW", re.IGNORECASE
    ),
    "cpu": re.compile(r"CPU Power\s*:\s*([\d.]+)\s*mW", re.IGNORECASE),
    "gpu": re.compile(r"GPU Power\s*:\s*([\d.]+)\s*mW", re.IGNORECASE),
    "ane": re.compile(r"ANE Power\s*:\s*([\d.]+)\s*mW", re.IGNORECASE),
}


def locate_yardstick() -> Path:
    env = os.environ.get("YARDSTICK_BIN")
    if env and Path(env).is_file():
        return Path(env)
    candidates = [
        REPO / "yardstick",
        REPO / "build" / "yardstick",
        Path("/tmp/yardstick-dd/Build/Products/Release/yardstick"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    which = shutil.which("yardstick")
    if which:
        return Path(which)
    raise SystemExit(
        "error: cannot find the `yardstick` binary. Set $YARDSTICK_BIN or "
        "build via `xcodebuild -scheme yardstick -configuration Release "
        "-derivedDataPath /tmp/yardstick-dd build` first."
    )


def parse_power_samples(text: str) -> dict[str, list[float]]:
    """Return per-subsystem power samples (mW), one entry per powermetrics
    interval, as ``{"combined": [...], "cpu": [...], "gpu": [...], "ane": [...]}``.

    powermetrics prints one block per sample. We keep the per-subsystem
    (CPU / GPU / ANE) series so callers can show *which* compute unit did
    the work — e.g. an ANE run leaves the GPU near-idle, which is the whole
    point of running an LLM on the Neural Engine. ``combined`` prefers a
    ``Combined Power`` line when present, else the CPU+GPU+ANE sum.
    """
    combined: list[float] = []
    cpu_s: list[float] = []
    gpu_s: list[float] = []
    ane_s: list[float] = []
    block_cpu = block_gpu = block_ane = block_combined = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = POWER_FALLBACK_PATTERNS["combined"].search(line)
        if m:
            block_combined = float(m.group(1))
            continue
        m = POWER_FALLBACK_PATTERNS["cpu"].search(line)
        if m:
            block_cpu = float(m.group(1))
            continue
        m = POWER_FALLBACK_PATTERNS["gpu"].search(line)
        if m:
            block_gpu = float(m.group(1))
            continue
        m = POWER_FALLBACK_PATTERNS["ane"].search(line)
        if m:
            block_ane = float(m.group(1))
            # ANE line tends to be last in a powermetrics block; commit the
            # per-subsystem readings (and a combined value) once we hit it.
            if block_cpu is not None:
                cpu_s.append(block_cpu)
            if block_gpu is not None:
                gpu_s.append(block_gpu)
            ane_s.append(block_ane)
            if block_combined is not None:
                combined.append(block_combined)
            elif block_cpu is not None and block_gpu is not None:
                combined.append(block_cpu + block_gpu + block_ane)
            block_cpu = block_gpu = block_ane = block_combined = None
            continue
        if line == "":
            block_cpu = block_gpu = block_ane = block_combined = None
    return {"combined": combined, "cpu": cpu_s, "gpu": gpu_s, "ane": ane_s}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="run",
        choices=["run"],
        help="yardstick subcommand to wrap (only 'run' is supported)",
    )
    parser.add_argument("--task", default="short-chat")
    parser.add_argument("--runtime", default="mlx-swift")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="Forced context budget passed through to `yardstick run` — the agreed "
        "protocol pins Gemma-4 at 2048. Without it the cell derives its context from "
        "the prompt and records no contextTokensConfigured, so it is not card-comparable.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Passed through to `yardstick run --runs` (energy cells use 1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. Defaults to results/raw/<device>-<runtime>-<model>-<task>-energy.jsonl.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device label prefix for the auto-generated output filename "
        "(must match a key in render_results.py::DEVICE_DISPLAY, e.g. "
        "'m4max', 'm3air'). Defaults to a generic 'mac'.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=int,
        default=SAMPLE_INTERVAL_MS,
        help=f"powermetrics sampling interval in ms (default {SAMPLE_INTERVAL_MS}).",
    )
    args = parser.parse_args()

    if os.geteuid() == 0:
        print(
            "error: do NOT run this script with sudo. It elevates `powermetrics`\n"
            "internally (which prompts for your password once). Running the\n"
            "whole script as root makes the yardstick subprocess re-download\n"
            "every model into /var/root/.cache/huggingface instead of using\n"
            "your existing ~/.cache/huggingface.",
            file=sys.stderr,
        )
        return 2

    bin_path = locate_yardstick()

    output_path = args.output
    if output_path is None:
        # Default to a sensibly-named file under results/raw/ so the user
        # doesn't have to type the full path on the sudo line.
        model_tag = (args.model or "default").split("/")[-1].lower()
        dev_tag = args.device or "mac"
        output_path = str(
            REPO / "results" / "raw"
            / f"{dev_tag}-{args.runtime}-{model_tag}-{args.task}-energy.jsonl"
        )

    yardstick_argv = [
        str(bin_path),
        "run",
        "--task", args.task,
        "--runtime", args.runtime,
        "--output", output_path,
    ]
    if args.model:
        yardstick_argv += ["--model", args.model]
    if args.context_tokens:
        yardstick_argv += ["--context-tokens", str(args.context_tokens)]
    if args.runs:
        yardstick_argv += ["--runs", str(args.runs)]

    power_log = tempfile.NamedTemporaryFile(
        prefix="yardstick-power-", suffix=".txt", delete=False
    )
    power_log.close()

    print(f"yardstick-energy: powermetrics → {power_log.name}", file=sys.stderr)
    print(f"yardstick-energy: yardstick → {output_path}", file=sys.stderr)

    # Prime sudo credentials NOW, with a foreground prompt, so the
    # background `sudo powermetrics` below doesn't block on a password
    # prompt while the bench is already running. `sudo -v` validates the
    # cached credential without executing anything.
    print(
        "yardstick-energy: validating sudo (needed for powermetrics)...",
        file=sys.stderr,
    )
    rc = subprocess.run(["sudo", "-v"]).returncode
    if rc != 0:
        print("error: sudo validation failed — aborting.", file=sys.stderr)
        return rc

    pm_cmd = [
        "sudo", "-n", "powermetrics",
        "-b", "1",                          # line-buffered so flushes land in the temp file promptly
        "-s", "cpu_power,gpu_power,ane_power",
        "-i", str(args.sample_interval_ms),
    ]
    # Open the temp file *as the user* and hand the FD to powermetrics
    # via stdout. If we'd passed `--output-file`, powermetrics would
    # re-create the file as root (mode 0600) and we couldn't read it
    # afterwards without another sudo.
    pm_log_fp = open(power_log.name, "w")
    # NOTE: do NOT pass `preexec_fn=os.setsid` here. Creating a new
    # session detaches the child from the controlling tty, and sudo's
    # credential cache is keyed on (uid, tty). Without the parent's tty
    # the `sudo -n` would silently fail (cache miss) and powermetrics
    # would never start, producing a 0-byte output file. Killing is
    # done via `sudo -n killall powermetrics` instead of pgid, so we
    # don't need the new session anyway.
    pm_proc = subprocess.Popen(
        pm_cmd,
        stdout=pm_log_fp,
        stderr=subprocess.PIPE,
    )
    # Wait until powermetrics has actually emitted its first sample
    # before starting the bench. Otherwise sub-second benches end up
    # entirely outside the measurement window.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            if Path(power_log.name).stat().st_size > 1024:
                break
        except FileNotFoundError:
            continue
    else:
        print(
            "warn: powermetrics didn't emit any output within 5s — proceeding anyway.",
            file=sys.stderr,
        )

    start_t = time.monotonic()
    bench = subprocess.run(yardstick_argv)
    end_t = time.monotonic()
    elapsed = end_t - start_t

    # Need `sudo kill` because powermetrics is running as root — the
    # earlier `sudo powermetrics` call will have cached the credential
    # so this second sudo doesn't re-prompt within the 5-minute window.
    # killall powermetrics is more reliable than targeting sudo's PID
    # (signal forwarding through sudo is not guaranteed on macOS).
    try:
        subprocess.run(
            ["sudo", "-n", "killall", "-INT", "powermetrics"],
            check=False,
        )
        pm_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["sudo", "-n", "killall", "-TERM", "powermetrics"],
            check=False,
        )
        pm_proc.wait(timeout=5)
    pm_log_fp.close()

    if bench.returncode != 0:
        # llama.cpp's Metal backend often abort()s during the cleanup
        # path on process exit (harmless — the bench output is already
        # flushed). If the JSONL exists and looks well-formed we still
        # patch it; otherwise bail out with the bench's exit code.
        out_exists = Path(output_path).is_file() and Path(output_path).stat().st_size > 0
        if not out_exists:
            print(
                f"yardstick exited {bench.returncode} and produced no JSONL; aborting.",
                file=sys.stderr,
            )
            return bench.returncode
        print(
            f"warn: yardstick exited {bench.returncode} but JSONL was written "
            "(likely llama.cpp's known Metal-cleanup abort on exit). Patching anyway.",
            file=sys.stderr,
        )

    text = Path(power_log.name).read_text(encoding="utf-8", errors="replace")
    series = parse_power_samples(text)
    samples_mW = series["combined"]
    # The first sample tends to be a "since boot" average rather than an
    # interval reading. Also drop ~0.5 s of warm-up (the wait loop
    # holds the bench until powermetrics has flushed its first block to
    # the log file). After that, samples are real per-interval readings
    # captured while the bench is actually running. We track the window as
    # [win_start:win_end) so the per-subsystem series get the same slice.
    warmup_samples = max(1, int(0.5 / (args.sample_interval_ms / 1000.0)))
    win_start = warmup_samples
    win_end: int | None = None
    samples_mW = samples_mW[win_start:]

    # If the wrapped binary crashes during exit (the well-known llama.cpp
    # Metal-cleanup `ggml_abort`), the subprocess can take 30+ seconds to
    # finalize while macOS writes a crash report — but the bench itself
    # ended in ~5 s. Naively averaging over the full window dilutes the
    # avg power and inflates joules. Read the bench's *own* reported
    # active time and clip the sample list to that window.
    try:
        with open(output_path) as f:
            for ln in f.read().split("\n"):
                if ln.strip():
                    last_obj = json.loads(ln)
        bench_active_s = float(
            (last_obj.get("metrics", {}).get("loadTimeSeconds") or 0)
            + (last_obj.get("metrics", {}).get("totalGenerationTimeSeconds") or 0)
        )
    except Exception:
        bench_active_s = elapsed

    if 0 < bench_active_s < elapsed:
        n_keep = max(1, int(bench_active_s / (args.sample_interval_ms / 1000.0)))
        if n_keep < len(samples_mW):
            win_end = win_start + n_keep
            samples_mW = samples_mW[:n_keep]
            elapsed = bench_active_s
            print(
                f"yardstick-energy: clipped window from script-elapsed to "
                f"bench-reported {bench_active_s:.2f}s "
                f"(kept {n_keep} samples).",
                file=sys.stderr,
            )
    if len(samples_mW) < 4:
        size = Path(power_log.name).stat().st_size
        print(
            f"warn: only {len(samples_mW)} power samples in the bench window "
            f"({elapsed:.2f}s) — too few to be useful. Skipping JSONL patch.",
            file=sys.stderr,
        )
        print(
            f"diagnostic: powermetrics log = {power_log.name} ({size} bytes).",
            file=sys.stderr,
        )
        print(
            "diagnostic: first 800 chars below — if it doesn't include lines "
            "matching `CPU Power: X mW` / `GPU Power: X mW` / `ANE Power: X mW` "
            "or `Combined Power (...): X mW`, the regexes in "
            "parse_power_samples() need updating for your macOS release.",
            file=sys.stderr,
        )
        print(text[:800], file=sys.stderr)
        # Drain powermetrics' stderr too — if sudo silently refused or
        # the sampler list isn't recognised on this macOS release, the
        # explanation lives there.
        try:
            pm_stderr = pm_proc.stderr.read().decode("utf-8", errors="replace") if pm_proc.stderr else ""
        except Exception:
            pm_stderr = ""
        if pm_stderr:
            print(f"diagnostic: powermetrics stderr: {pm_stderr[:400]}", file=sys.stderr)
        return 0

    avg_W = (sum(samples_mW) / len(samples_mW)) / 1000.0
    energy_J = avg_W * elapsed

    # Per-subsystem average power over the *same* window. This is what makes
    # the ANE case visible: a CoreML/ANE run shows ANE busy + GPU near-idle,
    # whereas an MLX/Metal run shows the GPU saturated. None when a sampler
    # produced no readings on this macOS release.
    def _avg_window_W(key: str):
        s = series.get(key, [])
        w = s[win_start:win_end] if win_end is not None else s[win_start:]
        return round((sum(w) / len(w)) / 1000.0, 4) if w else None

    ane_W = _avg_window_W("ane")
    gpu_W = _avg_window_W("gpu")
    cpu_W = _avg_window_W("cpu")

    # Patch the JSONL written by yardstick.
    out_path = Path(output_path)
    raw = out_path.read_text(encoding="utf-8").strip()
    if not raw:
        print("error: yardstick produced an empty output file", file=sys.stderr)
        return 1
    # The yardstick `run` command writes a single JSON object per
    # invocation; multiple invocations append, separated by newlines.
    # The user is supposed to point each measure_energy call at a fresh
    # file, but be defensive and patch only the last entry.
    lines = [l for l in raw.split("\n") if l.strip()]
    last = json.loads(lines[-1])
    metrics = last.setdefault("metrics", {})
    metrics["energyJoules"] = round(energy_J, 4)
    metrics["averagePackagePowerW"] = round(avg_W, 4)
    if ane_W is not None:
        metrics["averageANEPowerW"] = ane_W
    if gpu_W is not None:
        metrics["averageGPUPowerW"] = gpu_W
    if cpu_W is not None:
        metrics["averageCPUPowerW"] = cpu_W
    metrics["energyMeasurementWindowSeconds"] = round(elapsed, 4)
    metrics["energySource"] = "powermetrics"
    gen_tok = metrics.get("generatedTokenCount") or 0
    if gen_tok > 0:
        metrics["energyJoulesPerToken"] = round(energy_J / gen_tok, 6)

    lines[-1] = json.dumps(last)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    split = "  ".join(
        f"{name}={val:.2f}W"
        for name, val in (("ANE", ane_W), ("GPU", gpu_W), ("CPU", cpu_W))
        if val is not None
    )
    print(
        f"yardstick-energy: window={elapsed:.2f}s "
        f"samples={len(samples_mW)} "
        f"avgPkg={avg_W:.2f}W energy={energy_J:.2f}J"
        + (f"  [{split}]" if split else ""),
        file=sys.stderr,
    )
    if gen_tok > 0:
        print(
            f"yardstick-energy: J/tok={energy_J / gen_tok:.4f} (n={gen_tok})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

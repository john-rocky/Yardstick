# Android measurement semantics — what transfers from the Apple lane and what does not

The Android column shares the schema (result.v1), the accumulation layer, the
regression differ, and the fairness rules. The measurement *mechanics* differ;
every difference below is disclosed here once so tables don't have to.

## Arms

| arm | recorded runtime | acquisition | backend |
|---|---|---|---|
| LiteRT-LM | `litert-lm-cpu` / `litert-lm-gpu` | per-release bazel+NDK source build (`android/scripts/build_litert_lm_main.sh`) — releases ship no Android binary (verified v0.14–v0.16); upstream build bug + fix flag: google-ai-edge/LiteRT-LM#3247 | cpu, gpu (ML Drift .so set). NPU = n/a (Early Access only) — row stays with reason (failed-runs-stay) |
| llama.cpp | `llama.cpp` | official release artifact `llama-<tag>-bin-android-arm64.tar.gz`, same tag as the Apple arm (official-sdk rule) | CPU only in the official artifact; GPU (OpenCL) would need a custom NDK build — phase 2, disclosed as n/a until then |
| MLX / Core AI | — | Apple-only; render as n/a | |
| Cactus | — | phase 2 slot (`environment.lock.json` `arms.cactus.android: planned`) | |

Backend is part of arm identity (`litert-lm-gpu` vs `litert-lm-cpu`) because the
regression join key has no backend column — same convention as Core AI's
`-ane`/`-gpu` model ids on iOS.

The Maven `litertlm-android` AAR is the *official* Android artifact but exposes
only the Kotlin API (no CLI): using it means an instrumented APK harness. That
is the official-sdk-rule tension for this arm, recorded as a phase-2 option;
v1 measures the same engine through `litert_lm_main`.

## Regimes

- One engine process per run = **cold** by this repo's definition (fresh
  process, caches on disk). The very first run per (model, backend) builds the
  ML Drift / OpenCL caches → labelled `firstEver`, reported separately, never
  as the engine's speed (cold-warm-split).
- **No warm regime in v1**: the CLIs have no in-process repeat for prompt tasks
  (`--multi_turns` is interactive). Android cells therefore compare against
  Apple **cold** rows only. llama-bench (`native-benchmark-*` task) repeats
  in-process and is recorded `coldRun=false`; litert `--benchmark` is a fresh
  process (`coldRun=true`) — the two native rows are different regimes and the
  join key already separates them.

## Metrics — measured vs absent (absent stays absent)

| metric | litert_lm_main | llama-cli | llama-bench |
|---|---|---|---|
| decode / prefill tok/s | engine-reported | engine-reported | engine-reported (avg_ts) |
| TTFT | engine-reported (excludes init, like iOS loadTime split) | **absent** | **absent** |
| memory | driver RSS sampler (`/proc/<pid>/status` VmRSS, 0.5 s, median) → `memoryMedianResidentMB`; `memoryMedianMB` (phys_footprint) has **no Android equivalent — never fabricated** | same | same |
| energy | manual only (batterystats delta, unplugged) — phase 2 protocol, owner-triggered | same | same |

## Conditions pinned per run

- `taskset f0` (big cores; upstream's own recommendation) — in `conditions.cpuAffinity`.
- Thermal: `dumpsys thermalservice` status int; 0 → `"nominal"` so the repo's
  nominal gate works unchanged; the raw int rides along in
  `conditions.thermalRawStatus`. Gate: wait for status 0 up to `THERMAL_WAIT`
  (default 600 s), run anyway after timeout and record it (`THERMAL_GATE.txt`).
- Battery level/state from `dumpsys battery` (disclose-hw-state).
- Screen: benches run with USB attached, screen on (`conditions.screen:
  "on-usb"`). Energy cells will run unplugged by hand; speed cells accept the
  USB-attached state exactly like the iPhone plugged-speed protocol.
- Sampler: litert_lm_main exposes **no temperature/top-p flags** →
  `conditions.sampler: "engine-default"`. This is a disclosed same-budget-rule
  deviation; llama-cli runs `--temp 0 --top-p 1` (greedy) like the Apple arms.
- Prompts: byte-identical to the Swift tasks via committed `prompts/text/*.txt`
  (`scripts/gen_task_prompts.py`).

## Quality axis

No on-device GSM8K in v1. Quality is a property of (artifact, runtime
numerics); the Mac instrument measures each checkpoint once, and Android rows
link by artifact `sha256` recorded per run. **Caveat that must travel with any
quality claim**: the Mac yardstick exercises macOS kernels, not Mali/Tensor
kernels — cross-device numerical parity is assumed, not verified. Phase-2
mitigation: a 25-question on-device spot check through
`--input_prompt_file` to test the assumption cheaply.

## Session discipline

Anchors first, payload interleaved per round (interleave-arms), ≥120 s
cooldown, publish every round. Cross-session comparisons only through anchors
(`regression_diff.py --anchors`), same as Apple.

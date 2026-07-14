# Camera VLM methodology — iPhone (ANE vs GPU, sustained)

How the live-camera vision-language rows are produced: run **Qwen3-VL 2B** against
the iPhone camera **continuously for 10 minutes**, once on the **ANE** (CoreML)
and once on the **GPU** (MLX), on the **same phone** and the **same scene**, and
log what diverges as the chip heats. Single-shot "describe this photo" is table
stakes; the point here is *sustained* operation, where thermal throttling decides
the winner.

This is the vision counterpart to the text [`energy-ios.md`](energy-ios.md) flow
and reuses its battery-delta power method verbatim.

## The headline number

**Sustained FPS over a 10-minute camera session, plus the power / heat / throttle
it cost.** One image tells the story: FPS-over-time for ANE and GPU on the same
axes (`scripts/vlm_throttle_chart.py` → `docs/charts/vlm-camera-throttle.png`).
The expected shape — the GPU starts faster, then sags as it heats; the ANE starts
lower and holds — is the same divergence the text sustained-throttle chart shows,
now on a vision workload.

### Sub-metrics (all logged automatically)

| Metric | Field | How it's measured |
|---|---|---|
| Sustained FPS | `metrics.sustainedFPS` | median of the 5 s-rolling inferences/sec over the back 80% of the run (warm-up dropped) |
| Throttle | `metrics.startFPS` → `endFPS`, `fpsDropPercent` | first-10% vs last-10% FPS means |
| TTFT | `metrics.ttftMedianMS`, `ttftP95MS` | per-inference gap from frame-in to first decoded token |
| ANE residency | `metrics.aneResidencyPercent` | `MLComputePlan.deviceUsage(for:)` over the vision encoder's ops (see below) |
| Peak temp | `metrics.peakThermalState` | `ProcessInfo.thermalState` (4-level; °C needs an out-of-band sensor log) |
| Power | `metrics.averagePackagePowerW`, `energyJoules` | battery-delta, whole-system — identical to the text energy flow |
| FPS / heat series | `metrics.fpsOverTime`, `thermalLevelOverTime` | 1 Hz, drives the chart |

## The harness

Both paths run the **same logical model (Qwen3-VL 2B Instruct)** so it is a
GPU-vs-ANE comparison on identical weights, not model-vs-model:

- **GPU path** — `MLXVLMRuntime`, MLX/Metal, weights
  `mlx-community/Qwen3-VL-2B-Instruct-4bit` (~1.78 GB).
- **ANE path** — `CoreMLVLMRuntime`, CoreML with `computeUnits =
  .cpuAndNeuralEngine`, driving `john-rocky/CoreML-LLM`'s real Qwen3-VL pipeline
  (public API since v1.9.0): `Qwen3VL2BVisionEncoder` (SigLIP + merger, 448×448 →
  196×2048 tokens + DeepStack, ANE-resident) feeding `Qwen3VL2BGenerator`
  (chunked INT8 decoder). The model is the published
  `mlboydaisuke/qwen3-vl-2b-coreml` bundle (~4.7 GB), fetched via
  `CoreMLLLM.ModelDownloader`. Conversion lives in CoreML-LLM
  (`conversion/build_qwen3_vl_2b_*.py`).

Both paths run today.

The loop (`CameraVLMRunner`) grabs the **latest** camera frame whenever the model
is free for the next inference — never a queue of stale frames — so *sustained FPS
= inferences/sec*, paced by the model, not the camera.

### Frame source — pick reproducibility or realism

`CameraFrameProvider` has two sources:

- **`.loopingAsset(URL)`** — a bundled reference clip decoded on a loop.
  **Deterministic input → reproducible numbers**: identical frames every run, on
  any device. Use this for the rows that go in the repo.
- **`.liveCamera`** — the back camera. Use this for the demo clip (point it at a
  dense, complex scene). Numbers from a live scene are illustrative, not
  reproducible — note the scene in the PR.

## Measuring ANE residency

Two layers, because the static prediction and the runtime truth are different
questions:

1. **Static (in-app, automatic).** `ANEResidency.percent(ofCompiledModelAt:)`
   loads `MLComputePlan` (iOS 17.4+) and reports the fraction of the vision
   encoder's `MLProgram` ops whose **preferred** compute device is the Neural
   Engine. This is a *prediction* of placement, logged as
   `metrics.aneResidencyPercent`.
2. **Runtime (out of band, ground truth).** An **Instruments** trace with the
   **Core ML** + **Neural Engine** + **Power** templates shows what actually ran
   where and the **per-subsystem ANE/GPU/CPU power**. The in-app battery-delta
   number is whole-system; per-subsystem mW only comes from Instruments. Attach
   that trace to the PR for any headline ANE claim.

## Reproduction

```bash
# 1. Build + install (XcodeGen regenerates the project with the camera tab).
cd ios/BenchmarkApp && xcodegen generate
# build & install to the device as usual, then on the phone:

# 2. Camera tab → pick backend (MLX/GPU or CoreML/ANE) → pick 10 min → Start.
#    Run it twice (once per backend) against the same scene, back to back.

# 3. Export results (Files app / Finder; app has UIFileSharingEnabled) and import:
python3 scripts/import_ios_export.py <exported.jsonl>   # → results/raw/*camera-vlm*

# 4. Draw the chart:
python3 scripts/vlm_throttle_chart.py                   # → docs/charts/vlm-camera-throttle.png
```

## Pre-flight checklist (hold constant across the two backends)

- [ ] **Same scene** for both runs — looping-asset source, or a fixed live framing.
- [ ] **Unplugged**, on battery, if you want the power number (USB charging →
      `energyJoules` is `nil`; see [`energy-ios.md`](energy-ios.md)).
- [ ] **Low Power Mode OFF**, **brightness fixed**, **Auto-Brightness OFF**.
- [ ] **Start at the same thermal state** (`nominal`) — let the phone cool between
      the two backends, or the second run starts pre-heated and looks worse.
- [ ] **Same duration** (600 s) and **same prompt / `maxTokens`** for both.
- [ ] **Start battery 80–95%**; note ambient room temperature (a hot room throttles).

## The demo clip protocol (20–30 s)

Point the camera at a **dense, complex scene** (a busy street, a shelf of objects)
and let it caption/count continuously. Screen-record the Camera tab — the HUD
already overlays FPS, thermal state, battery and ANE residency. Capture the GPU
run and the ANE run on the same scene and place them **side by side**: the GPU's
FPS visibly sagging while the ANE holds is the whole point.

## What the numbers are and aren't

- **Sustained, not peak.** A cold first inference is fast on either backend; this
  measures what survives 10 minutes of heat. Warm-up is dropped from `sustainedFPS`.
- **FPS is end-to-end per-frame**: preprocess + vision encode + decode of a short
  caption. A longer `maxTokens` lowers FPS uniformly — keep it equal across backends.
- **`aneResidencyPercent` is a prediction**, not a trace (see above).
- **Power is whole-system**, battery-delta — same caveats and ±1% quantization as
  the text energy flow. Per-subsystem ANE/GPU watts need Instruments.
- **thermalState is 4-level**, not °C. iOS gives apps nothing finer; `peakTemperatureC`
  stays `nil` unless a sensor log was attached out of band.

## The question this axis answers

On sustained **text** decode, the iPhone's GPU runtimes (MLX, LiteRT) shed
~50–60% of their throughput within minutes (MLX within ~60 s, LiteRT a few
minutes later) while the ANE (CoreML) holds (~65% retained) — a power-headroom
story. **Does the
same hold for a heavier vision-language workload, where the vision encoder adds a
conv-heavy ANE-friendly burst to every frame?** If the ANE keeps its FPS while the
GPU melts, "run the camera model on the ANE" stops being folklore and becomes a
measured result.

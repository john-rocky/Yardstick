# One PyTorch conv+transformer model on Android: ExecuTorch vs LiteRT (litert-torch) vs ONNX Runtime

Measured 2026-09-05. Same model, same frozen weights, same golden input, one instrumented-test process per device.

## Model and exports

`TinyHybridNet` (`tinyhybridnet-three-runtimes/model.py`): conv stem (3 conv + BatchNorm + GELU) -> 784 tokens x 96 -> one `nn.TransformerEncoder`-style block (`nn.MultiheadAttention`, 4 heads, MLP x4) -> LayerNorm -> mean pool -> Linear(10). 263,434 parameters, input `1x3x224x224` float32, output 10 logits. Weights: `torch.manual_seed(0)` with BatchNorm statistics and the positional embedding randomised so every op sees non-trivial values (`tinyhybridnet-three-runtimes/scripts/save_weights.py`).

Host: Apple M4 Max, macOS 27.0, Python 3.12.13, torch 2.13.0 (litert-torch 0.9.4 pins `torch<2.14`; executorch 1.4.0 needs `>=2.13`).

| path | tool | export | artifact |
|---|---|---|---|
| ExecuTorch | `executorch==1.4.0` | `torch.export` (strict) -> `to_edge_transform_and_lower(partitioner=[XnnpackPartitioner()])` -> `.pte` | `tinyhybridnet_xnnpack.pte` 1,073,320 B, 9 XNNPACK delegate calls; `native_layer_norm`, `select/expand/where`, `mean` stay on portable kernels. Undelegated control: `tinyhybridnet_portable.pte` 1,074,096 B |
| LiteRT | `litert-torch==0.9.4` | `litert_torch.convert(model, (x,)).export(path)` (torch.export -> StableHLO -> LiteRT flatbuffer), float32, no ONNX hop | `tinyhybridnet.tflite` 1,073,260 B, 84 ops, 278.3 M MACs (converter estimate) |
| ONNX Runtime | `torch.onnx.export(dynamo=False, opset 18)`, `torch.backends.mha.set_fastpath_enabled(False)` | `.onnx` | `tinyhybridnet.onnx` 1,065,628 B, 132 nodes |

All three exports succeeded on the first attempt with the model unmodified. (The legacy ONNX exporter can hit `aten::_native_multi_head_attention` in eval mode; disabling the MHA fast path avoids it.)

## Host parity (`tinyhybridnet-three-runtimes/scripts/check_parity.py`)

Golden input + 8 random inputs at input scales 1.0 / 3.0 / 0.1, compared with eager PyTorch:

| runtime (host) | worst max abs diff | worst max rel diff | argmax agreement |
|---|---:|---:|---|
| ExecuTorch 1.4.0 python runtime, XNNPACK | 1.07e-06 | 7.85e-05 | 9/9 |
| LiteRT (`ai-edge-litert` 2.2.0 Interpreter, CPU) | 8.35e-07 | 6.12e-05 | 9/9 |
| ONNX Runtime 1.29.0, CPU EP | 7.15e-07 | 8.71e-05 | 9/9 |

## On-device latency (one process, 5 warm-up + 30 timed runs of write input + run + read output; median)

Harness: one Android instrumented test (`tinyhybridnet-three-runtimes/android`, Kotlin, compileSdk 36) with `org.pytorch:executorch-android:1.4.0`, `com.google.ai.edge.litert:litert:2.2.0`, `com.microsoft.onnxruntime:onnxruntime-android:1.24.3`, all from Maven Central. Configurations run back to back with a 1.5 s pause. "default threads" is whatever each runtime picks when nothing is set.

### Galaxy S26 (SM-S942Q, Snapdragon SM8850, 8 cores, Android 16)

| runtime | backend | median ms | min ms | p90 ms | max abs diff vs PyTorch | argmax |
|---|---|---:|---:|---:|---:|---|
| ExecuTorch 1.4.0 | XNNPACK (default threads) | 10.59 | 10.35 | 14.10 | 4.8e-07 | same |
| ExecuTorch 1.4.0 | XNNPACK (4 threads) | 12.24 | 10.29 | 12.89 | 4.8e-07 | same |
| ExecuTorch 1.4.0 | portable kernels, no delegate | 242.81 | 241.97 | 245.25 | 3.7e-07 | same |
| LiteRT 2.2.0 | CPU / XNNPACK (default threads) | 8.77 | 6.66 | 9.35 | 3.9e-07 | same |
| LiteRT 2.2.0 | CPU / XNNPACK (4 threads) | 5.42 | 5.31 | 5.48 | 3.9e-07 | same |
| LiteRT 2.2.0 | GPU | fails: `CompiledModel.create` -> "Failed to compile model" | | | | |
| ONNX Runtime 1.24.3 | CPU EP (default threads) | 7.46 | 7.07 | 7.59 | 4.2e-07 | same |
| ONNX Runtime 1.24.3 | XNNPACK EP (default) | 12.58 | 12.51 | 12.68 | 5.1e-07 | same |
| ONNX Runtime 1.24.3 | XNNPACK EP (4 threads) | 8.95 | 8.80 | 9.55 | 5.1e-07 | same |
| ONNX Runtime 1.24.3 | NNAPI EP | 85.60 | 84.80 | 87.18 | 4.2e-07 | same |

### Pixel API 36 emulator (arm64-v8a system image, 6 vCPU, Android 16; host M4 Max)

| runtime | backend | median ms | min ms | p90 ms | max abs diff vs PyTorch | argmax |
|---|---|---:|---:|---:|---:|---|
| ExecuTorch 1.4.0 | XNNPACK (default threads) | 7.90 | 7.46 | 18.34 | 4.5e-07 | same |
| ExecuTorch 1.4.0 | XNNPACK (4 threads) | 6.55 | 6.36 | 6.86 | 4.5e-07 | same |
| ExecuTorch 1.4.0 | portable kernels, no delegate | 153.94 | 149.44 | 158.50 | 3.7e-07 | same |
| LiteRT 2.2.0 | CPU / XNNPACK (default threads) | 8.45 | 7.53 | 8.90 | 3.8e-07 | same |
| LiteRT 2.2.0 | CPU / XNNPACK (4 threads) | 2.52 | 2.26 | 2.77 | 3.8e-07 | same |
| LiteRT 2.2.0 | GPU | fails: "Failed to compile model" | | | | |
| ONNX Runtime 1.24.3 | CPU EP (default threads) | 4.07 | 3.87 | 4.24 | 3.7e-07 | same |
| ONNX Runtime 1.24.3 | XNNPACK EP (default) | 8.16 | 7.76 | 8.41 | 5.1e-07 | same |
| ONNX Runtime 1.24.3 | XNNPACK EP (4 threads) | 5.82 | 5.38 | 7.93 | 5.1e-07 | same |
| ONNX Runtime 1.24.3 | NNAPI EP | 50.88 | 48.45 | 52.41 | 4.2e-07 | same |

## How to read this

- Numerics: every configuration that runs reproduces PyTorch to ~5e-7 on the golden input; all three converters handle `nn.MultiheadAttention`, LayerNorm and GELU without model changes.
- The three CPU paths land within about 2x of each other on both devices once thread counts are comparable; the spread between "default" and "4 threads" inside one runtime is as large as the spread between runtimes, so thread policy, not converter, is the first knob.
- LiteRT GPU on both devices: the GPU delegate rejects the 5-D `RESHAPE` and a `TRANSPOSE` that `nn.MultiheadAttention` lowers to (logcat: "Tensor dimensions must be less than 5"), plans 54 of 84 ops on GPU, then `CompiledModel` creation fails instead of falling back. Observation only; a hand-written attention (explicit 4-D reshapes) would likely avoid it, not tested.
- ORT NNAPI EP is 8-12x slower than its own CPU EP on both devices; NNAPI is deprecated from Android 15 and on this phone routes to a reference path. Included because it is still the path many Android examples reach for first.
- ExecuTorch "portable" is the no-delegate control (23x slower than XNNPACK on the phone).
- Emulator numbers are host-CPU numbers with virtualisation overhead and should not be compared with the phone column; they are here because that is where most first checks happen.

## Files

`tinyhybridnet-three-runtimes/`: `model.py`, `scripts/` (weights, three exports, host parity), `android/` (the Gradle project with the instrumented test), `results/{s26,emulator}_results.json`, and the golden logits. The exported artifacts are not committed; `scripts/save_weights.py` then the three export scripts regenerate them (torch 2.13.0, executorch 1.4.0, litert-torch 0.9.4). Build: `gradle assembleDebug assembleDebugAndroidTest` with the artifacts copied into `android/app/src/main/assets/`.

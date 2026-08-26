# Leaderboard — current standings per platform

Neutral, reproducible standings for local LLM runtimes on-device. Every number
carries its recipe (quantization + engine build): arms run at their own best
available build, which is only a fair comparison when the recipe is visible.
Method and rules: `methodology/fairness-rules.md`. Raw records: `results/`.
Regenerate: `python3 scripts/build_summary.py && python3 scripts/render_leaderboard.py`.

<!-- BEGIN GENERATED: scripts/render_leaderboard.py -->

Generated from `results/summary/*.csv` (latest capture 2026-08-26) by `scripts/render_leaderboard.py` — do not edit inside the markers.

Headline task: **short-chat**, warm = median of same-session warm runs (cold-warm-split); other tasks and full history: RESULTS.md. Rows sort by warm decode; the recipe (quantization, engine build) is part of every row — a faster number under a different recipe is a different deployment profile, not a win.

† = quantization label carries the audited in-place correction (Gemma-4 `.litertlm` is the wNa8o8 mobile schema; early rows recorded "INT4 (QAT)" — quant-label-rule). mem MB = phys_footprint on Apple rows, RSS on Android rows (no footprint equivalent; methodology/android.md).

## mac

### Mac Studio (M4 Max)

**Gemma 4 E2B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| litert-lm | `litert-community/gemma-4-E2B-it-litert-lm` | wNa8o8 (int2/int4/int8 + int8 activations, QAT) | v0.16.0 | 156.0 | 128.9 | 797.9 | 41.0 | 677.8 | — | 2026-08-17 |
| mlx-swift | `mlx-community/gemma-4-e2b-it-4bit` | Q4 | pre-stamp | — | 185.4 | 446.5 | 68.0 | — | — | 2026-05-16 |
| llama.cpp | `unsloth/gemma-4-E2B-it-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 119.2 | 2143.6 | 41.0 | — | — | 2026-05-17 |
| coreml-llm | `coreml-llm/gemma4-e2b` | INT4 palettized | pre-stamp | — | 32.9 | — | 525.0 | — | — | 2026-05-17 |

**Gemma 4 E4B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/gemma-4-e4b-it-4bit` | Q4 | pre-stamp | — | 111.4 | 302.2 | 90.0 | — | — | 2026-05-17 |
| llama.cpp | `unsloth/gemma-4-E4B-it-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 80.6 | 1825.7 | 62.0 | — | — | 2026-05-17 |

**Qwen 2.5 0.5B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | Q4 | pre-stamp | — | 528.4 | 1715.5 | 26.0 | — | — | 2026-05-17 |
| llama.cpp | `bartowski/Qwen2.5-0.5B-Instruct-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 301.1 | 2994.8 | 17.0 | — | — | 2026-05-17 |
| coreml-llm | `coreml-llm/qwen2.5-0.5b` | FP16 | pre-stamp | — | 178.0 | — | 171.0 | — | — | 2026-05-17 |

**Qwen 3 0.6B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3-0.6B-4bit` | Q4 | 60bd0d7880c82980f9481f8be78862e9b63c58a3 | 555.9 | 556.0 | 2018.9 | 10.0 | 646.5 | — | 2026-08-17 |
| litert-lm | `litert-community/Qwen3-0.6B` | INT4 (mixed, blockwise gs32) | v0.16.0 | 309.9 | 311.3 | 566.9 | 59.0 | 795.4 | — | 2026-08-17 |

**Qwen 3 1.7B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3-1.7B-4bit` | Q4 | pre-stamp | 325.0 | 324.2 | 1009.3 | 20.0 | — | — | 2026-07-13 |
| litert-lm | `litert-local/qwen3-1.7b-int4` | INT4 (mixed, int8 embed) | pre-stamp | 171.1 | 94.9 | — | 350.0 | — | — | 2026-07-13 |

**Qwen 3.5 0.8B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3.5-0.8B-MLX-4bit` | Q4 | pre-stamp | — | 421.0 | 847.8 | 36.0 | — | — | 2026-05-17 |
| llama.cpp | `bartowski/Qwen_Qwen3.5-0.8B-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 203.2 | 2488.1 | 22.0 | — | — | 2026-05-17 |
| coreml-llm | `coreml-llm/qwen3.5-0.8b` | INT8 | pre-stamp | — | 58.4 | 56.7 | 405.0 | — | — | 2026-05-17 |

**Qwen 3.5 2B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3.5-2B-MLX-4bit` | Q4 | pre-stamp | — | 295.3 | 702.4 | 42.0 | — | — | 2026-05-17 |
| llama.cpp | `unsloth/Qwen3.5-2B-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 150.9 | 2504.5 | 29.0 | — | — | 2026-05-17 |
| coreml-llm | `coreml-llm/qwen3.5-2b` | INT8 | pre-stamp | — | 35.0 | 34.6 | 665.0 | — | — | 2026-05-17 |

**Qwen 3.5 9B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3.5-9B-MLX-4bit` | Q4 | pre-stamp | — | 90.0 | 240.8 | 95.0 | — | — | 2026-05-17 |
| llama.cpp | `unsloth/Qwen3.5-9B-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 57.6 | 132.8 | 141.0 | — | — | 2026-05-17 |

**SmolLM 3B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/SmolLM3-3B-4bit` | Q4 | pre-stamp | 196.6 | 196.0 | 1714.6 | 158.0 | — | — | 2026-07-13 |
| litert-lm | `litert-local/smollm3-3b` | INT4 | pre-stamp | — | 90.9 | — | 129.0 | — | — | 2026-07-13 |

<details><summary>single-arm cells (no cross-runtime comparison)</summary>

| model | runtime | artifact | quant | engine | warm tok/s | cold tok/s | captured |
|---|---|---|---|---|---|---|---|
| Gemma 3 1B | mlx-swift | `mlx-community/gemma-3-1b-it-4bit` | Q4 | pre-stamp | 328.2 | 328.4 | 2026-07-13 |
| LFM 2.5 350M | coreml-llm | `coreml-llm/lfm2.5-350m` | INT8 | pre-stamp | — | 58.9 | 2026-05-16 |
| Llama 3.2 1B | llama.cpp | `bartowski/Llama-3.2-1B-Instruct-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 269.5 | 2026-05-16 |
| apple-fm/default | apple-fm | `apple-fm/default` | Apple-quant (~2-4 bit, adapters) | pre-stamp | — | 84.3 | 2026-05-16 |
| litert-community/DeepSeek-R1-Distill-Qwen-1.5B | litert-lm | `litert-community/DeepSeek-R1-Distill-Qwen-1.5B` | INT8 | pre-stamp | 119.0 | 119.3 | 2026-07-13 |
| litert-community/Gemma3-1B-IT | litert-lm | `litert-community/Gemma3-1B-IT` | INT4 | pre-stamp | 181.3 | 182.0 | 2026-07-13 |
| litert-community/Phi-4-mini-instruct | litert-lm | `litert-community/Phi-4-mini-instruct` | INT8 | pre-stamp | 64.9 | 42.0 | 2026-07-13 |
| litert-community/Qwen3-4B | litert-lm | `litert-community/Qwen3-4B` | INT4 (mixed, blockwise gs32) | pre-stamp | 110.9 | 111.2 | 2026-07-13 |
| litert-community/Qwen3-8B | litert-lm | `litert-community/Qwen3-8B` | INT4 (mixed, blockwise gs32) | pre-stamp | — | 62.4 | 2026-06-17 |
| litert-community/TinySwallow-1.5B-Instruct | litert-lm | `litert-community/TinySwallow-1.5B-Instruct` | INT8 | pre-stamp | 120.6 | 120.4 | 2026-07-13 |
| litert-community/VibeThinker-1.5B | litert-lm | `litert-community/VibeThinker-1.5B` | INT8 | pre-stamp | 120.4 | 120.3 | 2026-07-13 |
| litert-local/llama32-3b | litert-lm | `litert-local/llama32-3b` | INT4 | pre-stamp | 93.3 | 92.9 | 2026-07-13 |
| litert-local/minicpm5-1b | litert-lm | `litert-local/minicpm5-1b` | INT4 (ekv1024) | pre-stamp | — | 239.3 | 2026-06-17 |
| litert-local/ministral3-3b | litert-lm | `litert-local/ministral3-3b` | INT4 | pre-stamp | 92.0 | 91.7 | 2026-07-13 |
| litert-local/olmo2-1b | litert-lm | `litert-local/olmo2-1b` | INT4 | pre-stamp | 136.3 | 135.1 | 2026-07-13 |
| mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit | mlx-swift | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` | Q4 | pre-stamp | 332.8 | 330.8 | 2026-07-13 |
| mlx-community/LFM2-350M-4bit | mlx-swift | `mlx-community/LFM2-350M-4bit` | Q4 | pre-stamp | — | 1024.2 | 2026-06-17 |
| mlx-community/Llama-3.2-3B-Instruct-4bit | mlx-swift | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Q4 | pre-stamp | 208.1 | 207.6 | 2026-07-13 |
| mlx-community/MiniCPM5-1B-4bit | mlx-swift | `mlx-community/MiniCPM5-1B-4bit` | Q4 | pre-stamp | — | 526.2 | 2026-06-17 |
| mlx-community/Phi-4-mini-instruct-4bit | mlx-swift | `mlx-community/Phi-4-mini-instruct-4bit` | Q4 | pre-stamp | 169.1 | 169.0 | 2026-07-13 |
| mlx-community/Qwen3-4B-4bit | mlx-swift | `mlx-community/Qwen3-4B-4bit` | Q4 | pre-stamp | 162.2 | 163.1 | 2026-07-13 |
| mlx-community/Qwen3-8B-4bit | mlx-swift | `mlx-community/Qwen3-8B-4bit` | Q4 | pre-stamp | — | 98.3 | 2026-06-17 |
| mlx-community/TinySwallow-1.5B-Instruct-4bit | mlx-swift | `mlx-community/TinySwallow-1.5B-Instruct-4bit` | Q4 | pre-stamp | 328.0 | 328.8 | 2026-07-13 |
| own/DeepSeek-R1-1.5B-int4-BOCTAV4 | litert-lm | `own/DeepSeek-R1-1.5B-int4-BOCTAV4` | INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed) | pre-stamp | 136.7 | 136.6 | 2026-07-14 |
| own/Phi-4-mini-int4-BOCTAV4-128 | litert-lm | `own/Phi-4-mini-int4-BOCTAV4-128` | INT4 (BOCTAV4 blockwise-128 OCTAV, int8 embed, static-rope) | pre-stamp | 85.4 | 83.5 | 2026-07-14 |
| own/TinySwallow-1.5B-int4-BOCTAV4 | litert-lm | `own/TinySwallow-1.5B-int4-BOCTAV4` | INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed) | pre-stamp | 137.2 | 136.4 | 2026-07-14 |
| own/VibeThinker-1.5B-int4-BOCTAV4 | litert-lm | `own/VibeThinker-1.5B-int4-BOCTAV4` | INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed) | pre-stamp | 137.4 | 137.4 | 2026-07-14 |

</details>

### arm64

**Gemma 4 E2B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/gemma-4-e2b-it-4bit` | Q4 | pre-stamp | 196.9 | 197.9 | 626.1 | 34.0 | 2890.9 | — | 2026-07-27 |
| litert-lm | `litert-community/gemma-4-E2B-it-litert-lm` | wNa8o8 (int2/int4/int8 + int8 activations, QAT)† | pre-stamp | 154.3 | 155.4 | 771.0 | 34.0 | 682.0 | — | 2026-07-27 |
| llama.cpp | `unsloth/gemma-4-E2B-it-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | 149.2 | 149.4 | 7484.8 | 34.5 | 362.2 | — | 2026-07-27 |

<details><summary>single-arm cells (no cross-runtime comparison)</summary>

| model | runtime | artifact | quant | engine | warm tok/s | cold tok/s | captured |
|---|---|---|---|---|---|---|---|
| Gemma 4 E2B (QAT OptiQ) | mlx-swift | `mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit` | INT4 (QAT, OptiQ) | pre-stamp | 164.7 | 165.7 | 2026-07-27 |
| SmolLM 3B | litert-lm | `litert-local/smollm3-3b` | INT4 | pre-stamp | 67.3 | — | 2026-07-13 |

</details>

## ios

### iPhone 17 Pro

**Gemma 4 E2B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| litert-lm | `litert-community/gemma-4-E2B-it-litert-lm` | wNa8o8 (int2/int4/int8 + int8 activations, QAT) | v0.16.0 | 51.0 ⚠spread 21% | 55.4 | 408.7 | 80.5 | 479.7 | — | 2026-08-19 |
| mlx-swift | `mlx-community/gemma-4-e2b-it-4bit` | Q4 | pre-stamp | 47.8 | 44.1 | 183.6 | 218.5 | 2999.8 | — | 2026-07-30 |
| llama.cpp | `unsloth/gemma-4-E2B-it-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | 38.9 | 38.5 | 1710.5 | 108.0 | 198.6 | — | 2026-07-30 |
| core-ai | `core-ai/gemma4-e2b-gpu` | int4 q4_0 (QAT transplant) | pre-stamp | — | 46.9 | 23.5 | 3133.0 | 455.3 | — | 2026-07-30 |

**Qwen 3 0.6B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3-0.6B-4bit` | Q4 | 60bd0d7880c82980f9481f8be78862e9b63c58a3 | 178.8 | 178.7 | 533.6 | 37.0 | 488.2 | — | 2026-08-26 |
| core-ai | `core-ai/qwen3-0.6b-gpu` | INT4 (dynamic, macOS-26-era export) | 0.2.0+static-inputs-patch | 147.4 | 152.3 | 954.1 | 23.0 | 180.2 | — | 2026-08-26 |
| core-ai | `core-ai/qwen3-0.6b-ane-june` | mixed 4/8-bit (June static export) | pre-stamp | 122.4 | 124.9 | 675.6 | 29.0 | — | — | 2026-07-13 |
| litert-lm | `litert-community/Qwen3-0.6B` | INT4 (mixed, blockwise gs32) | v0.16.0 | 121.6 | 120.6 | 183.9 | 151.5 | 465.4 | — | 2026-08-26 |
| core-ai | `core-ai/qwen3-0.6b-ane` | 4-bit palettized (uniform g32) | pre-stamp | 116.9 | 118.7 | 697.3 | 28.5 | — | — | 2026-07-13 |
| coreml-llm | `coreml-llm/qwen3-0.6b` | INT8 palettized | pre-stamp | — | 37.8 | 33.2 | 572.0 | — | — | 2026-06-17 |

**Qwen 3 1.7B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| core-ai | `core-ai/qwen3-1.7b-gpu-june` | INT4 (dynamic, June export) | pre-stamp | 67.6 | 67.9 | 457.6 | 45.0 | — | — | 2026-07-13 |
| mlx-swift | `mlx-community/Qwen3-1.7B-4bit` | Q4 | pre-stamp | 65.1 | 63.9 | 248.6 | 77.5 | — | — | 2026-07-13 |
| litert-lm | `litert-local/qwen3-1.7b-int4` | INT4 (mixed, int8 embed) | pre-stamp | 49.3 | 46.8 | — | 855.5 | — | — | 2026-07-13 |
| litert-lm | `litert-local/qwen3-1.7b` | INT8 (dynamic, ekv1024) | pre-stamp | — | 30.2 | — | 453.0 | — | — | 2026-06-19 |
| core-ai | `core-ai/qwen3-1.7b-gpu` | INT4 (dynamic) | pre-stamp | — | 64.7 | 763.6 | 31.0 | — | — | 2026-07-14 |

**Qwen 3.5 2B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/Qwen3.5-2B-MLX-4bit` | Q4 | pre-stamp | — | 61.2 | 249.3 | 103.0 | — | — | 2026-05-27 |
| llama.cpp | `unsloth/Qwen3.5-2B-GGUF/Q4_K_M` | Q4_K_M | pre-stamp | — | 38.7 | 2503.9 | 96.0 | — | — | 2026-05-27 |
| coreml-llm | `coreml-llm/qwen3.5-2b` | INT8 | pre-stamp | — | 28.1 | 27.3 | 844.0 | — | — | 2026-05-28 |

**SmolLM 3B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-swift | `mlx-community/SmolLM3-3B-4bit` | Q4 | pre-stamp | 34.3 | 34.6 | 657.9 | 400.5 | — | — | 2026-07-13 |
| litert-lm | `litert-local/smollm3-3b` | INT4 | pre-stamp | 22.9 ⚠spread 11% | 22.7 | — | 1453.5 | — | — | 2026-07-13 |
| core-ai | `core-ai/smollm3-3b-gpu` | INT4 (dynamic) | pre-stamp | — | 34.6 | 877.9 | 308.0 | — | — | 2026-07-14 |

<details><summary>single-arm cells (no cross-runtime comparison)</summary>

| model | runtime | artifact | quant | engine | warm tok/s | cold tok/s | captured |
|---|---|---|---|---|---|---|---|
| Gemma 3 1B | mlx-swift | `mlx-community/gemma-3-1b-it-4bit` | Q4 | pre-stamp | 104.9 | 104.3 | 2026-07-13 |
| Gemma 4 E2B (CQ4 shipped default) | cactus | `Cactus-Compute/gemma-4-E2B-it-cq4` | CQ4 (rotation+codebook PTQ, calibrated) | pre-stamp | — | 36.7 | 2026-07-30 |
| Gemma 4 E2B (CQ4 uncalibrated) | cactus | `Cactus-Compute/gemma-4-E2B-it-cq4-uncalibrated` | CQ4 (rotation+codebook PTQ, uncalibrated pre-07-09 build) | pre-stamp | — | 50.6 | 2026-07-19 |
| Gemma 4 E2B (QAT OptiQ) | mlx-swift | `mlx-community/gemma-4-e2b-it-qat-OptiQ-4bit` | INT4 (QAT, OptiQ) | pre-stamp | 36.2 | 36.1 | 2026-07-30 |
| core-ai/deepseek-r1-1.5b-ane | core-ai | `core-ai/deepseek-r1-1.5b-ane` | 4-bit palettized (uniform g32) | pre-stamp | — | 81.9 | 2026-07-14 |
| core-ai/deepseek-r1-1.5b-gpu | core-ai | `core-ai/deepseek-r1-1.5b-gpu` | INT4 (dynamic) | pre-stamp | 74.8 | 75.8 | 2026-07-13 |
| core-ai/gemma3-1b-gpu | core-ai | `core-ai/gemma3-1b-gpu` | INT4 (dynamic) | pre-stamp | — | 94.3 | 2026-07-14 |
| core-ai/lfm2.5-1.2b-gpu | core-ai | `core-ai/lfm2.5-1.2b-gpu` | int8hu block32 sym (untied head) | 0.2.0+static-inputs-patch | 45.5 | 45.7 | 2026-08-26 |
| core-ai/llama-3.2-3b-ane | core-ai | `core-ai/llama-3.2-3b-ane` | 4-bit palettized (uniform g32) | pre-stamp | 38.0 | — | 2026-07-13 |
| core-ai/llama-3.2-3b-gpu | core-ai | `core-ai/llama-3.2-3b-gpu` | INT4 (dynamic) | pre-stamp | — | 35.5 | 2026-07-14 |
| core-ai/minicpm5-1b-gpu | core-ai | `core-ai/minicpm5-1b-gpu` | INT8 (sym, dynamic) | 0.2.0+static-inputs-patch | 72.1 | 72.2 | 2026-08-26 |
| core-ai/ministral-3b-gpu | core-ai | `core-ai/ministral-3b-gpu` | INT4 (dynamic) | pre-stamp | — | 29.3 | 2026-07-14 |
| core-ai/olmo2-1b-ane | core-ai | `core-ai/olmo2-1b-ane` | 4-bit palettized (uniform g32) | pre-stamp | — | 96.3 | 2026-07-14 |
| core-ai/olmo2-1b-gpu | core-ai | `core-ai/olmo2-1b-gpu` | INT4 (dynamic) | pre-stamp | — | 92.5 | 2026-07-14 |
| core-ai/qwen3-4b-ane | core-ai | `core-ai/qwen3-4b-ane` | 4-bit palettized (uniform g32) | pre-stamp | 30.0 | 27.0 | 2026-07-13 |
| core-ai/qwen3-4b-gpu | core-ai | `core-ai/qwen3-4b-gpu` | INT4 (dynamic) | pre-stamp | 26.4 | 27.0 | 2026-07-13 |
| core-ai/tinyswallow-1.5b-ane | core-ai | `core-ai/tinyswallow-1.5b-ane` | 4-bit palettized (uniform g32) | pre-stamp | — | 72.9 | 2026-07-14 |
| core-ai/tinyswallow-1.5b-gpu | core-ai | `core-ai/tinyswallow-1.5b-gpu` | INT4 (dynamic) | pre-stamp | — | 69.7 | 2026-07-14 |
| core-ai/vibethinker-1.5b-ane | core-ai | `core-ai/vibethinker-1.5b-ane` | 4-bit palettized (uniform g32) | pre-stamp | 73.8 | 73.4 | 2026-07-13 |
| core-ai/vibethinker-1.5b-gpu | core-ai | `core-ai/vibethinker-1.5b-gpu` | INT4 (dynamic) | pre-stamp | — | 71.0 | 2026-07-14 |
| litert-community/DeepSeek-R1-Distill-Qwen-1.5B | litert-lm | `litert-community/DeepSeek-R1-Distill-Qwen-1.5B` | INT8 | v0.16.0 | 26.7 | 31.2 | 2026-08-24 |
| litert-community/Gemma3-1B-IT | litert-lm | `litert-community/Gemma3-1B-IT` | INT4 | pre-stamp | 72.1 | 72.7 | 2026-07-13 |
| litert-community/LFM2.5-1.2B-Instruct | litert-lm | `litert-community/LFM2.5-1.2B-Instruct` | int4_gpu (litert-community descriptor) | v0.16.0 | 69.5 | 69.9 | 2026-08-26 |
| litert-community/MiniCPM5-1B | litert-lm | `litert-community/MiniCPM5-1B` | wi4b32_wi8_afp32 (gpu-opt) | v0.16.0 | 34.2 | 36.2 | 2026-08-26 |
| litert-community/Phi-4-mini-instruct | litert-lm | `litert-community/Phi-4-mini-instruct` | INT8 | pre-stamp | 11.6 | 11.4 | 2026-07-24 |
| litert-community/Qwen3-4B | litert-lm | `litert-community/Qwen3-4B` | INT4 (mixed, blockwise gs32) | v0.16.0 | 16.1 | 23.1 | 2026-08-19 |
| litert-community/TinySwallow-1.5B-Instruct | litert-lm | `litert-community/TinySwallow-1.5B-Instruct` | INT8 | pre-stamp | 29.4 | 29.2 | 2026-07-23 |
| litert-community/VibeThinker-1.5B | litert-lm | `litert-community/VibeThinker-1.5B` | INT8 | pre-stamp | 29.3 | 30.6 | 2026-07-23 |
| litert-local/llama32-3b | litert-lm | `litert-local/llama32-3b` | INT4 | pre-stamp | 18.2 | 19.2 | 2026-07-13 |
| litert-local/ministral3-3b | litert-lm | `litert-local/ministral3-3b` | INT4 | pre-stamp | 18.4 | 19.0 | 2026-07-14 |
| litert-local/olmo2-1b | litert-lm | `litert-local/olmo2-1b` | INT4 | pre-stamp | 20.6 | 26.4 | 2026-07-13 |
| mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit | mlx-swift | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` | Q4 | pre-stamp | 73.2 | 74.5 | 2026-07-13 |
| mlx-community/Llama-3.2-3B-Instruct-4bit | mlx-swift | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Q4 | pre-stamp | 33.8 | 34.4 | 2026-07-13 |
| mlx-community/Phi-4-mini-instruct-4bit | mlx-swift | `mlx-community/Phi-4-mini-instruct-4bit` | Q4 | pre-stamp | 29.3 | 29.5 | 2026-07-13 |
| mlx-community/Qwen3-4B-4bit | mlx-swift | `mlx-community/Qwen3-4B-4bit` | Q4 | pre-stamp | 28.0 | 28.4 | 2026-07-13 |
| mlx-community/TinySwallow-1.5B-Instruct-4bit | mlx-swift | `mlx-community/TinySwallow-1.5B-Instruct-4bit` | Q4 | pre-stamp | 72.9 | 72.6 | 2026-07-13 |
| own/DeepSeek-R1-1.5B-int4-BOCTAV4 | litert-lm | `own/DeepSeek-R1-1.5B-int4-BOCTAV4` | INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed) | pre-stamp | 44.5 | 45.8 | 2026-07-24 |
| own/Phi-4-mini-int4-BOCTAV4-128 | litert-lm | `own/Phi-4-mini-int4-BOCTAV4-128` | INT4 (BOCTAV4 blockwise-128 OCTAV, int8 embed, static-rope) | pre-stamp | 17.5 | 17.6 | 2026-07-24 |
| own/TinySwallow-1.5B-int4-BOCTAV4 | litert-lm | `own/TinySwallow-1.5B-int4-BOCTAV4` | INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed) | pre-stamp | 45.6 | 46.8 | 2026-07-23 |
| own/VibeThinker-1.5B-int4-BOCTAV4 | litert-lm | `own/VibeThinker-1.5B-int4-BOCTAV4` | INT4 (BOCTAV4 blockwise-32 OCTAV, int8 embed) | pre-stamp | 45.7 | 46.8 | 2026-07-23 |

</details>

**Structural exclusions** (failed-runs-stay — the row exists, the reason is the datum):

- `core-ai core-ai/phi-4-mini-gpu short-chat` — partial-rotary-unsupported
- `core-ai core-ai/qwen3-0.6b-ane short-chat` — invoke-fail-staticshape-logitsinference
- `core-ai core-ai/qwen3-1.7b-ane short-chat` — invoke-fail-bd71203

## android

### Pixel 8a

**Qwen 3 0.6B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | `unsloth/Qwen3-0.6B-GGUF` | Q4_K_M | b8999 | — | 29.7 | 221.4 | — | 1253.5 | — | 2026-08-17 |
| litert-lm-gpu | `litert-community/Qwen3-0.6B` | INT4 (mixed, blockwise gs32) | v0.16.0 | — | 15.2 | 41.5 | 550.0 | 764.6 | — | 2026-08-17 |
| litert-lm-cpu | `litert-community/Qwen3-0.6B` | INT4 (mixed, blockwise gs32) | v0.16.0 | — | 15.3 | 9.8 | 2110.0 | 1250.1 | — | 2026-08-17 |

<details><summary>single-arm cells (no cross-runtime comparison)</summary>

| model | runtime | artifact | quant | engine | warm tok/s | cold tok/s | captured |
|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | litert-lm-gpu | `litert-community/gemma-4-E2B-it-litert-lm` | wNa8o8 (int2/int4/int8 + int8 activations, QAT) | v0.16.0 | — | 8.4 | 2026-08-17 |

</details>

### Galaxy S26

**Qwen 3 0.6B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | `unsloth/Qwen3-0.6B-GGUF` | Q4_K_M | b8999 | — | 109.7 | 253.9 | — | 1251.6 | — | 2026-08-25 |
| litert-lm-gpu | `litert-community/Qwen3-0.6B` | INT4 (mixed, blockwise gs32) | v0.16.0 | — | 54.4 | 153.1 | 150.0 | 550.6 | — | 2026-08-25 |
| litert-lm-cpu | `litert-community/Qwen3-0.6B` | INT4 (mixed, blockwise gs32) | v0.16.0 | — | 27.7 | 23.8 | 870.0 | 2022.6 | — | 2026-08-25 |

**litert-community/DeepSeek-R1-Distill-Qwen-1.5B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| litert-lm-gpu | `litert-community/DeepSeek-R1-Distill-Qwen-1.5B` | INT8 | v0.16.0 | — | 22.9 | 101.5 | 200.0 | 558.2 | — | 2026-08-25 |
| litert-lm-cpu | `litert-community/DeepSeek-R1-Distill-Qwen-1.5B` | INT8 | v0.16.0 | — | 26.0 | 25.5 | 670.0 | 1950.3 | — | 2026-08-25 |

**litert-community/LFM2.5-1.2B-Instruct**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| litert-lm-gpu | `litert-community/LFM2.5-1.2B-Instruct` | int4_gpu (litert-community descriptor) | v0.16.0 | — | 45.6 | 184.9 | 140.0 | 477.4 | — | 2026-08-25 |
| litert-lm-cpu | `litert-community/LFM2.5-1.2B-Instruct` | int4 (litert-community descriptor) | v0.16.0 | — | 51.2 | 23.4 | 920.0 | 284.3 | — | 2026-08-25 |

**litert-community/MiniCPM5-1B**

| runtime | artifact | quant | engine | warm tok/s | cold tok/s | prefill tok/s | TTFT ms | mem MB | GSM8K | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| litert-lm-gpu | `litert-community/MiniCPM5-1B` | wi4b32_wi8_afp32 (gpu-opt) | v0.16.0 | — | 54.5 | 218.8 | 110.0 | 364.2 | — | 2026-08-25 |
| litert-lm-cpu | `litert-community/MiniCPM5-1B` | wi4b32_wi8_afp32 | v0.16.0 | — | 30.4 | 34.8 | 590.0 | 1047.3 | — | 2026-08-25 |

<details><summary>single-arm cells (no cross-runtime comparison)</summary>

| model | runtime | artifact | quant | engine | warm tok/s | cold tok/s | captured |
|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | litert-lm-gpu | `litert-community/gemma-4-E2B-it-litert-lm` | wNa8o8 (int2/int4/int8 + int8 activations, QAT) | v0.16.0 | — | 27.8 | 2026-08-25 |
| bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF | llama.cpp | `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | Q4_K_M | b8999 | — | 46.2 | 2026-08-25 |

</details>

<!-- END GENERATED: scripts/render_leaderboard.py -->

# Nemotron-3 Nano on Apple Silicon: llama.cpp vs MLX (batch 1, greedy, 4-bit)

Measured 2026-09-04 on one machine. Structured copy of the same numbers: `nemotron3-nano-apple-silicon-bench.json`.

## Hardware and software

- Apple M4 Max, 128 GiB unified memory (`hw.memsize` = 137438953472), macOS 27.0 (26A5416b), Xcode 27.0 beta 5 (27A5237l).
- llama.cpp: Homebrew package, build **8680**, commit **15f786e65**, backends reported `BLAS,MTL`, 12 threads. The Homebrew binary loaded both `nemotron_h` (4B) and `nemotron_h_moe` (30B-A3B) GGUFs, so no source build was needed.
- MLX: `mlx` **0.32.2**, `mlx-lm` **0.31.3**, Python 3.12.13 in a fresh venv.
- Mains power, no other GPU workload during measurements.

## Results

| Model | Stack (version) | Weights | Prompt tok/s | Decode tok/s | Peak memory | Notes |
|---|---|---|---:|---:|---:|---|
| Nemotron-3 Nano 30B-A3B | llama.cpp Metal (b8680, 15f786e65) | Q4_K_M, unsloth, 24.57 GB (22.88 GiB buffer) | pp512 **1212.81 ± 3.02** | tg256 **86.19 ± 0.47** | 24.38 GB max RSS | `-fa 0`: 1207.08 ± 3.25 / 85.73 ± 0.34 |
| Nemotron-3 Nano 30B-A3B | MLX mlx-lm 0.31.3 (mlx 0.32.2) | 4-bit, mlx-community, 17.78 GB | 642.6 (133-tok prompt) | **159.7** (median of 159.71 / 159.60 / 159.78) | 18.35 GB (mlx peak) | reasoning trace on by default |
| Nemotron-3 Nano 4B | llama.cpp Metal (b8680, 15f786e65) | Q4_K_M, nvidia, 2.84 GB (2.63 GiB buffer) | pp512 **1308.83 ± 1.05** | tg256 **88.38 ± 0.22** | 3.02 GB max RSS | |
| Nemotron-3 Nano 4B | MLX mlx-lm 0.31.3 (mlx 0.32.2) | 4-bit, mlx-community, 2.24 GB | 1058.7 (133-tok prompt) | **176.8** (median of 176.98 / 175.80 / 176.76) | 3.21 GB (mlx peak) | reasoning trace on by default |

How to read the columns:

- llama.cpp numbers are `llama-bench` output, mean ± std over `-r 3`. A second identical run wrapped in `/usr/bin/time -l` (for RSS) gave 1214.61 ± 3.01 / 86.84 ± 0.57 (30B-A3B) and 1309.95 ± 1.69 / 88.03 ± 0.13 (4B), i.e. run-to-run drift is under 1%.
- MLX numbers are what `mlx_lm generate` prints (`Prompt:` / `Generation:` tokens-per-sec, `Peak memory`), median of three separate processes. The MLX prompt column is a 133-token chat prompt, not a 512-token batch, so it is not the same measurement as pp512. The first 30B-A3B MLX process reported a slower prompt phase (154 tok/s, cold start); the other two agreed at 642.
- Peak memory is a different metric per stack: max resident set size for llama.cpp (includes the mmapped weight file), Metal peak allocation as reported by mlx-lm for MLX. Do not compare them across stacks at face value.
- The unsloth Q4_K_M for 30B-A3B is 24.57 GB, about 6.2 bits/weight averaged over 31.58 B params; the mlx-community 4-bit is 17.78 GB. The two 30B-A3B rows do not move the same number of bytes per token.
- Observation only, not investigated: on llama.cpp the dense 4B decodes at almost the same rate as the 30B-A3B (88 vs 86 tok/s) despite a 9x smaller weight file, so decode in this build does not look weight-bandwidth-bound for these `nemotron_h` models.
- `llama-completion` single-turn runs (34 prompt tokens, 160 max tokens) reported eval rates of 86.59 tok/s (30B-A3B) and 88.48 tok/s (4B), consistent with llama-bench.

## Commands

Downloads (models deleted after the run):

```sh
export HF_HOME=/private/tmp/odp-bench/hf HF_HUB_DISABLE_XET=1
huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF --include "*Q4_K_M*"          # rev 9ad8b366
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF --include "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"   # rev ba223d14
huggingface-cli download mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit   # rev 832f602e
huggingface-cli download mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit        # rev c4d79ba1
```

llama.cpp (Homebrew `llama.cpp` 8680):

```sh
llama-bench -m Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf -p 512 -n 256 -ngl 99 -fa 1 -r 3
llama-bench -m Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf -p 512 -n 256 -ngl 99 -fa 0 -r 3
llama-bench -m NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf -p 512 -n 256 -ngl 99 -fa 1 -r 3
/usr/bin/time -l llama-bench -m <same gguf> -p 512 -n 256 -ngl 99 -fa 1 -r 3     # "maximum resident set size"
llama-completion -m <gguf> -ngl 99 -fa on --temp 0 -n 160 -st --simple-io \
  -p "In two or three sentences, what is unified memory on Apple Silicon and why does it matter for running LLMs locally?"
```

MLX (venv: `pip install mlx-lm`, which pulled mlx 0.32.2 / mlx-lm 0.31.3). `python -m mlx_lm.generate` is deprecated in this version and prints a warning; `python -m mlx_lm generate` is the same entry point. Run three times each:

```sh
python -m mlx_lm generate --model mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit --prompt "$(cat prompt.txt)" --max-tokens 256 --temp 0
python -m mlx_lm generate --model mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit        --prompt "$(cat prompt.txt)" --max-tokens 256 --temp 0
```

`prompt.txt` (101 words; 133 tokens after the chat template):

> You are a helpful assistant. In about 150 words, explain to a software engineer who has never used Apple Silicon why unified memory matters for running large language models locally. Cover three points: how the GPU and CPU share the same physical memory pool, why this lets a laptop load a model that would not fit in a discrete GPU's VRAM, and what the practical trade-off is compared with a desktop GPU in terms of memory bandwidth. Keep the tone neutral and technical, avoid marketing language, and end with one sentence about when a discrete GPU is still the better choice.

## Coherence samples (first ~200 characters of one greedy output each)

Nemotron-3 Nano 30B-A3B, llama.cpp (no reasoning block appeared in the llama.cpp chat output):

> **Unified memory** on Apple Silicon is a single, coherent pool of RAM that the CPU, GPU, and Neural Engine can all access directly without copying data between separate memory spaces. Because the same physical memory serves all cores, data (like model weights or activations) …

Nemotron-3 Nano 30B-A3B, MLX (mlx-lm applies the model's chat template, which turns reasoning on, so the output starts with a reasoning trace):

> We need to produce about 150 words, neutral technical tone, covering three points: unified memory sharing, why it lets laptop load model that wouldn't fit in discrete GPU VRAM, and practical trade-off compared with desktop GPU in terms of memory bandwidth. […] Draft: "Unified memory on Apple Silicon is a single address space that both the CPU and GPU can access without copying data across separate pools. Because the memory is physically shared, a model's tensors reside in RAM that the GPU can read directly …

Nemotron-3 Nano 4B, llama.cpp:

> **Unified memory on Apple Silicon means that the CPU, GPU, and memory controller share a single pool of memory, allowing them to communicate and access data seamlessly.** This eliminates the need for data to be copied between systems, making it faster and more efficient. …

Nemotron-3 Nano 4B, MLX (reasoning trace, then `</think>`, then the answer):

> We need to produce about 150 words, neutral technical tone, covering three points: unified memory, GPU/CPU share same physical memory pool, allows laptop to load model that wouldn't fit in discrete GPU VRAM […] `</think>` Unified memory means the CPU and GPU access a single, shared physical memory pool rather than separate VRAM and system RAM. Because the model's tensors reside in this pool, a laptop can load a large language model that would otherwise exceed the discrete GPU's limited VRAM …

## Granite-4.0-H-Tiny (second hybrid data point)

Same protocol, same binaries. `ibm-granite/granite-4.0-h-tiny` is a hybrid Mamba-2 / Transformer MoE (`granitehybrid`, 6.94 B params, 64 experts, 6 used, llama.cpp size class "7B.A1B").

| Model | Stack (version) | Weights | Prompt tok/s | Decode tok/s | Peak memory | Notes |
|---|---|---|---:|---:|---:|---|
| Granite-4.0-H-Tiny | llama.cpp Metal (b8680, 15f786e65) | Q4_K_M, unsloth rev 56b37fdc, 4.25 GB (3.96 GiB buffer) | pp512 **2227.99 ± 6.47** | tg256 **117.31 ± 0.92** | 4.43 GB max RSS | RSS run: 2222.74 ± 9.35 / 116.40 ± 0.16; llama-completion eval 116.79 tok/s |
| Granite-4.0-H-Tiny | MLX mlx-lm 0.31.3 (mlx 0.32.2) | 4-bit affine gs64 (routers 8-bit), lmstudio-community rev a0f1fb8f, 3.91 GB | 1525 (123-tok prompt) | **202.2** (median of 201.49 / 202.18 / 202.83) | 4.44 GB (mlx peak) | stopped at 244 tokens (EOS); no reasoning trace |

The MLX weights come from `lmstudio-community/granite-4.0-h-tiny-MLX-4bit` rather than `mlx-community/granite-4.0-h-tiny-4bit`: the mlx-community download stayed under 5 MB/s (0-1 MB/s on resume) for over ten minutes and was abandoned; the lmstudio-community file came down at 11 MB/s. Both are plain 4-bit mlx-lm conversions of the same base model. The first MLX process again reported a slow prompt phase (280 tok/s, cold start); the other two agreed at 1525-1565.

```sh
huggingface-cli download unsloth/granite-4.0-h-tiny-GGUF --include "granite-4.0-h-tiny-Q4_K_M.gguf"
llama-bench -m granite-4.0-h-tiny-Q4_K_M.gguf -p 512 -n 256 -ngl 99 -fa 1 -r 3
huggingface-cli download lmstudio-community/granite-4.0-h-tiny-MLX-4bit
python -m mlx_lm generate --model lmstudio-community/granite-4.0-h-tiny-MLX-4bit --prompt "$(cat prompt.txt)" --max-tokens 256 --temp 0
```

Coherence samples (same prompts as above):

llama.cpp:

> Unified memory on Apple Silicon refers to the architecture where the CPU and GPU share a single pool of memory, allowing for seamless data transfer between the two components. This is important for running large language models (LLMs) locally because it enables efficient memory management …

MLX:

> Unified memory is a key feature of Apple Silicon that significantly impacts how large language models are run locally on laptops. Firstly, in Apple Silicon, the GPU and CPU share the same physical memory pool, which means there's no discrete VRAM. …

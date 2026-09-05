# Hybrid Mamba-2 / Transformer LLMs on Apple Silicon, part 2: Falcon-H1 and Granite-4.0-H small sizes (llama.cpp vs MLX)

Measured 2026-09-05 on the same machine and binaries as `nemotron3-nano-apple-silicon-bench.md` (2026-09-04). Structured copy: `hybrid-apple-silicon-bench-2.json`. Raw llama-bench / mlx_lm outputs, prompt and scripts: `raw-2026-09-05/`.

## Hardware and software

- Apple M4 Max, 128 GiB unified memory, macOS 27.0 (26A5416b). Mains power. No other GPU workload during measurements (sibling sessions confirmed idle; downloads only).
- llama.cpp: Homebrew build **8680**, commit **15f786e65**, backends `BLAS,MTL`, 12 threads (unchanged from 09-04).
- MLX: `mlx` **0.32.2**, `mlx-lm` **0.31.3** (PyPI latest on 2026-09-05), Python 3.12.13 in a fresh venv.
- Protocol: identical to 09-04. llama.cpp = `llama-bench -m <gguf> -p 512 -n 256 -ngl 99 -fa 1 -r 3` (mean ± std), a second identical run under `/usr/bin/time -l` for max RSS, and `llama-completion --temp 0 -n 160` for a coherence sample. MLX = `python -m mlx_lm generate --model <m> --prompt "$(cat prompt.txt)" --max-tokens 256 --temp 0`, three separate processes, median of the `Generation:` tok/s; the prompt is the same 101-word chat prompt as 09-04 (132-145 tokens after each model's chat template).

## Results

| Model | Stack (version) | Weights | Prompt tok/s | Decode tok/s | Peak memory | Notes |
|---|---|---|---:|---:|---:|---|
| Granite-4.0-H 350M | llama.cpp Metal (b8680) | Q4_K_M, ibm-granite rev a864f823, 208.94 MiB | pp512 **8496.05 ± 16.92** | tg256 **282.07 ± 1.03** | 0.35 GB max RSS | RSS run 8504.67 ± 11.13 / 278.79 ± 3.40; llama-completion eval 282.02 |
| Granite-4.0-H 350M | MLX mlx-lm 0.31.3 | 4-bit, mlx-community rev 2b96c1f6, 0.20 GB | 4231 (145-tok prompt) | **521.7** (513.6 / 521.7 / 530.3) | 0.57 GB (mlx peak) | stopped at 119 tokens (EOS) |
| Granite-4.0-H 1B | llama.cpp Metal (b8680) | Q4_K_M, ibm-granite rev c2cb1972, 856.00 MiB | pp512 **2659.35 ± 2.31** | tg256 **143.95 ± 1.97** | 1.07 GB max RSS | RSS run 2641.06 ± 5.44 / 146.92 ± 1.86; llama-completion eval 149.12 |
| Granite-4.0-H 1B | MLX mlx-lm 0.31.3 | 4-bit, mlx-community rev a5a21e23, 0.83 GB | 2086 (145-tok prompt) | **275.8** (280.0 / 275.8 / 273.9) | 1.46 GB (mlx peak) | stopped at 179 tokens (EOS) |
| Falcon-H1 1.5B-Instruct | llama.cpp Metal (b8680) | Q4_K_M, tiiuae rev 0d3a6cfe, 898.48 MiB | pp512 **2550.53 ± 4.11** | tg256 **148.58 ± 0.33** | 1.13 GB max RSS | RSS run 2554.44 ± 3.18 / 147.97 ± 0.17; llama-completion eval 148.23; coherent on the long prompt |
| Falcon-H1 1.5B-Instruct | MLX mlx-lm 0.31.3 | 4-bit, mlx-community rev 6f5e4f68 (uploaded 2025-09-26), 0.88 GB | 2018 (132-tok prompt) | **300.0** (301.5 / 300.0 / 299.8) | 1.43 GB (mlx peak) | **output degenerates** into repetition after two sentences on the long prompt (see below) |
| Falcon-H1 1.5B-Instruct | MLX mlx-lm 0.31.3 | 4-bit gs64 affine, **self-converted** from tiiuae bf16 rev 80ebc50d with mlx-lm 0.31.3 (4.505 bpw) | 2172 | **300.3** (300.6 / 300.3 / 300.0) | 1.43 GB | output **identical, word for word**, to the mlx-community 4-bit row (same repetition, same "Bubble Track Algorithm") |
| Falcon-H1 1.5B-Instruct | MLX mlx-lm 0.31.3 | 8-bit gs64, self-converted (8.503 bpw) | 1016 | **204.9** (one process) | 2.20 GB | fluent, not repetitive, semantically muddled |
| Falcon-H1 1.5B-Instruct | MLX mlx-lm 0.31.3 | bf16, tiiuae weights loaded directly | 1788 | **128.1** (one process, 219 tokens to EOS) | 3.63 GB | not repetitive, semantically muddled ("utilize a discrete GPU's VRAM…") |
| Falcon-H1 3B-Instruct | llama.cpp Metal (b8680) | Q4_K_M, **own conversion** from tiiuae bf16 rev 01087ec4 (`convert_hf_to_gguf.py` shipped with Homebrew llama.cpp 8680 -> f16 -> `llama-quantize Q4_K_M`, 4.79 BPW, 1.76 GiB) | pp512 **1300.65 ± 5.63** | tg256 **86.11 ± 0.40** | 2.15 GB max RSS | RSS run 1297.70 ± 5.30 / 86.45 ± 0.04; llama-completion eval 86.77; coherent. tiiuae's own Q4_K_M not downloaded (HF throttled) |
| Falcon-H1 3B-Instruct | MLX mlx-lm 0.31.3 | 4-bit gs64 affine, **self-converted** from tiiuae bf16 rev 01087ec4 with mlx-lm 0.31.3 (`mlx_lm convert -q --q-bits 4 --q-group-size 64`, 4.504 bpw, 1.7 GB) | 1169 (132-tok prompt) | **167.9** (167.9 / 168.2 / 165.9) | 2.44 GB (mlx peak) | fluent but semantically muddled ("CPUs and CPUs") |
| Falcon-H1 3B-Instruct | MLX mlx-lm 0.31.3 | 8-bit gs64, self-converted (8.503 bpw) | 1133 | **109.4** (one process) | 4.01 GB | coherent |
| Falcon-H1 3B-Instruct | MLX mlx-lm 0.31.3 | bf16, tiiuae weights loaded directly | 945 | **65.1** (one process) | 6.90 GB | coherent |
| Falcon-H1 7B-Instruct | llama.cpp Metal (b8680) | Q4_K_M, tiiuae rev 058c8c8f, 4.28 GiB | pp512 **686.96 ± 1.16** | tg256 **52.91 ± 0.40** | 4.89 GB max RSS | RSS run 683.65 ± 2.10 / 53.03 ± 0.93; llama-completion eval 54.53; coherent |
| Falcon-H1 7B-Instruct | MLX mlx-lm 0.31.3 | mlx-community 4-bit rev 3a0e8b0b | NOT MEASURED (download throttled) | | | |

Rows from 09-04 that belong in the same table: Nemotron-3 Nano 30B-A3B (86.19 / 159.7), Nemotron-3 Nano 4B (88.38 / 176.8), Granite-4.0-H Tiny 7B-A1B (117.31 / 202.2). Apple Core AI rows from the HF cards (measured earlier by the user, not today): Nemotron-3 Nano 4B int8-head M4 Max 85.2, iPhone 17 Pro 16.0; Granite-4.0-H 1B int8lin M4 Max 136.5, iPhone 17 Pro 30.2-31.3 (int8hu 35.4-37.1); Granite-4.0-H 350M fp16 M4 Max 191.1.

## How to read the columns

- Same asymmetries as 09-04: llama.cpp prompt = pp512 batch, MLX prompt = a 132-145 token chat prompt; peak memory = max RSS (llama.cpp, includes the mmapped file) vs Metal peak allocation (MLX); Q4_K_M is not the same bytes/token as 4-bit affine (Q4_K_M ≈ 4.8-5 bpw with k-quant scales; mlx 4-bit gs64 = 4.5 bpw). Core AI rows are int8 or fp16 exports, not 4-bit.
- MLX decode is 1.7-2.0x llama.cpp on every row where both ran (350M 1.85x, 1B 1.92x, Falcon 1.5B 2.02x, Falcon 3B 1.95x, Granite Tiny 1.72x on 09-04, Nemotron 4B 2.00x).

## Falcon-H1 in mlx-lm 0.31.3: what was observed (one prompt, greedy, not a quality evaluation)

- `mlx-community/Falcon-H1-1.5B-Instruct-4bit` (re-uploaded 2025-09-26, after ml-explore/mlx-lm#504): fluent for two sentences, then repetitive nonsense ("Track shared memory on Apple Silicon performance due to hybrid memory limits being suboptical…"); on the #504 prompt ("Implement bubble sort from scratch") it writes "Bubble Track Algorithm" mid-answer. llama.cpp with tiiuae's Q4_K_M answers the same long prompt coherently.
- Control for the 1.5B: converting tiiuae's bf16 myself with mlx-lm 0.31.3 (`-q --q-bits 4 --q-group-size 64`) gives the **same text, word for word**, on both prompts. So the mlx-community 1.5B checkpoint is a current conversion, not a stale one; whatever produces the repetition is in the 4-bit gs64 affine path. The 1.5B at 8-bit (204.9 tok/s) and bf16 (128.1 tok/s) does not repeat, but both read muddled on this prompt, whereas llama.cpp's Q4_K_M of the same weights reads cleanly. One prompt, greedy; I did not investigate that bf16-vs-llama.cpp gap (template, numerics or tie-breaking could all be involved) and do not claim a cause.
- Self-converted 3B with the current mlx-lm: bf16 and 8-bit are coherent; 4-bit gs64 affine is fluent but confused. So on this sample the 4-bit degradation tracks the 4-bit affine quantization (bf16 through the same code path is fine at 3B). Not tested: 4-bit with group size 32, `--q-mode mxfp4`, DWQ; a proper perplexity or task score.
- `mlx-community/Falcon-H1-3B-Instruct-4bit` card still says "converted using mlx-lm version 0.25.2" (2025-06-16, before the implementation was completed per #504); download pending, so its output is unverified today. T08 Sonnet's report ("incoherent garbage at ~172 tok/s") stands as that agent's observation.

## Commands

```sh
hf download ibm-granite/granite-4.0-h-350m-GGUF granite-4.0-h-350m-Q4_K_M.gguf
hf download ibm-granite/granite-4.0-h-1b-GGUF granite-4.0-h-1b-Q4_K_M.gguf
hf download tiiuae/Falcon-H1-1.5B-Instruct-GGUF Falcon-H1-1.5B-Instruct-Q4_K_M.gguf
hf download tiiuae/Falcon-H1-7B-Instruct-GGUF Falcon-H1-7B-Instruct-Q4_K_M.gguf
hf download mlx-community/granite-4.0-h-350m-4bit; hf download mlx-community/granite-4.0-h-1b-4bit; hf download mlx-community/Falcon-H1-1.5B-Instruct-4bit
hf download tiiuae/Falcon-H1-3B-Instruct --exclude "*.gguf"; hf download tiiuae/Falcon-H1-1.5B-Instruct --exclude "*.gguf"
python -m mlx_lm convert --hf-path tiiuae/Falcon-H1-1.5B-Instruct --mlx-path models/Falcon-H1-1.5B-Instruct-4bit-mlxlm0.31.3 -q --q-bits 4 --q-group-size 64   # and --q-bits 8
/opt/homebrew/opt/llama.cpp/bin/convert_hf_to_gguf.py <snapshot dir> --outtype f16 --outfile Falcon-H1-3B-Instruct-f16.gguf   # needs `pip install gguf`
llama-quantize Falcon-H1-3B-Instruct-f16.gguf Falcon-H1-3B-Instruct-Q4_K_M.gguf Q4_K_M
python -m mlx_lm convert --hf-path tiiuae/Falcon-H1-3B-Instruct --mlx-path models/Falcon-H1-3B-Instruct-4bit-mlxlm0.31.3 -q --q-bits 4 --q-group-size 64
llama-bench -m <gguf> -p 512 -n 256 -ngl 99 -fa 1 -r 3
/usr/bin/time -l llama-bench -m <gguf> -p 512 -n 256 -ngl 99 -fa 1 -r 3
llama-completion -m <gguf> -ngl 99 -fa on --temp 0 -n 160 -st --simple-io -p "In two or three sentences, what is unified memory on Apple Silicon and why does it matter for running LLMs locally?"
python -m mlx_lm generate --model <repo-or-dir> --prompt "$(cat prompt.txt)" --max-tokens 256 --temp 0   # x3
```

Raw outputs: `raw-2026-09-05/` (llama-bench JSON + text, RSS runs, completions, all mlx_lm outputs).

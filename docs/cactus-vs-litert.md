# Cactus vs LiteRT-LM vs MLX — Gemma-4-E2B on Apple silicon

Cactus (YC S25) [published](https://github.com/cactus-compute/cactus) an Apple-device
performance table for Gemma-4-E2B-CQ4 and a one-line reproduction recipe
(`cactus benchmark --ios`). This is an independent reproduction on the exact device they
list — an iPhone 17 Pro — plus a like-for-like comparison against LiteRT-LM and MLX on the
same phone, the same model, the same prompt length, and the same decode budget.

Everything below was produced by the scripts in this repo against Cactus commit `main` and
their published bundle `Cactus-Compute/gemma-4-E2B-it` @ `v2.0.1`. Raw per-run JSON is in
[`results/raw/2026-07-10-cactus-parity/`](../results/raw/2026-07-10-cactus-parity/).

## The short version

1. **Their published numbers reproduce.** iPhone 17 Pro, 704.8 tok/s prefill / 36.3 tok/s
   decode / 651 MB peak, against a published 729 / 37 / 644 — within 3%.
2. **Their prefill metric is honest.** `prefill_tps` does subtract a KV-cache handoff term
   from its denominator, but that term measures 0.01–0.03 ms. The headline number equals
   the wall-clock `promptTokens / TTFT` to within 0.01%.
3. **LiteRT-LM prefills ~5× faster and decodes ~1.6× faster** on the same phone and model.
4. **The quality gap is larger than the speed gap.** On GSM8K with a required answer format,
   Cactus's CQ4 build of Gemma-4-E2B never once produced the requested `#### <number>` line
   across 100 questions, scoring 3%. LiteRT-LM's int4-QAT build produced it 98 times and scored
   88%, against a 92% unquantized anchor.

The speed comparison is therefore not the interesting result. A 4-bit format that loses
multi-step reasoning is cheaper to run for the same reason it is cheaper to be wrong.

Three side findings, all in this repo's own harness, were fixed or flagged on the way and are
worth knowing about independently: `mlx-swift-lm` cannot load the *current* Gemma-4-E2B
conversion (re-uploaded 2026-07-06 with KV sharing), the llama.cpp adapter never cleared its
KV cache between runs and ignored the task's sampling parameters, and
`totalGenerationTimeSeconds` counted a 200 ms teardown sleep as generation time.

## Method

Cactus's LLM benchmark (`cactus-engine/tests/test_benchmark.cpp`) builds a prompt, truncates
it to exactly **1000 tokens**, decodes exactly **100 tokens**, and reports the mean of
**three runs after one warmup**, resetting the KV cache before each. `CactusParityTask` in
this repo mirrors that: a 1022-token prompt (17 lorem blocks — the same 8×128 prefill-chunk
bucket as their 1000, so neither side pays for an extra chunk), a 100-token budget, greedy,
driven with `--runs 4` so run 1 is cold and runs 2–4 are warm.

Metrics are defined so both sides are measured the same way:

- **prefill tok/s** = `promptTokens / TTFT`. Every engine here can produce it and it is what
  a user waits through. Cactus emits it as `ttft_prompt_tps`.
- **decode tok/s** = the engine's own steady-state rate, excluding TTFT. Cactus reports
  `(n-1)/(total-TTFT)`; this repo reports `n/decodeTime`. At n=100 the definitions differ by ~1%.
- **peak MB** = `phys_footprint`, the same `task_info(TASK_VM_INFO)` value both harnesses read.

`scripts/cactus_parity_report.py` rebuilds every table below from the raw artifacts.

### On `prefill_tps`

Reading `cactus-engine/src/complete.cpp:1503` suggests the headline prefill rate is inflated:

```cpp
cache_prime_compute_ms = max(0.0, cache_prime_ms - cache_state_copy_ms);
prefill_tps            = cache_prime_tokens * 1000.0 / cache_prime_compute_ms;
```

`cache_state_copy_ms` times `move_cache_states()`, the KV handoff between Cactus's separate
prefill and decode graphs — a real cost of that design, excluded from the published rate.
**Measured, it is inert.** The copy takes 0.01–0.03 ms out of a 365 ms (Mac) or 1420 ms
(iPhone) prefill, so `prefill_tps` and the honest `ttft_prompt_tps` agree to <0.01%. The
report script asserts this on every run rather than taking it on trust.

| run | headline `prefill_tps` | `ttft_prompt_tps` | `cache_state_copy_ms` | skew |
|---|---:|---:|---:|---:|
| iPhone 17 Pro, auto | 704.8 | 704.8 | 0.02 | 0.004% |
| iPhone 17 Pro, metal | 710.8 | 710.8 | 0.01 | 0.001% |
| iPhone 17 Pro, cpu | 220.8 | 220.8 | 0.02 | 0.000% |
| M4 Max, auto | 2738.4 | 2738.2 | 0.01 | 0.008% |

## Speed — iPhone 17 Pro (A19 Pro), Gemma-4-E2B, warm mean of 3

| engine | quant | prefill tok/s | decode tok/s | TTFT ms | peak MB |
|---|---|---:|---:|---:|---:|
| **LiteRT-LM** (own `benchmark()` API, 1000 tok) | int4 QAT | **3893.7** | 56.9 | 275.8 | — |
| **LiteRT-LM** (this repo's task, 1022 tok) | int4 QAT | **3190.5** | **57.3** | 320.3 | 688 |
| llama.cpp | Q4_K_M | 1989.9 † | n/a ‡ | — | 278 |
| Cactus (`--backend metal`) | CQ4 | 710.8 | 36.4 | 1408.7 | 648.8 |
| Cactus (`--backend auto`) | CQ4 | 704.8 | 36.3 | 1420.1 | 651.1 |
| Cactus (`--backend cpu`) | CQ4 | 220.8 | 15.0 | 4538.9 | 552.2 |
| CoreML-LLM | INT4 palettized | ≈28 § | 26.4 | 36472 | 1704 |
| MLX-Swift | Q4 | — ¶ | — ¶ | — | — |
| _Cactus, as published_ | CQ4 | _729_ | _37_ | — | _644_ |

† llama.cpp's own prefill window; it decodes 0 tokens on this prompt, so there is no TTFT.
‡ llama.cpp greedily samples an EOG token immediately on any prompt longer than short-chat's
12 tokens (a 543-token prompt fails identically, while short-chat generates 128 tokens at
37.5 tok/s). Suspected chat-template / special-token mismatch against the unsloth Gemma-4 GGUF.
§ CoreML reports no prompt token count; `1022 / 36.5 s` from the observed TTFT.
¶ `mlx-swift-lm` cannot load the **current** `mlx-community/gemma-4-e2b-it-4bit`. That repo was
re-uploaded on 2026-07-06 ("Re-upload MLX conversion from google/gemma-4-E2B-it@70af34e2"): the
prior revision (`2c3e507`, 2026-05-19) materialised `k_proj`/`v_proj` on all 35 layers (2649
tensors), the current one carries them only on layers 0–15 and shares KV from layer 16 on (2511
tensors). `mlx-swift-lm`'s Gemma4 attention builds `k_proj`/`v_proj` on every layer and does not
skip the KV-shared ones, so it fails on the new layout with a missing-key error at layer 15.
Reproduced on the app's resolved `main` (`b95dc78`), the revision pinned in `Package.resolved`
(`5b7e543`), and the macOS CLI. This repo's earlier iPhone MLX number (decode 46.2 tok/s,
2026-06-17) was measured on the prior revision. Python `mlx-lm` loads the current repo fine, and
is the arm scored in the quality gate below.

Two independent LiteRT measurements agree: its own `benchmark(prefillTokens:decodeTokens:)`
entry point (the direct analogue of `cactus_benchmark_tokens`) and this repo's app path with
a real 1022-token prompt. Their TTFTs are consistent with their prefill rates
(`1000/4171 = 240 ms` + one decode step ≈ the observed 257 ms).

CoreML-LLM is the outlier in the other direction: a 1K-token prompt costs it **30–41 seconds**
of TTFT (and the cost grows run over run as the phone heats), against LiteRT's 0.32 s and
Cactus's 1.4 s.

`auto` ≈ `metal`, and `cactus_backend_select("auto")` in fact returns `-1` and falls through
to `cactus_default_backend()`, which picks Metal when it is available. So the published row is
a **Metal GPU** number. Neither engine touches the ANE.

### The protocol does not reach steady state on a phone

Cactus's three warm runs take about 40 seconds, and prefill decays monotonically across them:

| | warmup | run 1 | run 2 | run 3 |
|---|---:|---:|---:|---:|
| prefill tok/s | 764.3 | 733.8 | 698.1 | 682.7 |
| TTFT ms | 1308.5 | 1362.9 | 1432.6 | 1464.8 |

The published 729 tok/s is essentially the first measured run. On the M4 Max the same
protocol shows no decay (2719.7 → 2745.0 → 2750.4). A phone number taken over 40 seconds is a
burst number, not a sustained one.

### Peak memory is `phys_footprint`, so mmap'd weights are invisible

Both harnesses read `task_info(TASK_VM_INFO).phys_footprint` — the value jetsam uses. Clean,
file-backed pages are not counted, so a 3.8 GB extracted bundle reports 651 MB on iPhone and
1433 MB on the Mac. Correct as a jetsam headroom metric; not a statement about model size.

## Quality — GSM8K, 0-shot CoT, greedy, identical prompt and extraction

Speed across engines only means something at equal output quality, and each engine ships its
own 4-bit format: Cactus CQ4 is Hadamard rotation + per-group codebook PTQ, LiteRT-LM ships
int4 QAT, MLX uses affine 4-bit group quant. Same 100 questions, same prompt (which demands a
final `#### <number>` line), same answer extraction, `max_tokens=1024`.

| build | quant | GSM8K | never emitted `#### <n>` | mean output chars | n |
|---|---|---:|---:|---:|---:|
| _bf16 (unquantized anchor, CPU)_ | _none_ | _92.0%_ | _1 / 100_ | _1290_ | _100_ |
| LiteRT-LM `.litertlm` | int4 QAT | **88.0%** | 2 / 100 | 1287 | 100 |
| MLX 4-bit | affine PTQ | 78.0% | 25 / 100 | 2527 | 100 |
| **Cactus CQ4** | rotation + codebook PTQ | **3.0%** | **100 / 100** | 4241 | 100 |

The anchor puts the ceiling at 92%. LiteRT's int4-QAT lands 4 points under it; generic 4-bit PTQ
(MLX) gives up 14; CQ4 does not survive at all. That ordering — **QAT near parity, generic PTQ
degraded** — is the same result this repo found on Falcon3 and Qwen3, now with an unquantized
reference on Gemma-4.

MLX's 78% is a floor: 25 of its answers were still running when the budget ran out. LiteRT's 88%
is near-final. Cactus's 3% is what the extraction's last-number fallback scores when the model
never reaches the answer line; it is not a reasoning score.

Three earlier passes were discarded, and each was caught by one cheap control:

- At `max_tokens=640` the three 4-bit builds scored 2% / 76% / 33%. Truncation and wrong answers
  are indistinguishable at that budget, which is why the table reports the marker rate next to
  accuracy.
- The bf16 anchor first ran on MPS, where it degenerated into token repetition (`**\n\n**\n\n**`,
  `}\n}\n}`) on 18 of 50 questions and scored 64%. That run measures a Metal numerics bug, not the
  model; it is quarantined under `results/raw/2026-07-10-cactus-parity/rejected/`.
- The anchor was then run at n=25 while every other arm was at n=100 — different question sets, and
  it made LiteRT's 88% look like it beat an 84% ceiling. On the same 25 questions the anchor was
  84% and LiteRT 80%. Re-run at n=100 the anchor is 92%, and the ordering is what a quantized model
  should produce.

### This is the model, not the measurement

`scripts/cactus_cq4_ablation.py` kills the four alternative explanations, in order:

- **Chat template.** `cactus_render_prompt` returns `<bos><|turn>user\nHi<turn|>\n<|turn>model\n`
  — byte-identical to the HuggingFace reference template for Gemma-4.
- **Stop sequences.** Gemma-4 ends turns with `<turn|>`, not `<end_of_turn>`. Running with
  `<end_of_turn>`, with `<turn|>`, and with none at all produces the same 3851 characters.
- **Thinking mode.** Cactus's own benchmark prefixes its system prompt with `/no_think`.
  Setting `enable_thinking_if_supported=False`, adding that system prompt, or both, changes
  nothing.
- **Token budget.** At 1536 tokens, 10 of 10 questions still never reach the marker, averaging
  6656 characters. It is not short of room.

With those gone, the failure localises. CQ4 answers `What is 17 + 25?` correctly in 40 tokens
and terminates. It answers a one-step word problem (`A box has 48 apples. Half are red.`)
correctly, though it hedges — *"if 'half' means exactly half"*. Given the two-step problem it
never commits to a reading at all:

> Natalia sold clips to 48 friends in April. This implies that she sold $N_{April} = 48$ clips
> (assuming the problem implies a direct relationship, or we assume the number of friends is
> the quantity). Let's assume Natalia sold $X$ clips in April. … Let's re-evaluate the structure.

and continues until the budget runs out. LiteRT-LM's int4-QAT build answers both prompts
directly. Fluency and one-step arithmetic survive CQ4; committing to a reading of a multi-step
word problem does not.

### It is not an artefact of the `####` marker either

CQ4 half-fails to emit the marker: told to copy `#### 42` verbatim it does, but told to count to
30 and finish with the marker it writes `# 30` after 85 tokens, and asked for four hash characters
it writes `abcd 42`. That raises the obvious objection — maybe it reasons fine and merely cannot
type `####`. Re-scored with a marker it can produce (`The answer is <number>.`) on the same 50
questions, CQ4 gets **2.0%** where LiteRT gets **82.0%** (under `#### <n>`, the same 50 give 4.0%
and 86.0%). CQ4 still averages 4237 characters before hitting the budget. The marker defect is
real and independent; it is not the cause of the score.

### Their hybrid does not rescue it

Cloud handoff was disabled to measure the on-device model, which could be called unfair since
handoff is Cactus's shipped default. It is not. On GSM8K questions CQ4 answers wrongly, Cactus's
own confidence probe returns **0.9399–0.9999** against its own 0.81 threshold and reports
`cloud_handoff=false, "above threshold"` (6/6 sampled). The shipped default answers locally,
confidently, and wrong.

This contradicts Cactus's claim that CQ "enables significant model compression while
maintaining quality across benchmarks like ARC, MMLU, and GSM8K". It is measured against the
calibrated bundle their own `cactus benchmark` downloads, on their own `cactus_complete` FFI,
with greedy sampling and cloud handoff disabled.

### Where each number was measured

Every GSM8K number is from the Mac; only the speed table is from the iPhone. For Cactus that gap
is closed by measurement: with the same 1000-token prompt and greedy decode, its Metal path emits
**token-for-token identical output on the iPhone 17 Pro and the M4 Max** (100/100 completion ids;
its CPU path diverges at token 6). So the quality result transfers to the device that was
benchmarked. LiteRT's GSM8K ran on Mac CPU; its iPhone Metal output on the parity prompt is
coherent, but GSM8K was not re-run on-device — worth stating, because this repo has a documented
case (gemma-4-12B) of LiteRT producing degenerate output on the Apple GPU while the CPU path was
fine. The MLX quality arm is Python `mlx-lm`, a path that does not run on the phone at all.

## Reproducing

```bash
# Cactus, on the phone. Their tests/ios/run.sh needs two patches to launch on iOS 26+
# (a UIScene delegate, and IPHONEOS_DEPLOYMENT_TARGET ≥ 15.0) — see the notes below.
cactus benchmark --ios

# This repo, matched protocol
scripts/bench_cactus_parity_iphone.sh all
python3 scripts/cactus_parity_report.py

# Quality gate
python3 scripts/gsm8k_cactus_vs_litert.py --backend litertlm --path <…>.litertlm --n 100 --max-tokens 1024
python3 scripts/gsm8k_cactus_vs_litert.py --backend cactus --cactus-repo <clone> --path <clone>/weights/gemma-4-e2b-it-cq4 --n 100 --max-tokens 1024
python3 scripts/cactus_cq4_ablation.py --cactus-repo <clone> --path <clone>/weights/gemma-4-e2b-it-cq4
```

Two things block `cactus benchmark --ios` on a current toolchain, neither related to the
engine. Its test app is `UIApplicationDelegate`-only, so iOS 26+ traps it at launch in
`__UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption` (`EXC_BREAKPOINT`); it needs a
scene manifest *and* a real `UIWindowSceneDelegate`. And `tests/ios/run.sh` sets
`IPHONEOS_DEPLOYMENT_TARGET=13.0`, below Xcode 27's 15.0 floor.

## Caveats

- Cactus's Mac row is an M4 **Pro**; the Mac here is an M4 **Max** (2738 tok/s prefill /
  141.6 decode / 1433 MB). Those are not comparable, which is why no Mac head-to-head is drawn.
- ~~macOS LiteRT-LM runs CPU-only (its GPU path is OpenCL, dead on Apple silicon)~~ **Retracted
  2026-07-13:** this was a misreading of a benign log line. The `Failed to create OpenCL context`
  INFO line appears on the *working* GPU path too, immediately followed by `Created Metal device
  from provided device id` — macOS LiteRT-LM runs on the GPU via WebGPU/Dawn→Metal, and warm it
  reproduces the official E2B card numbers within ~5% (decode ~152 vs 160, prefill ~7.5k vs 7,835;
  logs in [`results/raw/2026-07-13-e2b-mac-webgpu/`](../results/raw/2026-07-13-e2b-mac-webgpu/)).
  The iPhone 17 Pro remains the row Cactus published, which is why it is the head-to-head here.
- LiteRT stops at EOS after ~32 tokens on this prompt where Cactus is forced to 100. Decode
  rate is near-independent of count at this KV depth (1022 → 1055 vs 1122), but the windows are
  not identical.
- GSM8K was scored on the first 100 test questions, not the full 1319.

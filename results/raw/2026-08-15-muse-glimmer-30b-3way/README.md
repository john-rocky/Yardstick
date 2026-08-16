# Muse-Glimmer-30B text decoder — Core AI vs MLX vs ExecuTorch, M4 Max (2026-08-15)

Decode tok/s, greedy, batch 1, 192 new tokens, **interleaved per prompt** with 45 s cooldown
between every run. Environment and per-arm provenance: [`ENV.md`](ENV.md). Raw log:
[`threeway-interleaved-192.log`](threeway-interleaved-192.log).

| arm | artifact | quant | weights | p1 r1 | p1 r2 | p2 r1 | p2 r2 | mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Core AI** (fork engine, reference — see ENV.md) | `mlboydaisuke/Muse-Glimmer-30B-CoreAI` | int4hu | 16.35 GB | 27.5 | 27.1 | 27.5 | 27.6 | **27.43** |
| **MLX** (mlx-vlm 0.6.13) | `mlx-community/Muse-Glimmer-30B-4bit` | 4-bit | 18 GB | 27.18 | 27.38 | 27.50 | 27.37 | **27.36** |
| **ExecuTorch** (Metal, Meta's `solo_runner`) | `meta-models/…-ExecuTorch-PTE` `k-quant-17G…text-solo-metal` | k-quant | 17.9 GB | 24.0 | 23.8 | 24.1 | 24.1 | **24.00** |

**Spread check:** every arm agrees within ~1.8 % across its four rounds — passes.

**Reading:** Core AI and raw MLX are indistinguishable (+0.3 %). Both beat Meta's shipped
on-device artifact by ~14 %. Since ExecuTorch's `metal` backend is MLX-native (their README),
the finding is not "engine X is fast" — it is that **Meta's shipped artifact leaves ~14 % on
the table against the runtime it is built on**.

## Two disclosures that must travel with the table

1. **The Core AI arm is not this repo's pinned 0.2.0 arm.** The runner is a public fork
   build (+ a 2-file session diff, archived in `patches/`); released 0.2.0 neither runs on
   this OS seed nor supports this model. Label class: "fork engine (reference)", same as the
   Gemma-4 PLE rows. Details in `ENV.md`.
2. **The Core AI artifact is self-made** (exported by this repo's author); the MLX and
   ExecuTorch arms consume community/vendor artifacts. The bundle is published and gated
   token-exact against its fp16 oracle, but a third party cannot yet rebuild the export.
   Details in `ENV.md`.

## The failed first attempt (kept per fairness rule 4)

[`headtohead-blockordered-256-failed.log`](headtohead-blockordered-256-failed.log) — the
block-ordered first attempt (all ExecuTorch prompts, then all Core AI prompts, 256 tokens):
ExecuTorch decayed **23.5 → 17.4 tok/s inside its own block**, and Core AI, running second
on the GPU the first block had heated, read **15.6–20.7** against its own true ~27.4. Both
arms wrong, in opposite directions. This capture is the worked example behind
[fairness rule 11](../../../methodology/fairness-rules.md) (interleave arms).

[`headtohead2-interleaved-192.log`](headtohead2-interleaved-192.log) — the interleaved
two-way (CA vs ET) that preceded the three-way. Two artifacts visible there, both kept:
the very first Core AI run read 16.2 (cold page cache — ExecuTorch had just pulled 17.9 GB
through it; every later round reads 27.3–27.7), and ExecuTorch emitted a single token on
prompt 3 despite `--ignore_eos=true` (runner behaviour; not counted as a win for anyone).

## Not carried

- **Meta's DFlash figure (37.8 tok/s)**: their published average over an unpublished prompt
  set on their machine — not on this footing, deliberately excluded from every table here.
- **Quality**: no quality arm was run; Meta's published 1.0 %-degradation figure for the
  17G quant has no matched counterpart here and is not cited as one.
- This result is **not** part of the `pb-random-v1` protocol tables — different protocol,
  different meaning; do not merge.

## Reproduce

```bash
./reproduce mac muse-glimmer-30b-3way          # prereq check + pinned command
```

## Quality — GSM8K, same 100 questions (added 2026-08-17)

Greedy, scored by [`scripts/parity_gsm8k.py`](../../../scripts/parity_gsm8k.py) — **that file
was not modified.** The harness here imports its question set, CoT suffix, extractor and
scoring, and adds the two arms it lacks: ExecuTorch through Meta's own `solo_runner`, and MLX
through `mlx_vlm` (`mlx_lm` raises `Model type muse_glimmer not supported`).
Harness: [`gsm8k-3way-harness.py`](gsm8k-3way-harness.py) ·
raw: [`gsm8k-100q-twopass.log`](gsm8k-100q-twopass.log) ·
per-question: [`gsm8k-100q-twopass.json`](gsm8k-100q-twopass.json).

| arm | weights | GSM8K/100 | rescued | still capped | median gen tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Core AI** `int4hu` | 16.35 GB | **98** | 32 | 3 | 532 |
| **ExecuTorch** `k-quant-17G` | 17.9 GB | **97** | 14 | 1 | 326 |
| **MLX** 4-bit | 18 GB | **95** | 26 | 2 | 453 |

**Reading:** three questions apart is not resolvable at n=100. This says "no arm is
meaningfully worse", not "Core AI wins". Its value is that the speed table above compares
three different weights and Core AI's is the smallest — the obvious objection is that some of
the speed is just fewer bytes, and this is the measurement that answers it.

### Two-pass budget — a fairness rule this result depends on

`llm-runner`'s wall time on this bundle is `5 s + 0.037 × max_tokens`, **independent of tokens
actually generated**: it steps to the budget after the stop token halts output. ExecuTorch
does not do this. So a single budget wide enough for the longest answer taxes every question
on one arm only.

Pass 1 ran at 700, pass 2 re-ran only the questions that hit it, at 2048. **This changes the
result, not just the runtime:** Core AI scores **87** at a flat 700 and **98** after the
un-truncated re-run. A truncated answer is not blank — `extract()` falls back to "the last
number in the text", so it scores a number lifted from mid-reasoning, usually wrong and
occasionally right by accident.

Truncation rates differ per arm (32 / 26 / 14), so one fixed budget penalises whichever arm
reasons longer, and the quality column silently becomes a verbosity column.

**Residue:** 3 / 1 / 2 questions (Core AI / ExecuTorch / MLX) still hit 2048 and are scored
from truncated output. Core AI and MLX also emit visibly longer answers than ExecuTorch for
the same questions — unexplained, and not visible in the score.

**Not measured:** speculative decoding on any arm. Core AI has an n-gram result (36.7 tok/s,
no drafter) but mlx-vlm ships `--draft-kind {dflash,eagle3,mtp}` and ExecuTorch ships DFlash,
so a speculative row needs all three or none.

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

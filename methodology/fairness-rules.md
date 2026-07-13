# Fairness rules

The benchmark is only useful if the comparisons are honest. The full rules:

## 1. Same prompt, same token budget

Every runtime sees the same prompt text and the same `maxTokens` limit. If a runtime needs special chat-template wrapping (e.g., `<|im_start|>` tokens for ChatML), that wrapping is applied uniformly per model, not per runtime.

## 2. Cold and warm runs are reported separately

Three distinct regimes — a table must say which one each number is (tightened 2026-07-13 after
the E2B card reconciliation, where conflating them produced a 2.6× prefill "discrepancy" that
was purely protocol):

- **First-ever** — first launch after install: compilation/weight caches (ML Drift cache,
  Core AI Metal pipeline cache) are built during the run. One-time cost; report separately
  where relevant, never as the engine's speed.
- **Cold** — fresh process launch, caches already on disk, first generation. This is what this
  repo's historical "median of 3 cold launches" tables measure. It is a *first-use latency*
  metric, *not* comparable with vendor model cards.
- **Warm** — in-process steady state: run generation N≥2 in the same process (`--runs 4`,
  discard run 1, report the **median of runs 2–4**). This is the vendor-card convention
  (HF litert-community cards, Core AI marketing numbers) and the headline for cross-vendor
  comparison.

Warm-up gains are **engine-dependent** (measured: Core AI ~2.5×, LiteRT E2B ~1.1×, MLX ≈ flat),
so cold rankings do not imply warm rankings — a table that compares engines must hold the regime
fixed, and a table meant to be checked against official cards must be warm.

Thermal guard for warm campaigns: ≥100 s cooldown between cells; verify
`initialThermalState == nominal` in the per-run JSONL post-hoc, re-run flagged cells.

Both numbers matter. Cold matters for app launch latency. Warm matters for the user's second message in a chat — and for any comparison against published vendor numbers.

## 3. Quantization is explicit

Every result row shows model size, quantization, runtime format, and backend. We never compare a 4-bit GGUF and an FP16 CoreML model as if they were the same model running on different runtimes. They are different deployment profiles.

## 4. Failed runs stay in the table

If a runtime crashes, OOMs, hangs, or cannot support a configuration, the row stays in the table with a clear failure reason. Hiding failures makes a benchmark useless.

| Runtime | Model | Device | Result | Reason |
|---|---|---|---|---|
| _example_ | Llama-3-8B Q4 | iPhone 15 Pro | Failed | OOM during prefill at 4K context |

## 5. Don't hide integration difficulty

A runtime that hits 50 tok/s but requires writing 800 lines of glue code, has no streaming API, and cannot be cancelled is described that way. Integration difficulty is a separate dimension and is not folded into the speed score.

## 6. Same device class

Cross-device numbers (iPhone 15 Pro vs iPhone 17 Pro) are valuable but always shown in distinct rows. We never average across device classes.

## 7. Same build configuration where possible

Debug and Release builds give different numbers. Default to Release. If a number is from a Debug build (e.g., during integration), it is flagged.

## 8. No cherry-picking

For runs where the variance matters (sustained-generation tasks, lifecycle tasks), publish the median of N>=3 runs and note the spread. Don't publish "best" numbers.

## 9. Disclose hardware state

- charging state at start of run
- low-power-mode state at start of run
- thermal state at start of run

Any of these can change measured throughput by 30%+. Hiding them lets a runtime look better than it is.

## 10. Prefer the official runtime SDK

When multiple integration paths exist (e.g., the runtime's own Swift package vs. a community wrapper), prefer the official one and note the version. Wrapper-induced overhead is a real concern but should be documented separately, not silently included.

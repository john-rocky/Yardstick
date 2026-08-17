# Fairness rules

The benchmark is only useful if the comparisons are honest. The full rules:

> **Rule citations in code use slugs, not numbers.** This file and CLAUDE.md's
> "Rules that produce wrong numbers when broken" both number their rules, and the
> two numberings collide (this file's §3 is quantization; CLAUDE.md's #3 is
> budget/mode mixing — scripts used to cite "rule 3" ambiguously). Each heading
> below carries a `slug:` comment; CLAUDE.md's five working rules map to:
>
> | CLAUDE.md working rule | slug |
> |---|---|
> | 1. State the quantization for every arm | `quant-per-arm-rule` |
> | 2. "int4" is not a spec | `quant-label-rule` |
> | 3. Never mix modes or budgets across arms | `budget-mode-rule` |
> | 4. Decode trials must agree within a few percent | `spread-rule` |
> | 5. A number without a stored report is not a measurement | `stored-report-rule` |

## 1. Same prompt, same token budget  <!-- slug: same-budget -->

Every runtime sees the same prompt text and the same `maxTokens` limit. If a runtime needs special chat-template wrapping (e.g., `<|im_start|>` tokens for ChatML), that wrapping is applied uniformly per model, not per runtime.

## 2. Cold and warm runs are reported separately  <!-- slug: cold-warm-split -->

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

## 3. Quantization is explicit  <!-- slug: quant-explicit -->

Every result row shows model size, quantization, runtime format, and backend. We never compare a 4-bit GGUF and an FP16 CoreML model as if they were the same model running on different runtimes. They are different deployment profiles.

## 4. Failed runs stay in the table  <!-- slug: failed-runs-stay -->

If a runtime crashes, OOMs, hangs, or cannot support a configuration, the row stays in the table with a clear failure reason. Hiding failures makes a benchmark useless.

| Runtime | Model | Device | Result | Reason |
|---|---|---|---|---|
| _example_ | Llama-3-8B Q4 | iPhone 15 Pro | Failed | OOM during prefill at 4K context |

## 5. Don't hide integration difficulty  <!-- slug: disclose-difficulty -->

A runtime that hits 50 tok/s but requires writing 800 lines of glue code, has no streaming API, and cannot be cancelled is described that way. Integration difficulty is a separate dimension and is not folded into the speed score.

## 6. Same device class  <!-- slug: same-device-class -->

Cross-device numbers (iPhone 15 Pro vs iPhone 17 Pro) are valuable but always shown in distinct rows. We never average across device classes.

## 7. Same build configuration where possible  <!-- slug: same-build-config -->

Debug and Release builds give different numbers. Default to Release. If a number is from a Debug build (e.g., during integration), it is flagged.

## 8. No cherry-picking  <!-- slug: no-cherry-pick -->

For runs where the variance matters (sustained-generation tasks, lifecycle tasks), publish the median of N>=3 runs and note the spread. Don't publish "best" numbers.

## 9. Disclose hardware state  <!-- slug: disclose-hw-state -->

- charging state at start of run
- low-power-mode state at start of run
- thermal state at start of run

Any of these can change measured throughput by 30%+. Hiding them lets a runtime look better than it is.

## 10. Prefer the official runtime SDK  <!-- slug: official-sdk -->

When multiple integration paths exist (e.g., the runtime's own Swift package vs. a community wrapper), prefer the official one and note the version. Wrapper-induced overhead is a real concern but should be documented separately, not silently included.

## 11. Interleave arms; never run one arm's block, then the next  <!-- slug: interleave-arms -->

Run order is a hardware-state variable, and rule 9 (disclose hardware state) is not enough to
neutralize it — disclosing thermal state does not stop order from deciding the result. A
large model heats the GPU within seconds, and absolute decode tok/s on a Mac swings ~30% with
thermal state, so block ordering measures the order, not the engines.

Worked example (2026-08-15, Muse-Glimmer-30B on M4 Max —
`results/raw/2026-08-15-muse-glimmer-30b-3way/`): a block-ordered first attempt had
ExecuTorch decaying **23.5 → 17.4 tok/s inside its own block**, and Core AI — the block that
went second, on the GPU the first block had heated — read **15.6–20.7 against its own true
~27.4**. Both arms were wrong, in opposite directions, and the per-trial spread check catches
only the arm whose spread is wide, not the one that is uniformly depressed.

Therefore:

- **Interleave arms per prompt** (A/B/…/A/B), never block per arm.
- **Cool down between every run** (tens of seconds for a 30B-class model on a Mac; §2's
  ≥100 s guard for warm campaigns).
- **Publish every round**, so the reader can see the order effects that remain.
- First-position cache effects are real too: the first run after another arm has pulled a
  multi-GB model through the page cache can read far below true (measured: 16.2 vs 27.4).
  Discard such a round only on the strength of the following rounds, and say so in the table.

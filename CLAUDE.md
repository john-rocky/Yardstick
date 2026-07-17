# apple-silicon-llm-bench — working notes

**Launch your session in this directory.** Driving this repo from another root (e.g.
`~/code/coreai`) is how a session ends up running `scripts/bootstrap.sh` from the wrong cwd and
missing `methodology/next-session-brief.md` — both happened on 2026-07-17 and cost hours.

## What this repo is

A **neutral, reproducible** benchmark for local LLMs on Apple Silicon. Google's LiteRT team is
evaluating extending it into continuous, automated competitive benchmarking:
- engines: **LiteRT-LM, llama.cpp, Core AI, MLX, Cactus**
- platforms: Android, iOS, macOS

"Neutral" and "reproducible" are load-bearing, not marketing. See the rules below.

## Read these first

- `methodology/next-session-brief.md` — carries hard-won gotchas. **Read before building.**
- `methodology/fairness-rules.md` — what makes a cell publishable.
- `~/code/coreai/GEMMA4_LU_BENCH_HANDOFF.md` — current Gemma-4 / Lu campaign state (2026-07-17).

## Rules that produce wrong numbers when broken

1. **State the quantization for every arm, in the table.** Each arm runs at *its own best
   available* build; that is a fair and useful comparison — but only if the recipe is visible.
   Audited 2026-07-17: the published Gemma-4-E2B table crowned LiteRT 🏆 while **MLX ran PTQ and
   llama.cpp ran a third-party PTQ**, and LiteRT ran its best. The trophy was measuring who had
   the better checkpoint. Google ships an official QAT build for every one of these ecosystems —
   use it. PTQ→QAT alone is worth ~9 points of GSM8K on MLX (78 → 87).
2. **"int4" is not a spec.** Google publishes four QAT checkpoints for Gemma-4. LiteRT's
   `.litertlm` is the **wNa8o8 mobile schema** (2-bit decode layers, optimized KV cache, static
   int8 activations) — a co-designed weights+runtime package, not a bit width. It does **not
   transfer**: the same weights score 85% on LiteRT and 48% on any fp16-activation runtime
   (verified — two independent implementations return identical wrong answers). Never label it
   "INT4 (QAT)"; never rank it against uniform-int4 arms as if the delta were runtime speed.
   Details: `coreai-models-community/knowledge/gemma4-wna8o8-requires-int8-activations.md`.
3. **Never mix modes or budgets across arms.** `max_tokens` means a *generation budget* for
   MLX/Core AI but *total context* for `litert-mac-verify` — and undersizing that one **corrupts**
   output instead of truncating. Chat templates differ on thinking-mode defaults (HF renders
   thinking ON; swift-transformers OFF). Probe one item through every arm and compare prompt/output
   token counts before trusting a run. Checklist:
   `coreai-models-community/knowledge/cross-runtime-quality-benchmarking.md`.
4. **Decode trials must agree within a few percent.** GPU contention roughly halves decode and
   the only tell is per-trial spread (measured: 72.7/117.9/72.4 contended vs 171.3/171.6/171.0
   idle). Quote the spread; if it's wide, throw the number out.
5. **A number without a stored report is not a measurement.** `results/raw/*.jsonl` already
   records `hfRepoId` + `quantization` per run — keep it that way, and surface them in tables.

## Build

```bash
cd ios/BenchmarkApp && ./scripts/bootstrap.sh    # note the path: NOT ./scripts from repo root
```
```bash
xcodebuild -project ios/BenchmarkApp/BenchmarkApp.xcodeproj -scheme BenchmarkApp \
  -configuration Release -destination "generic/platform=iOS" \
  -skipPackagePluginValidation -skipMacroValidation -allowProvisioningUpdates \
  PRODUCT_BUNDLE_IDENTIFIER="com.daisukemajima.llmbench" \
  DEVELOPMENT_TEAM=MFN25KNUGJ CODE_SIGN_STYLE=Automatic build
```
- Every flag is required. `-skipPackagePluginValidation`: mlx-swift vendors a plugin whose trust
  prompt is GUI-only. `-allowProvisioningUpdates` + a **unique bundle id**: the canonical
  `com.iosllmbenchmark.benchmarkapp` is unregisterable ("not available"), and the Xcode GUI hits
  the same wall — don't burn a session on it.
- **Never** send xcodebuild to `/dev/null`: failures then print only `** BUILD FAILED **`.
- Entitlements `increased-memory-limit` + `extended-virtual-addressing` **cannot be dropped** —
  the first is what raises jetsam to ~6.44 GB, i.e. the reason multi-GB models fit at all. A
  wildcard provisioning profile cannot carry them.

## Arms

- **Core AI** is pinned to Apple's released **0.2.0**. Per-layer-embedding models (Gemma-4
  E2B/E4B) additionally need `EngineOptions.staticInputBuffers`, which is **not Apple API** —
  it's a local engine patch (absent from 0.1.0 and 0.2.0). Gated behind
  `SWIFT_ACTIVE_COMPILATION_CONDITIONS="$(inherited) COREAI_STATIC_INPUTS"`, **off by default**,
  so a clean clone builds and PLE models report `unsupported` rather than breaking the build.
  A patched engine lives at `~/code/coreai-models-020-bench` (0.2.0 + that patch only).
  **Do not point this repo at `~/code/coreai/coreai-models`** — that is a shared working checkout
  other sessions depend on.
- **CoreML/ANE is off by default** (2026-07-17): not in the engine list above; it is this repo
  author's own library, which sits badly in a tool presented as neutral; Core AI supersedes it;
  and it only ever built from an unpushed local checkout. `CoreMLRuntime.swift` is
  `canImport`-guarded, so it compiles out. Past results in `results/raw/` are unaffected.

## Reproducibility is the product

If a clean clone can't build it, it isn't a benchmark anyone can check — it's our private claim.
On 2026-07-17 this repo could not build from a clean clone at all: two arms referenced APIs that
exist only in unpushed local checkouts (`~/Downloads/CoreML-LLM`, a patched coreai-models). When
you add an arm, verify with a fresh clone or a wiped `Vendored/` before believing it works.

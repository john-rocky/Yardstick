# LiteRT-LM thinking axis — re-run runbook (waiting on the next tagged release)

**Trigger:** the first LiteRT-LM tagged release after v0.14.0 whose Swift API contains
`ThinkingConfig` (added to main Jul 9 2026, commit `ba9a470` — hours after the v0.14.0 cut,
so it is in no tagged release as of 2026-07-21).

**Verify the trigger before doing anything** (both must pass):

```bash
# 1. Swift surface at the tag
curl -sL https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/<TAG>/swift/Config.swift | grep -c ThinkingConfig   # > 0
# 2. C symbols in the released binary (the Swift wrapper is useless without them)
curl -sL -o /tmp/m.zip https://github.com/google-ai-edge/LiteRT-LM/releases/download/<TAG>/CLiteRTLM_mac.xcframework.zip
unzip -oq /tmp/m.zip -d /tmp/m && nm -gU /tmp/m/CLiteRTLM_mac.xcframework/macos-arm64_x86_64/libCLiteRTLM_mac.dylib | grep -c litert_lm_thinking   # > 0
```

## Steps

1. **Bump the fork** (`~/code/swift-litert-lm`, used by litert-mac-verify): update
   `liteRTLMVersion` + both binaryTarget checksums in `Package.swift`
   (`shasum -a 256` the two release zips), and re-vendor the official Swift wrapper
   (`Sources/LiteRTLM/` ← upstream `swift/` at the tag) so `ThinkingConfig` +
   `ConversationConfig(thinkingConfig:)` exist.
2. **litert-mac-verify**: add a `--thinking` flag that passes
   `ThinkingConfig(enableThinking: true, thinkingTokenBudget: -1)` into the
   `ConversationConfig` next to the existing greedy sampler
   (`SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0)` — keep it identical to the
   86.0-row protocol).
3. **Harness**: `scripts/parity_gsm8k.py` `run_litertlm()` currently has no `thinking`
   param — thread `--thinking` through to the verify binary like the other arms.
4. **Capture** (GPU, the same surface as the published 86.0 row):
   `--which int4 --greedy --max-tokens 2048 --n 100`, OFF and ON, tags
   `litert-gemma4-e2b-<tag>` / `...-thinking`. Bundle stays the June snapshot
   `9262660` (`litert-community/gemma-4-E2B-it-litert-lm`) — its template already has the
   `enable_thinking` branch; no model-side change is needed.
5. **Docs**: add the LiteRT thinking cell to the campaign table; replace the
   "LiteRT is structurally locked out of thinking" phrasing with an
   "as of v0.14.0 (released artifacts)" caveat wherever it survives
   (post-v2 / xpost drafts / Shuangfeng doc — the latter is meeting-session-managed,
   propose, don't edit).
6. Optional iPhone: `MediaPipeRuntime.swift` can take the same `ThinkingConfig` for an
   on-device thinking decode/depth check (campaign expectation: flat, unlike Core AI's
   34→12 collapse).

## Reference numbers already in hand (2026-07-21, pre-release)

Controlled OFF/ON pair on one build — litert-lm **main d98e1872** + the 49-line CLI patch
(`litert-thinking-cli-flags.patch`, applied in `~/code/litert-lm-main-wt`), CPU backend,
greedy, max 2048, pinned n=100, canonical extractor
(driver: `scripts/gsm8k_litert_main_thinking.py`):

| arm | GSM8K | median gen tok | end-to-end tok/s |
|---|---:|---:|---:|
| thinking OFF | 87.0 | 376 | 44.0 |
| thinking ON | **90.0** | 824 | 44.6 |

OFF 87.0 ≈ the published v0.13.1/GPU 86.0 (cross-build sanity). ON 90.0 = MLX-OptiQ
thinking; band: Core AI 92 / bf16 anchor 92. Decode flat OFF↔ON (no depth penalty).
Reports: `results/quality/gsm8k_litert-gemma4-e2b-maincpu-{off,thinking}.json`.
These are **reference numbers, not table rows** (main build + CPU vs the table's
v0.13.1 + GPU) — the reply to Yu-hui (2026-07-21) says exactly this.

## Gotchas rediscovered the hard way

- The C++ CPU sampler implements **only `TOP_P`**; `GREEDY`/`TOP_K` return UNIMPLEMENTED
  and `litert_lm_main` **still exits 0** (the conversation callback swallows the error) —
  check output text, not exit codes. Greedy on CPU = `TOP_P + k=1 + p=1.0 + temp=0`.
- The bazel-built `litert_lm_main` GPU executor does not initialize on Apple-Silicon
  macOS; the Swift/xcframework path is the GPU route (WebGPU/Dawn→Metal).
- Worktree clones need `git lfs pull --include "prebuilt/macos_arm64/*"` or the link
  fails with "unknown file type" on the LFS pointer stub.
- `~/code/litert-lm` (the non-worktree checkout) holds #2724 work — don't touch it;
  the thinking worktree is `~/code/litert-lm-main-wt` (removable via
  `git -C ~/code/litert-lm worktree remove litert-lm-main-wt` + `bazel clean --expunge`
  there first if disk is needed).

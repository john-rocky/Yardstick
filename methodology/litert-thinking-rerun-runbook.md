# LiteRT-LM thinking axis — re-run runbook

**Trigger:** the first LiteRT-LM tagged release after v0.14.0 whose Swift API contains
`ThinkingConfig` (added to main Jul 9 2026, commit `ba9a470` — hours after the v0.14.0 cut).

## Status 2026-08-04 — "0.15" landed, but split across two artifact channels

Checked against the runbook triggers:

| artifact | state | thinking |
|---|---|---|
| git tag `v0.15.0-alpha0` (Jul 13) | prerelease, **zero assets** — no xcframework, no CLI binary | Swift source has `ThinkingConfig` (3 hits) |
| GitHub release w/ `CLiteRTLM*.xcframework.zip` | **does not exist yet** (latest with assets is still v0.14.0) | — |
| PyPI `litert-lm` / `litert-lm-api` **0.15.0** (uploaded **2026-08-03**) | released; macOS arm64 wheel ships `liblitert-lm.dylib` | **full**: `litert_lm_thinking_config_create` etc. (19 symbols), Python `ThinkingConfig` API, CLI `--thinking/--thinking_budget` |

So trigger condition 1 (API surface) is met at the alpha tag; condition 2 (C symbols in a
*released binary*) is met by the **pip wheel**, not the xcframework. Consequences:

- **Step 1 below (Swift fork bump) is still blocked** — there are no v0.15.0 release zips
  to checksum. Watch: `https://api.github.com/repos/google-ai-edge/LiteRT-LM/releases`.
- **A released-artifact capture is possible NOW via pip** — and unlike the July bazel CLI,
  the pip dylib's **GPU backend initializes on macOS** (WebGPU/Dawn→Metal, the same
  backend class as the published v0.13.1 GPU row). Probe 2026-08-04, June E2B bundle
  `9262660`: GPU decode ~122 tok/s, CPU ~45 tok/s (≈ the July main-CPU 44.x); thinking ON
  injects `<|think|>` (prefill +6 tok, same as July), decode flat OFF↔ON; thinking trace
  arrives on a separate `channels` key — `content` carries only the answer.
- Environment: repo venv `.venv-litert015/` (`pip install litert-lm-api==0.15.0`).
  Driver: `scripts/gsm8k_litert_pip_thinking.py` (GPU default, `LITERT_BACKEND=cpu` for
  the July-comparable surface; tags `litert-gemma4-e2b-pip015{gpu,cpu}-{off,thinking}`).

### Captured 2026-08-04 — released 0.15.0 pip/GPU pair (n=100)

| arm | GSM8K | median gen tok | decode tok/s (median) |
|---|---:|---:|---:|
| thinking OFF | 89.0 | 362 | 155.5 |
| thinking ON | **92.0** | 812 | 123.2 |

Same +3-point OFF→ON delta as the July main-CPU pair (87.0→90.0), near-identical gen-token
medians (376/824 vs 362/812). ON 92.0 ties Core AI thinking / the bf16 anchor (both 92).
Audit: 0 driver errors; 2 cap-reached (ON q9/q44, `pred=None`, counted wrong — same
treatment as July); decode is auxiliary here (quality axis) — per-question wall-clock proxy
spread OFF 129–171 / ON 109–138, no contention outliers. ON decode sits below OFF
(123 vs 155) consistently, as in the probe — longer sequences, not contention.
Reports: `results/quality/gsm8k_litert-gemma4-e2b-pip015gpu-{off,thinking}.json`;
raw log: `results/raw/2026-08-04-litert-pip015-thinking/run.log`.
- Caveat for docs: the pip dylib is the same engine core but **not the Swift xcframework
  wrapper** the 86.0 table row went through. Publish as its own release-artifact pair;
  the table-surface re-run still waits on the xcframework release (steps below).
- The C++ CPU sampler gotcha (TOP_P only) is bypassed correctly by the API's
  `SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)`; GPU falls back to the statically
  linked C sampler (warning in logs, harmless).

**Verify the xcframework trigger before doing the steps below** (both must pass):

```bash
# 1. Swift surface at the tag
curl -sL https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/<TAG>/swift/Config.swift | grep -c ThinkingConfig   # > 0
# 2. C symbols in the released binary (the Swift wrapper is useless without them)
curl -sL -o /tmp/m.zip https://github.com/google-ai-edge/LiteRT-LM/releases/download/<TAG>/CLiteRTLM_mac.xcframework.zip
unzip -oq /tmp/m.zip -d /tmp/m && nm -gU /tmp/m/CLiteRTLM_mac.xcframework/macos-arm64_x86_64/libCLiteRTLM_mac.dylib | grep -c litert_lm_thinking   # > 0
```

## Steps (xcframework route — the published table's surface)

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
6. **iPhone** (the campaign's one missing thinking cell): `MediaPipeRuntime.swift` can
   take the same `ThinkingConfig` for an on-device thinking decode/depth check
   (campaign expectation: flat, unlike Core AI's 34→12 collapse). Verified 2026-08-04:
   this is hard-blocked on the same release — the iOS arm's engine is the *release-asset*
   `CLiteRTLM.xcframework.zip` (remote binaryTarget in the vendored local package;
   the local Swift wrapper alone can't help, its C symbols must exist in that zip).
   No pip-style side channel exists for iOS, and upstream `prebuilt/ios_arm64/` holds
   only 4 auxiliary dylibs, not the engine. When the tagged release lands: bump the
   vendored binaryTarget URL + checksum, wire `ThinkingConfig` through
   `MediaPipeRuntime.swift`, capture OFF/ON decode on device — in the same pass as
   steps 1–4.

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
  macOS; the Swift/xcframework path is the GPU route (WebGPU/Dawn→Metal) — **and, as of
  0.15.0, so is the pip dylib** (see Status above).
- Worktree clones need `git lfs pull --include "prebuilt/macos_arm64/*"` or the link
  fails with "unknown file type" on the LFS pointer stub.
- `~/code/litert-lm` (the non-worktree checkout) holds #2724 work — don't touch it;
  the thinking worktree is `~/code/litert-lm-main-wt` (removable via
  `git -C ~/code/litert-lm worktree remove litert-lm-main-wt` + `bazel clean --expunge`
  there first if disk is needed).

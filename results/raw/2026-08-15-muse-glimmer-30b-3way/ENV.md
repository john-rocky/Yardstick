# Environment stamp — 2026-08-15 Muse-Glimmer-30B text decoder, Mac 3-way

Core AI vs MLX vs ExecuTorch (Metal), MacBook Pro M4 Max. Summary + how-to-read: `README.md`
in this directory. Pins are mirrored in `environment.lock.json` (capture
"2026-08-15 muse-glimmer-30b 3-way").

## Hardware / OS / toolchain

- Apple M4 Max (40-core GPU), 128 GB RAM
- macOS 27.0 (26A5406e) — verified via `sw_vers` on the capture day
- Toolchain as observed on the capture day: Xcode 26.1.1 (17B100), Apple Swift 6.2.1
  (swift-driver 1.127.14.1), the only Xcode installed. The session notes that forced the
  Core AI rebuild call the toolchain "Xcode 27 beta 5"; that label does not match
  `xcodebuild -version` on this machine and is recorded as unverified.

## Model

`meta-models/Muse-Glimmer-30B` (Apache-2.0) — 30B agentic VLM; **text tower only** in every
arm (27.855 B params, 52 layers, GQA 32q/2kv, `sliding(2048)×3 + full(NoPE)×1` pattern,
vocab 202048 untied). The 2.5 B vision encoder is not part of any arm.

## Arms

| arm | engine build | artifact (HF) | HF revision | quant | weights |
| --- | --- | --- | --- | --- | ---: |
| Core AI | fork `john-rocky/coreai-models` @ `58aab35` + session diff (below) | `mlboydaisuke/Muse-Glimmer-30B-CoreAI` | `dcda323` | int4 block-32 sym, absmax head ("int4hu") | 16.35 GB |
| MLX | mlx-vlm 0.6.13 / mlx 0.32.0 | `mlx-community/Muse-Glimmer-30B-4bit` | `3e7677d` | community 4-bit | 18 GB |
| ExecuTorch | upstream `pytorch/executorch` @ `abc5586` + 1 local commit (below) | `meta-models/Muse-Glimmer-30B-ExecuTorch-PTE` | `fc6fa93` | Meta official k-quant | 17.9 GB |

### Core AI — ⚠ NOT this repo's pinned 0.2.0 arm ("fork engine (reference)")

- Runner: `llm-runner` release build (2026-08-15) of the public fork
  `github.com/john-rocky/coreai-models` @ `58aab35` (branch `zoo-0.2`, tag `0.2.1-zoo` +1),
  plus two uncommitted Swift diffs archived at
  `patches/coreai-models-58aab35-muse-bench-session.diff`:
  1. `LanguageModelCapabilities` init-label fix — required to build at all on this OS seed
     (the released package's prebuilt CLI tools die in dyld against 26A5406e);
  2. `LanguageConfig` turn-end-token broadening — **non-binding for these runs**: every run
     generated the full 192-token budget, no early stop occurred.
- This repo pins the Core AI arm to Apple's released **0.2.0** (`environment.lock.json`).
  The fork is 98 files / ~27.5k lines beyond 0.2.0. A 0.2.0 re-measure is **not available**:
  0.2.0 does not build or run on this OS seed (dyld/ABI, above), and it has no Muse-Glimmer
  authoring — the bundle itself was exported by the fork toolchain. This is the same
  disclosure class as the Gemma-4 "patched engine (reference)" rows
  (`methodology/core-ai-arm-provenance.md`); read the number as the fork engine's, not 0.2.0's.
- Invocation: `--max-tokens 192 --temperature 0.0 --inference-engine-variant coreai-pipelined
  --warmup off`.
- ⚠ **Self-made artifact.** The other two arms consume artifacts someone else built
  (mlx-community; Meta). This bundle was exported by this repo's author
  (`conversion/export_muse_glimmer_decode_pipelined.py` in `john-rocky/coreai-model-zoo`,
  branch `muse-glimmer-30b` @ `a11f0f8`), gated token-exact against the fp16 authoring oracle.
  Third parties can run the published bundle; they **cannot yet rebuild it** — the export
  depends on authoring files not pushed anywhere as of this capture.

### MLX

- `python -m mlx_vlm.generate --max-tokens 192 --temperature 0 --verbose`.
  mlx-lm 0.31.3 has no `muse_glimmer` support; mlx-vlm is the community-published path for
  this model. Full venv freeze: `tools/python-envs/mlx-vlm-muse.requirements.txt`.

### ExecuTorch

- Meta's own example runner (`examples/models/muse-glimmer/solo_runner`, in upstream tree at
  `abc5586` on `main`) — fairness rule 10 (prefer the official runtime SDK) satisfied.
- Built with one local commit on top ("treat an empty LLM data path as absent"), archived at
  `patches/executorch-abc5586-empty-data-path.diff`. Arg-handling in the runner factory, not
  a performance change; unpushed as of this capture, so it is vendored here.
- `.pte` variant: `muse-glimmer-k-quant-17G-128K-text-solo-metal`. Meta's README states the
  `metal` backend is MLX-native — which is why raw MLX is the third arm (it separates
  "beats ExecuTorch" from "beats MLX").
- Invocation: `--max_new_tokens 192 --temperature 0 --ignore_eos=true`.

## Protocol

- 2 free-form prompts (in the driver), greedy, batch 1, **192 new tokens**
- **Interleaved CA → ET → MLX per prompt**, 45 s cooldown between every run,
  2 prompts × 2 rounds (fairness rule 11 — this capture is its worked example)
- Driver: `scripts/bench_muse_glimmer_3way_mac.sh` (generalized from the as-run script;
  every path env-overridable)
- Hardware state (fairness rule 9): **not instrumented** by the as-run driver — this is a
  CLI-log capture, not a BenchmarkResult JSONL, so charging/low-power/thermal at start were
  not recorded. The interleaving plus the ≤1.8 % per-arm spread across rounds is the
  evidence that order and thermal state did not decide the result.

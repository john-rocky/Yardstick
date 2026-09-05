# Patches applied to the vendored LiteRT-LM Swift package

`ios/BenchmarkApp/Vendored/` is gitignored, so anything patched there is invisible to git and
is lost the moment the package is re-vendored. These patch files are the tracked copy. Re-apply
them after any re-vendor:

```bash
cd ios/BenchmarkApp/Vendored/LiteRT-LM
for p in ../../Vendored-patches/*.patch; do patch -p1 < "$p"; done
```

Baseline: LiteRT-LM **v0.16.0** (regenerated 2026-08-17 from the v0.16.0 re-vendor; the
original patch was against v0.13.0 and no longer applied — the upstream signature gained
visionBackend/audioBackend/prompt parameters). Each patch is a diff against the same tag's
`swift/` sources on GitHub, so `patch` applying cleanly is itself the check that the vendored
tree is that version and otherwise unmodified.

## 0001 — `benchmark(maxNumTokens:)`

`LiteRTLM.benchmark()` is the entry point the agreed Gemma-4 protocol specifies for prefill
(`methodology/agreed-protocol-gemma4.md`). The protocol pins two things the stock signature
cannot express together:

- prefill exactly 1024, decode 256 — `benchmark()` takes these;
- context length forced to **2048** — `benchmark()` does not take it, and hardcodes
  `maxNumTokens: max(prefillTokens, decodeTokens) + 32`, i.e. **1056** at 1024x256.

KV is pre-allocated to `maxNumTokens`, so this is not a cosmetic difference: it is most of what
a memory cell measures, and it is why our footprints sat far under the model card's 1,450 MB.
`Engine.initializeForBenchmark` is module-internal, so the override cannot be written from
outside the package — hence a patch rather than a wrapper in our own sources.

The patch adds an optional `maxNumTokens:` parameter and defaults it to the stock expression,
so unpatched call sites keep upstream behaviour exactly.

**This is worth raising upstream** — the Swift `benchmark()` helper cannot reproduce the
configuration the LiteRT team benchmarks Gemma-4 at.

### If you re-vendor and forget

`MediaPipeRuntime.nativeBenchmark` passes `maxNumTokens:` explicitly and never relies on the
default, so a stock package **fails to compile** instead of silently reverting the context to
1056. That is deliberate: a build error is a signal, an off-protocol number is not.

## 0002 — `ExperimentalFlags.forceFloat32Activations`

`MediaPipeRuntime` sets `ExperimentalFlags.forceFloat32Activations = true` when the app is
launched with `--litert-fp32-act` (the diagnostic knob for the speculative-decoding leg: the
iOS engine defaults to fp16 activations, the desktop CLI to fp32, and fp16 rounding of the
drafter/verifier handoff is a candidate cause for the acceptance collapse seen on iOS).

The stock wrapper has no such flag (checked against upstream `main` on 2026-09-06), and the
activation data type is applied inside `Engine.init`, which is where the C setting
`litert_lm_engine_settings_set_activation_data_type` is available — so, like 0001, this cannot
be written from outside the package. The patch adds the flag (opt-in gated like the other
experimental flags, default `false`) and the one call in `Engine.init`. Unpatched call sites
keep upstream behaviour exactly.

Until 2026-09-06 this edit lived only in one working clone's `Vendored/` and was invisible to
git; the `catalog-2026-09-04` merge added the tracked copy. The compile error without it
(`type 'ExperimentalFlags' has no member 'forceFloat32Activations'`) is the same kind of signal
as 0001's: a stock package fails to build rather than silently measuring fp16.

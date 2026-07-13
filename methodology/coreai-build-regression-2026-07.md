# coreai-build regression on macOS 27 beta 26A5378j — Core AI iOS bundles unassemblable (2026-07-13)

> **RESOLVED same evening — root cause found and verified.** The updated beta's odiec only
> accepts IR carrying "AICode versioned locations", which **coreai-torch 0.4.1** emits and
> 0.4.0 (the June pin) does not. With a genuine 0.4.1 export
> (`.venv/bin/coreai.llm.export …` — NOT `uv run`, which silently resyncs the venv back to
> the 0.4.0 pin and invalidated the first probe), BOTH paths work again on 26A5378j:
> in-process JIT (`llm-benchmark` on qwen3-0.6b ct041: 578 tok/s gen probe) and the
> `coreai-build` iOS AOT CLI (h18p `.aimodelc` produced). Discovery credit: the leaderboard
> project's eval-driver compiled a coreai-torch-0.4.1 gemma-4-e2b export successfully at
> 13:06 on this build (`~/Library/Caches/coreai-cache/26A5378j/`), proving a working path
> existed. Two residual consequences: (1) every 0.4.0-era IR — including the archived
> "irreplaceable" macOS-26 artifacts — is now uncompilable on this OS; (2) 0.4.1 IRs are a
> NEW artifact generation whose lowering may differ from the June bundles, so re-measured
> Core AI numbers must be labelled with their artifact lineage (coreai-export-lowering.md
> discipline). Also note: the June-built `llm-benchmark` binary is ABI-broken against the
> updated FoundationModels.framework — rebuild with
> `DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.3.app swift build -c release`.
> The original (superseded) analysis follows for the record.

**Impact**: the Core AI Qwen3-0.6B (`qwen3_0_6b_{ane,gpu}`) and 1.7B (`qwen3_1_7b_gpu`) iPhone
bundles cannot be (re)assembled on this Mac, so their **warm** rows cannot be captured in the
2026-07 warm re-capture campaign. The June-compiled Qwen3-4B bundles already side-loaded on the
device (`Documents/CoreAIModels/qwen3_4b_{ane,gpu}`, compiled 2026-06-18) still load and run, so
the Core AI warm story is carried by the 4B 3-way instead.

## Symptom

Every `xcrun coreai-build compile … --platform iOS|macOS` on this machine now aborts:

```
$ xcrun coreai-build compile exports/qwen3_0_6b_ios_pure4bit/qwen3_0_6b_ios_pure4bit.aimodel \
    --platform iOS --preferred-compute neural-engine --architecture h18p --output /tmp/out
# toolchain 3600.67.5.8.1 (MetalToolchain v27.1.5194.15):
error: expected AICode versioned location, got: loc(fused<…>)
error: Failed to convert to versioned IR
LLVM ERROR: cannot unwrap empty `odiec_module_t`
Abort trap: 6
# toolchain 3600.75.3 (MetalToolchain 27A5218h, via Xcode 27.0 Beta 3):
LLVM ERROR: cannot unwrap empty `odiec_module_t`
Abort trap: 6
```

## Scope (all measured 2026-07-13)

| variable | tried | result |
|---|---|---|
| IR era | archived macOS-26-era IR (Jun 9), fresh exports (today, coreai-torch 0.4.0) | both fail |
| platform | `--platform iOS` (h18p) and `--platform macOS` (h13c) | both fail |
| preferred-compute | gpu / neural-engine / none | all fail |
| toolchain | 3600.67.5.8.1 (v27.1.5194.15) and 3600.75.3 (27A5218h, downloaded via `xcodebuild -downloadComponent metalToolchain` under Xcode 27 Beta 3) | both fail (error shape differs, same abort) |
| python wheels | unchanged vs June pins (coreai-core 1.0.0b1 / coreai-torch 0.4.0 / coreai-opt 0.2.0 / coreai-models 0.1.0, torch 2.9.0) | n/a |

Export (`coreai.llm.export`) itself works — `exports/qwen3_0_6b_ios_pure4bit/`,
`exports/qwen3_0_6b_dynamic/` (fresh 4bit), `exports/qwen3_1_7b_{dynamic,ios_pure4bit}/` were all
produced today. Only the AOT **compile** stage is broken.

## Timeline

- 2026-06-18: compiles work (`_MATRIX_COMPLETION_STATE.md`, macOS 26A5353q, coreai-build
  3600.67.5.8.1) — produced the 4B/8B bundles.
- between Jun 19 and Jul 8: Mac updated to a newer macOS 27 beta (now 26A5378j).
- 2026-07-11: first `coreai-build` crash reports in `~/Library/Logs/DiagnosticReports`
  (`coreai-build-2026-07-11-165644.ips`), i.e. the regression predates this session.
- Dead ends checked: the two older cached MetalToolchain assets (17A324 / 17B54, Dec 2025 =
  macOS-26 era) contain **no coreai-build** (tool first shipped in the 27 toolchain); the June
  compiled 0.6B/1.7B bundles are unrecoverable (old app container uninstalled, exports/ wiped).

`odiec` is the on-device-inference compiler layer that ships with the OS, not with the
toolchain — consistent with both toolchain versions failing identically after an OS beta update.

## Unblock paths (for a later session)

1. Next macOS 27 beta / MetalToolchain drop → retry
   `scripts/export_coreai_qwen3.sh 0_6b Qwen/Qwen3-0.6B 4096 h18p` (exports are pinned on disk;
   only the assemble step reruns).
2. Any second Mac still on 26A5353q-era macOS 27 beta can compile the pinned IRs as-is.
3. If Apple's feedback channel is worth it: file with the two .ips crash logs.

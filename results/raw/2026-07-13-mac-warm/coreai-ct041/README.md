# Core AI Mac sweep — coreai-torch 0.4.1 ("ct041") lineage (2026-07-13 night)

Apple `llm-benchmark` (rebuilt with the Xcode 27 Beta 3 SDK; the June binary is ABI-broken
on 26A5378j), June protocol: `--prompt-tokens 512 --generation-tokens 1024 --num-trials 5`,
Mac Studio M4 Max, macOS 27.0 26A5378j, GPU idle. IRs are fresh coreai-torch **0.4.1**
exports (`exports/*_dynamic_ct041`) — the 0.4.0-era IRs no longer compile on this OS
(methodology/coreai-build-regression-2026-07.md).

## Generation tok/s (vs the June Mac table, cold-era 0.4.0 artifacts)

| Model | ct041 today | June ref | Δ |
|---|--:|--:|---|
| qwen3-0.6b | 503.1 | ~500 (27β re-export era; macOS-26 artifact was 1,121) | ✓ |
| qwen3-1.7b | 298.2 | 239.1 | **+25% — outlier, needs a second look** (June value may have had its own issue; flagged, not published as a comparison) |
| qwen3-4b | 148.6 | 145.4 (official-recipe matrix) | ✓ +2% |
| deepseek-r1-1.5b | 324.4 | 319.5 | ✓ |
| gemma3-1b | 331.8 | 327.2 | ✓ |
| tinyswallow-1.5b | 326.5 | 324.1 | ✓ |
| vibethinker-1.5b | 326.9 | 322.7 | ✓ |
| olmo2-1b | 386.4 | 384.4 | ✓ |
| smollm3-3b | 196.2 | 192.9 | ✓ |
| llama32-3b | 195.3 | 198.3 | ✓ |
| ministral3-3b | 188.8 | 186.0 | ✓ |
| phi4-mini | **2.8 (degenerate)** | ✗ compile wall (June) | the 0.4.1 toolchain now COMPILES partial-rotary (both AOT h18p and Mac JIT) but runtime performance is pathological (gen 2.8 tok/s, prompt 79.7) — recorded as measured-abnormal; not usable, not a wall |

**Key validation:** for 10 of 12 models the ct041 lineage reproduces the June Core AI Mac
numbers within ±2% — the 0.4.1 re-export is measurement-equivalent to the June artifacts for
the fleet (the qwen3-0.6b macOS-26-era artifact remains the known exception, and qwen3-1.7b
is flagged). This licenses the iPhone campaign's ct041 bundles as comparable hardware-lineage,
with the label kept anyway.

Raw per-model JSON: `*.json` (llm-benchmark `--output-json`), 5 trials each. phi4-mini json
absent (timeout at full protocol); its degenerate probe is documented above.

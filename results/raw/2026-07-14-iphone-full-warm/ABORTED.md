# Campaign r1 — ABORTED at cell 8/50 (04:04), superseded by `2026-07-14-iphone-full-warm-r2`

Stopped deliberately after discovering that every ct041-lineage Core AI cell
hard-crashes at load: the device (iOS 27 beta 24A5355q, June-era) SIGSEGVs in
`MPSGraphAICodeCompilerDelegate getInitializedAICodeBytecode` while re-consuming
AICode bytecode that the Mac's newer beta toolchain (macOS 26A5378j / Xcode 27
Beta 3) emitted into the `.aimodelc` (crash logs: `BenchmarkApp-2026-07-14-034527.ips`,
`-034830.ips`). A `--min-deployment-version 26.0` recompile is rejected
("Model requires OS 27.0"), so ct041 artifacts cannot be made compatible from
the Mac side — Core AI folders on device were swapped to June-compiled bundles
(proven lineage; deepseek GPU probe ran decode 67.14 cold) and the full matrix
was restarted as r2 in a single measurement session.

The 4 MLX/LiteRT cells completed here are valid but re-measured in r2 for
same-session comparability; the device-jsonl files are kept for reference only.
Do NOT import rows from this directory.

Cells completed before abort:
- mlx Qwen3-0.6B: cold 167.33, warm 167.34/168.66/166.93 (verdict OK)
- litert Qwen3-0.6B: cold 119.75, warm 119.50/119.46/119.35 (verdict OK)
- core-ai qwen3-0.6b-ane/-gpu: CRASH (ct041, see above)
- mlx Qwen3-1.7B: cold 66.33, warm 66.14/66.14/66.13 (verdict OK)
- litert qwen3-1.7b-int4: verdict OK

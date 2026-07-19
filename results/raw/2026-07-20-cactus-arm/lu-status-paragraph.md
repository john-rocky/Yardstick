# Cactus arm status (for Lu — one paragraph, EN)

Cactus is now the 7th arm of the Gemma-4-E2B table, integrated the same way as the other
engines: an in-app adapter over their C FFI (`cactus_init`/`cactus_complete`), their
xcframework built from source and vendored, same headless protocols and JSONL, cloud-handoff
and telemetry forced off. The build-selection audit produced the headline: the CQ4 bundle
their CLI ships as the default download is reasoning-dead — GSM8K 3.0% (n=100, the table's
single shared harness), failing even one-step arithmetic in the on-device sanity — while the
pre-July-9 CQ4 they renamed to "-uncalibrated" in the same HF repo scores 87.0%, the
QAT-class band (between Core AI's 88 and LiteRT's 85). The two builds are speed-identical
on device (50.6 vs 50.2 tok/s), so this is purely an artifact-lineage defect, and their
newer mixed-precision prebuilts (cq3.26/cq2.54) probe at 4%/16% — the whole post-July-9
"calibrated" lineage is broken; their published CQ claims (GSM8K 71–74) only match the
artifact they demoted. The row therefore runs the uncalibrated build as best-usable (the
MLX-OptiQ rule), with the shipped default as a second row at 3.0%. Device cells, iPhone 17
Pro at protocol parity: decode 50.6 tok/s — the clear #2 arm; a same-session LiteRT
control re-anchor (60.9 today vs its published 52.7, the known session drift, measured
not assumed) puts Cactus at 0.83× LiteRT's decode at matched conditions, at +2 GSM8K pts
over its 85.0 — ITL p50 19.6 ms, peak ~1,061 MB (2.2× LiteRT, far under MLX's 3–4.7 GB),
deep-context p=1024/g=256 decode 40.3 tok/s with prefill 683 tok/s (LiteRT's prefill lead
stays ~5×), energy 0.322 J/tok on the 600 s battery-delta protocol (2.6× LiteRT's 0.122 —
LiteRT keeps all three of speed/memory/energy). Thinking mode: their toggle works (unlike
LiteRT's locked-out API) but buys nothing (87→87). Integration friction worth passing on:
their shipped xcframework header can't compile as-is (bundles a C++ include the FFI never
uses), their bundle graphs embed the converter machine's absolute weight paths, and their
messages parser is JSON-key-order-sensitive (role must precede content — any
order-randomizing serializer silently generates from an empty prompt; Swift's
JSONSerialization does exactly that ~half the time).

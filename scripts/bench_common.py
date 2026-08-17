"""Shared display tables for the result renderers (RESULTS.md and
LEADERBOARD.md must not drift on naming).

LOGICAL_MODELS: different runtimes pull weights from different HF orgs, so the
same logical model ends up under several string IDs; entries map a
case-insensitive substring of model.id to a canonical display name. Order
matters — first hit wins, so more specific patterns come first (see the OptiQ
and CQ4 comments: builds that must not pool stay distinct).
"""

DEVICE_DISPLAY = {
    "m4max": "Mac M4 Max",
    "m3air": "MacBook Air M3",
    "m4air": "MacBook Air M4",
    "m1pro": "MacBook Pro M1",
    "m2pro": "MacBook Pro M2",
    "m3pro": "MacBook Pro M3",
    "m4pro": "MacBook Pro M4",
    "m2max": "MacBook Pro M2 Max",
    "m3max": "MacBook Pro M3 Max",
    "iphone15pro": "iPhone 15 Pro",
    "iphone16pro": "iPhone 16 Pro",
    "iphone17pro": "iPhone 17 Pro",
    "iphone17promax": "iPhone 17 Pro Max",
    "iphone17air": "iPhone 17 Air",
    "ipadprom4": "iPad Pro M4",
    # modelIdentifier keys (campaign-shaped records carry the identifier, not a label)
    "Mac16,9": "Mac Studio (M4 Max)",
    "iPhone18,1": "iPhone 17 Pro",
}


LOGICAL_MODELS: list[tuple[str, str]] = [
    # Gemma 4
    ("gemma-4-26b-a4b", "Gemma 4 26B-A4B (MoE)"),
    ("gemma-4-31b",     "Gemma 4 31B"),
    # OptiQ before the generic e2b pattern: MLX has two 4-bit builds of E2B in the table
    # (quality-best QAT OptiQ vs speed-best PTQ) and they must not pool in the pivots.
    ("gemma-4-e2b-it-qat-optiq", "Gemma 4 E2B (QAT OptiQ)"),
    # Cactus ships two CQ4 lineages of E2B (same repo, different files) that must not
    # pool either: the pre-07-09 "uncalibrated" (the row: GSM8K 87.0) vs the shipped
    # default "calibrated" (footnote: GSM8K 3.0). Uncal pattern first — it contains cq4.
    ("gemma-4-e2b-it-cq4-uncal", "Gemma 4 E2B (CQ4 uncalibrated)"),
    ("gemma-4-e2b-it-cq4",       "Gemma 4 E2B (CQ4 shipped default)"),
    ("gemma-4-e2b",     "Gemma 4 E2B"),
    ("gemma-4-e4b",     "Gemma 4 E4B"),
    ("gemma4-e2b",      "Gemma 4 E2B"),
    ("gemma4-e4b",      "Gemma 4 E4B"),
    # Gemma 3
    ("gemma-3-270m",    "Gemma 3 270M"),
    ("gemma-3-1b",      "Gemma 3 1B"),
    # Qwen 3.5
    ("qwen3.5-35b-a3b", "Qwen 3.5 35B-A3B (MoE)"),
    ("qwen3.5-27b",     "Qwen 3.5 27B"),
    ("qwen3.5-9b",      "Qwen 3.5 9B"),
    ("qwen3.5-2b",      "Qwen 3.5 2B"),
    ("qwen3.5-0.8b",    "Qwen 3.5 0.8B"),
    # Qwen 3
    ("qwen3-1.7b",      "Qwen 3 1.7B"),
    ("qwen3-0.6b",      "Qwen 3 0.6B"),
    # Qwen 2.5
    ("qwen2.5-0.5b",    "Qwen 2.5 0.5B"),
    ("qwen2.5-1.5b",    "Qwen 2.5 1.5B"),
    ("qwen2.5-3b",      "Qwen 2.5 3B"),
    ("qwen2.5-7b",      "Qwen 2.5 7B"),
    # LFM / Llama / others
    ("lfm2.5-350m",     "LFM 2.5 350M"),
    ("lfm-2.5-350m",    "LFM 2.5 350M"),
    ("llama-3.2-1b",    "Llama 3.2 1B"),
    ("llama-3.3-1b",    "Llama 3.3 1B"),
    ("llama-3.3-3b",    "Llama 3.3 3B"),
    ("smollm3-3b",      "SmolLM 3B"),
]


def logical_model(model_id: str) -> str:
    """Canonical display name for a model, collapsing per-runtime HF IDs."""
    needle = model_id.lower()
    for pat, name in LOGICAL_MODELS:
        if pat in needle:
            return name
    return model_id


def runtime_display(runtime: str) -> str:
    return {
        "mlx-swift": "mlx-swift",
        "llama.cpp": "llama.cpp",
        "coreml-llm": "coreml-llm",
        "executorch": "executorch",
        "anemll": "anemll",
        "litert-lm": "litert-lm",
        "apple-fm": "apple-fm",
        "core-ai": "core-ai",
    }.get(runtime, runtime)


def corrected_quant(runtime: str, model_id: str, quant: str) -> tuple[str, bool]:
    """Apply the audited in-place quantization-label correction (quant-label-rule):
    Gemma-4 .litertlm bundles are the wNa8o8 mobile schema, not uniform int4 —
    early rows recorded "INT4 (QAT)" before the 2026-07-17 audit corrected the
    label for the SAME artifact. Scope is deliberately narrow (litert-lm +
    gemma-4 only); other labels render as recorded. Returns (label, corrected?).
    """
    if (runtime == "litert-lm" and "gemma-4" in model_id.lower()
            and quant == "INT4 (QAT)"):
        return "wNa8o8 (int2/int4/int8 + int8 activations, QAT)", True
    return quant, False

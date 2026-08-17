"""Parsers for the Android engine CLIs' console output.

Formats verified against the v0.16.0 sources (runtime/framework/io_types.cc
BenchmarkInfo operator<<) and llama.cpp b8999. Absent metrics stay absent —
never derived, never defaulted (a fabricated field is worse than a hole).
"""
import json
import re

# litert_lm_main --benchmark / task mode (io_types.cc)
RE_TTFT = re.compile(r"Time to first token: ([\d.]+) s")
RE_PREFILL = re.compile(r"Prefill Speed: ([\d.]+) tokens/sec")
RE_DECODE = re.compile(r"Decode Speed: ([\d.]+) tokens/sec")
# real v0.16.0 device output: "Prefill Turn 1: Processed 20 tokens in 764.48ms duration."
RE_PREFILL_TOKENS = re.compile(r"Prefill Turn \d+: Processed (\d+) tokens")
RE_DECODE_TOKENS = re.compile(r"Decode Turn \d+: Processed (\d+) tokens")
RE_PEAK_MEM = re.compile(r"[Pp]eak memory.*?([\d.]+) *(MB|GB|bytes)")

# llama-cli perf lines. b8999's -st mode prints the bracket summary
# "[ Prompt: 217.7 t/s | Generation: 20.6 t/s ]" (no token counts — those stay
# absent); the llama_perf_context_print form is kept as a fallback for other
# builds.
RE_LLAMA_BRACKET = re.compile(
    r"\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]")
RE_LLAMA_PROMPT = re.compile(
    r"prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_LLAMA_EVAL = re.compile(
    r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second")


def _last_float(rx, text):
    hits = rx.findall(text)
    return float(hits[-1]) if hits else None


def parse_litert(text):
    """litert_lm_main console -> metrics dict (BenchmarkResult field names)."""
    m = {}
    ttft = _last_float(RE_TTFT, text)
    if ttft is not None:
        m["firstTokenLatencyMS"] = ttft * 1000
    prefill = _last_float(RE_PREFILL, text)
    if prefill is not None:
        m["promptTokensPerSecond"] = prefill
    decode = _last_float(RE_DECODE, text)
    if decode is not None:
        m["decodeTokensPerSecond"] = decode
    pt = RE_PREFILL_TOKENS.findall(text)
    if pt:
        m["promptTokenCount"] = sum(int(x) for x in pt)
    dt = RE_DECODE_TOKENS.findall(text)
    if dt:
        m["generatedTokenCount"] = sum(int(x) for x in dt)
    peak = RE_PEAK_MEM.search(text)
    if peak:
        val, unit = float(peak.group(1)), peak.group(2)
        mb = val * 1024 if unit == "GB" else val / (1024 * 1024) if unit == "bytes" else val
        m["memoryPeakEngineReportedMB"] = mb
    return m


def parse_llama_cli(text):
    """llama-cli perf print -> metrics dict. No TTFT (absent, not derived)."""
    m = {}
    b = RE_LLAMA_BRACKET.search(text)
    if b:
        m["promptTokensPerSecond"] = float(b.group(1))
        m["decodeTokensPerSecond"] = float(b.group(2))
        return m
    p = RE_LLAMA_PROMPT.search(text)
    if p:
        m["promptTokenCount"] = int(p.group(1))
        m["promptTokensPerSecond"] = float(p.group(2))
    e = RE_LLAMA_EVAL.search(text)
    if e:
        m["generatedTokenCount"] = int(e.group(1))
        m["decodeTokensPerSecond"] = float(e.group(2))
    return m


def parse_llama_bench_json(text):
    """llama-bench -o json -> list of per-test dicts {kind, avg_ts, n_prompt, n_gen}."""
    data = json.loads(text[text.index("["):text.rindex("]") + 1])
    out = []
    for t in data:
        kind = "prefill" if t.get("n_prompt", 0) > 0 and t.get("n_gen", 0) == 0 else "decode"
        out.append({"kind": kind, "avg_ts": t.get("avg_ts"),
                    "n_prompt": t.get("n_prompt"), "n_gen": t.get("n_gen"),
                    "stddev_ts": t.get("stddev_ts")})
    return out

#!/usr/bin/env python3
"""One-shot auditor for this session's device pulls.

Prints every cactus/litert record in a pull dir with the fields the audit
decisions need: task, prompt_tokens sanity (the key-order bug's tell: short-chat
must be ~20, long-context ~1090, NOT 9), thermal gating, decode, memory, and for
energy records the battery-delta validity (joules non-nil, energySource).

    python3 audit_pull.py /tmp/cactus-final
"""
import json, sys
from pathlib import Path

pull = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cactus-final")
rows = []
for f in sorted(pull.glob("*.json")):
    if "cactus" not in f.name and "litert" not in f.name:
        continue
    d = json.loads(f.read_text())
    m = d.get("metrics", {})
    ts = f.name.rsplit("_", 1)[-1].replace(".json", "")
    rec = {
        "ts": ts,
        "runtime": d.get("runtime"),
        "model": d.get("model", {}).get("id", "?").split("/")[-1],
        "task": d.get("task"),
        "ptok": m.get("promptTokenCount"),
        "gtok": m.get("generatedTokenCount"),
        "decode": round(m.get("decodeTokensPerSecond", 0), 1),
        "prefill": round(m.get("promptTokensPerSecond", 0), 0),
        "ttft_ms": m.get("firstTokenLatencyMS"),
        "itl50": round(m.get("interTokenLatencyP50MS", 0), 1),
        "peakMB": round(m.get("memoryPeakDuringDecodeMB", 0)),
        "therm": f"{m.get('initialThermalState','?')}->{m.get('peakThermalState','?')}",
        "load_s": round(m.get("loadTimeSeconds", 0), 1),
    }
    if d.get("task") == "energy":
        for k in ("energyJoules", "energyJoulesPerToken", "energySource",
                  "batteryDeltaPercent", "energyMeasurementWindowSeconds",
                  "averagePackagePowerW"):
            if k in m:
                rec[k] = m[k]
    # validity flags
    flags = []
    if d.get("task") == "short-chat" and (rec["ptok"] or 0) < 15:
        flags.append("EMPTY-PROMPT-BUG")
    if d.get("task") == "long-context-1024" and (rec["ptok"] or 0) < 900:
        flags.append("EMPTY-PROMPT-BUG")
    if m.get("initialThermalState") not in (None, "nominal"):
        flags.append("NOT-NOMINAL")
    rec["flags"] = ",".join(flags) or "ok"
    rows.append(rec)

for r in rows:
    print(" ".join(f"{k}={v}" for k, v in r.items()))

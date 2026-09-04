#!/usr/bin/env python3
"""Score backend-fidelity runs → per-arm tables + backend-delta report.

Consumes the pinned suite (evaldata/fidelity_suite_v1.jsonl) plus one or more
runner output JSONL files. An output record needs at least:

  {"case_id": ..., "backend": "cpu"|"gpu"|..., "rep": 1, "output": "..."}

and should carry "runtime", "model", "device" (defaulted from the filename
otherwise). Records with an "error" field are scored as ERROR.

Per-field classification, in priority order:
  PASS           expected string present verbatim in the output
  DIGIT_CORRUPT  format skeleton matches (same punctuation/layout) but the digits
                 differ — the LiteRT-LM #2814 signature; the smoking gun
  FORMAT_DRIFT   the digit sequence survives but the formatting changed
                 (separators dropped, date rewritten, ...) — a fidelity failure,
                 but semantically intact
  MISSING        value absent; flags: EMPTY / DEGENERATE / ERROR

Claims discipline (refute-first): a case enters the backend-delta table only if
the SAME arm's cpu backend passes it in ≥90% of reps ("claim-eligible"). A case
the model can't do on cpu is a capability limit, not a backend bug, and is
reported separately.

    python3 scripts/fidelity_score.py results/fidelity/*.jsonl
    python3 scripts/fidelity_score.py --selftest
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUITE_PATH = REPO / "evaldata" / "fidelity_suite_v1.jsonl"
REPORT_DIR = REPO / "results" / "fidelity"
CPU_GATE = 0.9


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------- classification

def skeleton_re(value, payload_class):
    """Regex matching the exact layout of `value` with digit slots freed."""
    out = []
    for ch in value:
        if ch.isdigit():
            out.append(r"\d")
        elif payload_class == "uuid" and ch in "abcdef":
            out.append("[0-9a-f]")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out))


def digits_of(s):
    return "".join(ch for ch in s if ch.isdigit())


def degenerate(text):
    """Loop / special-token spam heuristic (same spirit as quality_check.py)."""
    words = text.split()
    if len(words) >= 10:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and Counter(grams).most_common(1)[0][1] >= 3:
            return True
        if len(set(words)) / len(words) < 0.30:
            return True
    if len(text) >= 40 and len(set(text)) < 15:
        return True
    return text.count("<|") >= 5 or text.count("<pad>") >= 5


def classify_field(value, payload_class, output):
    if value in output:
        return "PASS", None
    m = skeleton_re(value, payload_class).search(output)
    if m:
        return "DIGIT_CORRUPT", {"expected": value, "got": m.group(0)}
    dseq = digits_of(value)
    if dseq and dseq in digits_of(output):
        return "FORMAT_DRIFT", {"expected": value}
    return "MISSING", None


RANK = {"PASS": 0, "FORMAT_DRIFT": 1, "MISSING": 2, "DIGIT_CORRUPT": 3, "ERROR": 4}


def classify_rep(case, rec):
    """→ (case-level class, per-field dict, evidence list, flags)"""
    if rec.get("error"):
        return "ERROR", {}, [], ["ERROR"]
    output = rec.get("output", "") or ""
    flags = []
    if not output.strip():
        flags.append("EMPTY")
    elif degenerate(output):
        flags.append("DEGENERATE")
    fields, evidence = {}, []
    for fname, value in case["expected"].items():
        cls, ev = classify_field(value, case["payload_class"], output)
        fields[fname] = cls
        if ev:
            ev["field"] = fname
            evidence.append(ev)
    worst = max(fields.values(), key=lambda c: RANK[c])
    return worst, fields, evidence, flags


# --------------------------------------------------------------- aggregation

def arm_key(rec, src_name):
    return (rec.get("runtime", src_name), rec.get("model", "?"),
            rec.get("device", "?"), rec.get("backend", "?"))


def score(suite, records_by_src):
    cases = {c["id"]: c for c in suite}
    # arm → case_id → list of (class, evidence, output, rep)
    arms = defaultdict(lambda: defaultdict(list))
    unknown = Counter()
    for src, records in records_by_src.items():
        for rec in records:
            case = cases.get(rec.get("case_id"))
            if not case:
                unknown[rec.get("case_id")] += 1
                continue
            cls, fields, ev, flags = classify_rep(case, rec)
            arms[arm_key(rec, src)][case["id"]].append(
                {"class": cls, "fields": fields, "evidence": ev,
                 "flags": flags, "output": rec.get("output", ""), "rep": rec.get("rep")})
    return arms, unknown


def pass_rate(reps):
    return sum(r["class"] == "PASS" for r in reps) / len(reps)


def stability(reps):
    return len({r["output"] for r in reps})


def fmt_pct(x):
    return f"{100 * x:.0f}%"


def report(arms, cases_by_id, out_path):
    lines = ["# Backend fidelity report", "",
             f"Suite: `{SUITE_PATH.name}` · gate: cpu pass-rate ≥ {fmt_pct(CPU_GATE)}", ""]
    # group arms by (runtime, model, device); pair cpu vs accelerated backends
    groups = defaultdict(dict)
    for (runtime, model, device, backend), casemap in arms.items():
        groups[(runtime, model, device)][backend] = casemap

    for (runtime, model, device), backends in sorted(groups.items()):
        lines += [f"## {runtime} · {model} · {device}", ""]
        cpu = backends.get("cpu")
        for backend, casemap in sorted(backends.items()):
            n_reps = sum(len(v) for v in casemap.values())
            all_classes = Counter(r["class"] for v in casemap.values() for r in v)
            lines.append(f"**{backend}** — {len(casemap)} cases · {n_reps} generations · "
                         + ", ".join(f"{k} {v}" for k, v in sorted(all_classes.items())))
            nondet = [cid for cid, v in casemap.items() if len(v) > 1 and stability(v) > 1]
            if nondet:
                lines.append(f"  · non-deterministic under greedy on {len(nondet)} cases: "
                             + ", ".join(sorted(nondet)[:8]) + ("…" if len(nondet) > 8 else ""))
            lines.append("")

        if not cpu:
            lines += ["_No cpu arm — backend-delta table skipped (no baseline)._", ""]
            continue
        eligible = {cid for cid, v in cpu.items() if pass_rate(v) >= CPU_GATE}
        capability = sorted(set(cpu) - eligible)
        for backend, casemap in sorted(backends.items()):
            if backend == "cpu":
                continue
            rows = []
            for cid in sorted(eligible & set(casemap)):
                cpu_r, acc_r = pass_rate(cpu[cid]), pass_rate(casemap[cid])
                if acc_r < cpu_r:
                    classes = Counter(r["class"] for r in casemap[cid] if r["class"] != "PASS")
                    rows.append((cid, cpu_r, acc_r, dict(classes)))
            lines.append(f"### cpu → {backend} delta ({len(eligible & set(casemap))} claim-eligible cases)")
            if not rows:
                lines.append(f"No regressions: {backend} matches cpu on every claim-eligible case.")
            else:
                lines += ["", "| case | cpu | " + backend + " | failure classes |", "|---|---|---|---|"]
                for cid, c, a, cl in rows:
                    lines.append(f"| {cid} | {fmt_pct(c)} | {fmt_pct(a)} | {cl} |")
                # evidence: the smoking-gun examples
                lines.append("")
                lines.append("**DIGIT_CORRUPT evidence (expected → got):**")
                shown = 0
                for cid, _, _, _ in rows:
                    for r in casemap[cid]:
                        for ev in r["evidence"]:
                            lines.append(f"- `{cid}` rep {r['rep']}: `{ev['expected']}` → `{ev['got']}`")
                            shown += 1
                            if shown >= 20:
                                break
                        if shown >= 20:
                            break
                    if shown >= 20:
                        break
                if shown == 0:
                    lines.append("- (none — regressions are FORMAT_DRIFT / MISSING only)")
            lines.append("")
        if capability:
            lines.append(f"_Capability-limited (cpu < {fmt_pct(CPU_GATE)}), excluded from claims: "
                         + ", ".join(capability[:12]) + ("…" if len(capability) > 12 else "") + "_")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


# ------------------------------------------------------------------ selftest

def selftest(suite):
    case = next(c for c in suite if c["payload_class"] == "amount" and c["family"] == "copy")
    value = list(case["expected"].values())[0]
    corrupt = re.sub(r"\d", lambda m: str((int(m.group(0)) + 3) % 10), value)
    drift = digits_of(value)
    checks = [
        ({"case_id": case["id"], "backend": "cpu", "rep": 1, "output": f"REFERENCE: {value}"}, "PASS"),
        ({"case_id": case["id"], "backend": "gpu", "rep": 1, "output": f"The line says {corrupt}."}, "DIGIT_CORRUPT"),
        ({"case_id": case["id"], "backend": "gpu", "rep": 2, "output": f"It is {drift} exactly."}, "FORMAT_DRIFT"),
        ({"case_id": case["id"], "backend": "gpu", "rep": 3, "output": "I cannot find it."}, "MISSING"),
        ({"case_id": case["id"], "backend": "gpu", "rep": 4, "output": "", "error": "TIMEOUT"}, "ERROR"),
        ({"case_id": case["id"], "backend": "gpu", "rep": 5, "output": "ha " * 60}, "MISSING"),
    ]
    for rec, want in checks:
        got, _, _, flags = classify_rep(case, rec)
        assert got == want, f"selftest: wanted {want}, got {got} for {rec['output'][:40]!r}"
    assert "DEGENERATE" in classify_rep(case, checks[5][0])[3]
    ecase = next(c for c in suite if c["family"] == "extract_json")
    good = json.dumps(ecase["expected"], ensure_ascii=False)
    bad = dict(ecase["expected"])
    k0 = list(bad)[0]
    bad[k0] = re.sub(r"\d", "7", bad[k0])
    got, fields, _, _ = classify_rep(ecase, {"case_id": ecase["id"], "backend": "gpu", "rep": 1, "output": good})
    assert got == "PASS", fields
    got, fields, _, _ = classify_rep(ecase, {"case_id": ecase["id"], "backend": "gpu", "rep": 1,
                                             "output": json.dumps(bad, ensure_ascii=False)})
    assert got in ("DIGIT_CORRUPT", "MISSING") and fields[k0] != "PASS", fields
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="*", help="runner output JSONL files")
    ap.add_argument("--suite", default=str(SUITE_PATH))
    ap.add_argument("--report", default=str(REPORT_DIR / "report.md"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    suite = load_jsonl(args.suite)
    if args.selftest:
        selftest(suite)
        return
    if not args.outputs:
        ap.error("no output files (or use --selftest)")
    records_by_src = {Path(p).stem: load_jsonl(p) for p in args.outputs}
    arms, unknown = score(suite, records_by_src)
    if unknown:
        print(f"WARNING: {sum(unknown.values())} records with unknown case_id", file=sys.stderr)
    cases_by_id = {c["id"]: c for c in suite}
    path = report(arms, cases_by_id, Path(args.report))
    print(f"report → {path}")
    for (runtime, model, device, backend), casemap in sorted(arms.items()):
        classes = Counter(r["class"] for v in casemap.values() for r in v)
        print(f"  {runtime}/{model}/{device}/{backend}: " +
              ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))


if __name__ == "__main__":
    main()

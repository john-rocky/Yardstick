#!/usr/bin/env python3
"""Generate the pinned backend-fidelity suite → evaldata/fidelity_suite_v1.jsonl.

The suite measures COPY FIDELITY: can a backend reproduce exact strings (dates,
amounts, IDs, UUIDs, JSON values) it was just shown — the failure class of
LiteRT-LM #2814 (iOS/Metal digit corruption, CPU clean) and #3012 (Adreno 750,
~2k-token prompts). It is NOT a quality eval; the semantic content is trivial on
purpose so the CPU baseline sits near 100% and any backend delta is attributable.

Design invariants (checked by assertions at generation time):
  * Filler prose contains NO digits — every digit in an output traces to the payload.
  * Deterministic: same --seed → byte-identical suite. The generated file is pinned
    (part of the protocol, like evaldata/gsm8k_test.jsonl); regenerate only with a
    version bump.
  * Runtime-agnostic: records carry prompt text + expectations only; runners add
    backend/model/device fields to their output records.

Record schema (one JSON object per line):
  suite            "fidelity-v1"
  id               e.g. "copy-id_digits-c2048-mid-en-01"
  tier             smoke | core | full   (smoke ⊂ core ⊂ full at run time)
  family           copy | extract_json
  payload_class    date_iso | date_text | amount | id_digits | uuid | phone |
                   order_id | version | mixed (extract_json)
  context_bucket   target prompt size in tokens (est.): 128 | 1024 | 2048 | 4096
  position         early | middle | late  (payload location in the filler)
  lang             en | de | ja  (filler language; de@2k matches #3012)
  prompt           full prompt text
  expected         {field: exact string}  — verbatim-match ground truth
  max_tokens       generation budget (NOT total context; see methodology/fidelity.md
                   for the litert-mac-verify total-context sizing rule)
  est_prompt_tokens rough size estimate (chars/CPT); runners should record actuals

    python3 scripts/gen_fidelity_suite.py            # writes evaldata/fidelity_suite_v1.jsonl
    python3 scripts/gen_fidelity_suite.py --stats    # print composition, write nothing
"""
import argparse, json, random, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evaldata" / "fidelity_suite_v1.jsonl"
SUITE = "fidelity-v1"
SEED = 20260901  # part of the protocol — bump the suite version if you change it

# chars-per-token rough estimates for filler sizing (runners record actuals)
CPT = {"en": 4.0, "de": 4.4, "ja": 1.8}

# ---------------------------------------------------------------- filler prose
# Digit-free by construction; numbers are spelled out. Neutral logistics/weather
# prose so no model refuses and no payload-like strings appear.

EN_SENTENCES = [
    "The morning shipment left the coastal warehouse before sunrise and reached the sorting hub by early afternoon.",
    "Cloud cover thickened over the harbor while the crews finished loading the last of the crates.",
    "A steady breeze moved through the loading yard as the forklifts rolled between the tall shelves.",
    "The dispatcher reviewed the route notes twice and confirmed that the northern road remained open.",
    "Rain was expected toward evening, so the pallets were wrapped and moved under the long canopy.",
    "The night crew swept the platform, stacked the empty containers, and checked the seals on every door.",
    "By midweek the backlog had cleared and the drivers returned to their usual rotation.",
    "The harbor office kept its lights on late as the manifest review stretched into the evening.",
    "Fog settled over the river crossing, and the early convoy waited for the signal to proceed.",
    "The warehouse team rotated the older stock to the front racks before the seasonal delivery arrived.",
    "A calm stretch of weather let the maintenance crew repaint the lane markings across the yard.",
    "The regional depot reported smooth operations and no delays through the middle of the month.",
    "Two of the loading bays were closed for cleaning while the others handled the reduced volume.",
    "The afternoon briefing covered the revised schedule and the new procedure for sealed cargo.",
    "Snow in the high passes slowed the mountain route, and traffic shifted to the valley road.",
    "The clerk filed the inspection notes and returned the clipboard to its hook beside the door.",
    "Sunlight broke through after the storm and the crews resumed work on the outer platform.",
    "The transfer went smoothly and the receiving team signed off without any remarks.",
]

DE_SENTENCES = [
    "Die Morgenlieferung verließ das Lager an der Küste vor Sonnenaufgang und erreichte das Verteilzentrum am frühen Nachmittag.",
    "Über dem Hafen zog dichte Bewölkung auf, während die Teams die letzten Kisten verluden.",
    "Eine stetige Brise wehte durch den Ladehof, während die Gabelstapler zwischen den hohen Regalen rollten.",
    "Der Disponent prüfte die Routenhinweise zweimal und bestätigte, dass die nördliche Straße frei blieb.",
    "Gegen Abend wurde Regen erwartet, daher wurden die Paletten verpackt und unter das lange Vordach gebracht.",
    "Die Nachtschicht fegte die Rampe, stapelte die leeren Behälter und prüfte die Siegel an jeder Tür.",
    "Zur Wochenmitte war der Rückstand abgearbeitet, und die Fahrer kehrten zu ihrem üblichen Rhythmus zurück.",
    "Das Hafenbüro ließ das Licht lange brennen, während die Prüfung der Frachtpapiere sich in den Abend zog.",
    "Nebel legte sich über die Flussquerung, und der frühe Konvoi wartete auf das Zeichen zur Weiterfahrt.",
    "Das Lagerteam stellte die ältere Ware nach vorn, bevor die saisonale Lieferung eintraf.",
    "Eine ruhige Wetterlage erlaubte es dem Wartungstrupp, die Fahrbahnmarkierungen im Hof zu erneuern.",
    "Das regionale Depot meldete einen reibungslosen Betrieb und keine Verzögerungen bis zur Monatsmitte.",
]

JA_SENTENCES = [
    "朝の便は夜明け前に沿岸の倉庫を出発し、昼過ぎには仕分け拠点に到着した。",
    "港の上空には雲が厚くなり、作業班は最後の木箱の積み込みを終えた。",
    "積み場にはおだやかな風が吹き、フォークリフトが高い棚のあいだを行き来した。",
    "配車係は経路の注意書きを二度確認し、北側の道路が通行できることを確かめた。",
    "夕方には雨が見込まれたため、荷は覆いを掛けて長い屋根の下へ移された。",
    "夜勤の班は作業台を掃き、空の容器を積み重ね、すべての扉の封を点検した。",
    "週の半ばには滞りが解消され、運転手たちはいつもの持ち回りに戻った。",
    "霧が川の渡し場に降り、早朝の車列は進行の合図を待った。",
    "倉庫の班は季節の入荷が届く前に、古い在庫を棚の手前へ移した。",
    "天候が落ち着いたため、保守の班は構内の区画線を塗り直した。",
]

FILLER = {"en": EN_SENTENCES, "de": DE_SENTENCES, "ja": JA_SENTENCES}

# ------------------------------------------------------------------- payloads

def gen_payload(cls, rng, lang="en"):
    """Return (field_name, exact_string). Digit-dense values, no 'nice' patterns."""
    if cls == "date_iso":
        y = rng.randint(2027, 2041)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        return "date", f"{y:04d}-{m:02d}-{d:02d}"
    if cls == "date_text":
        y = rng.randint(2027, 2041)
        d = rng.randint(1, 28)
        if lang == "de":
            months = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
                      "August", "September", "Oktober", "November", "Dezember"]
            return "date", f"{d}. {rng.choice(months)} {y}"
        if lang == "ja":
            return "date", f"{y}年{rng.randint(1,12)}月{d}日"
        months = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
        return "date", f"{rng.choice(months)} {d}, {y}"
    if cls == "amount":
        if lang == "de":  # EU format: dot thousands, comma decimals
            whole = rng.randint(1_000, 9_999_999)
            cents = rng.randint(0, 99)
            s = f"{whole:,}".replace(",", ".")
            return "amount", f"{s},{cents:02d} €"
        if lang == "ja":
            return "amount", f"{rng.randint(10_000, 99_999_999):,}円"
        whole = rng.randint(1_000, 9_999_999)
        cents = rng.randint(0, 99)
        return "amount", f"${whole:,}.{cents:02d}"
    if cls == "id_digits":
        # 15-digit run, first digit nonzero, no repeats-of-one-digit runs > 2
        while True:
            s = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(14))
            if not re.search(r"(\d)\1\1", s):
                return "account_id", s
    if cls == "uuid":
        hexd = "0123456789abcdef"
        parts = ["".join(rng.choice(hexd) for _ in range(n)) for n in (8, 4, 4, 4, 12)]
        return "uuid", "-".join(parts)
    if cls == "phone":
        # NANP fictional range 555-01XX — never a real number
        return "phone", f"+1-{rng.randint(201, 989)}-555-{rng.randint(100, 199):04d}"
    if cls == "order_id":
        letters = "ABCDEFGHJKMNPQRSTUVWXYZ"  # no I/L/O
        seg = lambda n: "".join(rng.choice(letters + "23456789") for _ in range(n))
        return "order_id", f"ORD-{seg(4)}-{rng.randint(10000, 99999)}-{rng.choice(letters)}"
    if cls == "version":
        return "version", f"v{rng.randint(2, 9)}.{rng.randint(0, 29)}.{rng.randint(0, 12)}-rc.{rng.randint(1, 9)}"
    raise ValueError(cls)


EXTRACT_SCHEMA = ["order_id", "date", "amount", "account_id"]
EXTRACT_CLASS_FOR = {"order_id": "order_id", "date": "date_iso",
                     "amount": "amount", "account_id": "id_digits"}

FIELD_SENTENCE = {
    "en": {
        "order_id":   "The order reference recorded for this shipment is {v}.",
        "date":       "The delivery is scheduled for {v}.",
        "amount":     "The invoiced total for the consignment is {v}.",
        "account_id": "The customer account on file is {v}.",
    },
    "de": {
        "order_id":   "Die für diese Sendung erfasste Auftragsnummer lautet {v}.",
        "date":       "Die Zustellung ist für den {v} vorgesehen.",
        "amount":     "Der in Rechnung gestellte Gesamtbetrag beläuft sich auf {v}.",
        "account_id": "Das hinterlegte Kundenkonto ist {v}.",
    },
    "ja": {
        "order_id":   "この出荷の注文番号は {v} と記録されている。",
        "date":       "配達は {v} に予定されている。",
        "amount":     "請求額の合計は {v} である。",
        "account_id": "登録されている顧客口座は {v} である。",
    },
}

COPY_INSTRUCTION = {
    "en": "The document above contains exactly one line that begins with \"REFERENCE:\". "
          "Output the text of that line after the colon, exactly as written, and nothing else.",
    "de": "Das obige Dokument enthält genau eine Zeile, die mit \"REFERENCE:\" beginnt. "
          "Gib den Text dieser Zeile nach dem Doppelpunkt exakt so aus, wie er geschrieben ist, und sonst nichts.",
    "ja": "上の文書には \"REFERENCE:\" で始まる行がちょうど一つある。その行のコロン以降の文字列を、書かれている通りに一字も変えず、それだけを出力せよ。",
}

EXTRACT_INSTRUCTION = {
    "en": "From the document above, extract the following fields and output ONLY a JSON object "
          "with these keys, each value copied exactly as it appears in the document, as a JSON string: ",
    "de": "Extrahiere aus dem obigen Dokument die folgenden Felder und gib NUR ein JSON-Objekt "
          "mit diesen Schlüsseln aus; jeder Wert exakt so wie im Dokument, als JSON-String: ",
    "ja": "上の文書から次のフィールドを抜き出し、これらのキーを持つ JSON オブジェクトだけを出力せよ。"
          "各値は文書に現れる文字列を一字も変えずに JSON 文字列として出力すること: ",
}

# ------------------------------------------------------------------ assembly

POS_FRAC = {"early": 0.10, "middle": 0.50, "late": 0.85}


def build_filler(lang, target_chars, rng):
    """Digit-free prose of ~target_chars, sentences drawn with a seeded RNG."""
    bank = FILLER[lang]
    out, chars = [], 0
    while chars < target_chars:
        s = rng.choice(bank)
        out.append(s)
        chars += len(s) + 1
    return out


def make_copy_case(cls, bucket, pos, lang, k, tier):
    cid = f"copy-{cls}-c{bucket}-{pos[:3]}-{lang}-{k:02d}"
    rng = random.Random(f"{SEED}:{cid}")
    field, value = gen_payload(cls, rng, lang)
    ref_line = f"REFERENCE: {value}"
    instr = COPY_INSTRUCTION[lang]
    target_chars = max(int(bucket * CPT[lang]) - len(ref_line) - len(instr) - 40, 60)
    sents = build_filler(lang, target_chars, rng)
    idx = min(int(len(sents) * POS_FRAC[pos]), len(sents) - 1)
    sents.insert(idx + 1, ref_line)
    doc = "\n".join(sents)
    prompt = f"{doc}\n\n{instr}"
    return {
        "suite": SUITE, "id": cid, "tier": tier, "family": "copy",
        "payload_class": cls, "context_bucket": bucket, "position": pos, "lang": lang,
        "prompt": prompt, "expected": {field: value}, "max_tokens": 64,
        "est_prompt_tokens": int(len(prompt) / CPT[lang]),
    }


def make_extract_case(bucket, lang, k, tier):
    cid = f"extract-mixed-c{bucket}-spr-{lang}-{k:02d}"
    rng = random.Random(f"{SEED}:{cid}")
    expected = {}
    field_sents = []
    for key in EXTRACT_SCHEMA:
        _, value = gen_payload(EXTRACT_CLASS_FOR[key], rng, lang)
        expected[key] = value
        field_sents.append(FIELD_SENTENCE[lang][key].format(v=value))
    instr = EXTRACT_INSTRUCTION[lang] + ", ".join(f'"{k}"' for k in EXTRACT_SCHEMA) + "."
    target_chars = max(int(bucket * CPT[lang]) - sum(len(s) for s in field_sents) - len(instr) - 40, 60)
    sents = build_filler(lang, target_chars, rng)
    # spread the field sentences through the filler at fixed fractions
    for frac, fs in zip((0.15, 0.40, 0.65, 0.88), field_sents):
        sents.insert(min(int(len(sents) * frac), len(sents) - 1) + 1, fs)
    doc = "\n".join(sents)
    prompt = f"{doc}\n\n{instr}"
    return {
        "suite": SUITE, "id": cid, "tier": tier, "family": "extract_json",
        "payload_class": "mixed", "context_bucket": bucket, "position": "spread", "lang": lang,
        "prompt": prompt, "expected": expected, "max_tokens": 160,
        "est_prompt_tokens": int(len(prompt) / CPT[lang]),
    }


def build_suite():
    cases = []
    # --- smoke: one copy case per payload class, 1k bucket, middle, en (pipeline debug)
    classes = ["date_iso", "date_text", "amount", "id_digits", "uuid", "phone", "order_id", "version"]
    for i, cls in enumerate(classes):
        cases.append(make_copy_case(cls, 1024, "middle", "en", i, "smoke"))

    # --- core: the claim set. Digit-heavy classes × context sweep (the #3012 trigger
    # variable), position sweep at 2k, extract_json sweep, German at 1k–4k (#3012 shape).
    core_classes = ["date_iso", "id_digits", "amount", "uuid"]
    for cls in core_classes:
        for bucket in (128, 1024, 2048, 4096):
            cases.append(make_copy_case(cls, bucket, "middle", "en", 10 + bucket // 128, "core"))
        for pos in ("early", "late"):
            cases.append(make_copy_case(cls, 2048, pos, "en", 30, "core"))
    for bucket in (128, 1024, 2048, 4096):
        cases.append(make_extract_case(bucket, "en", 40, "core"))
        cases.append(make_extract_case(bucket, "en", 41, "core"))
    for cls in ("id_digits", "amount", "date_text"):
        for bucket in (1024, 2048, 4096):
            cases.append(make_copy_case(cls, bucket, "middle", "de", 50, "core"))

    # --- full: remaining classes × buckets, ja arm, position sweep for extract-adjacent
    full_classes = ["date_text", "phone", "order_id", "version"]
    for cls in full_classes:
        for bucket in (128, 1024, 2048, 4096):
            cases.append(make_copy_case(cls, bucket, "middle", "en", 60 + bucket // 128, "full"))
    for cls in core_classes:
        for pos in ("early", "late"):
            for bucket in (1024, 4096):
                cases.append(make_copy_case(cls, bucket, pos, "en", 70, "full"))
    for cls in ("date_text", "amount", "id_digits"):
        for bucket in (1024, 2048):
            cases.append(make_copy_case(cls, bucket, "middle", "ja", 80, "full"))
    for bucket in (1024, 2048):
        cases.append(make_extract_case(bucket, "de", 90, "full"))
        cases.append(make_extract_case(bucket, "ja", 91, "full"))
    return cases


def validate(cases):
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in cases:
        # every digit in the prompt must come from the payload (filler is digit-free)
        stripped = c["prompt"]
        for v in c["expected"].values():
            stripped = stripped.replace(v, "")
        leaked = re.findall(r"\d", stripped)
        assert not leaked, f"{c['id']}: filler leaked digits {leaked[:5]}"
        for v in c["expected"].values():
            assert v in c["prompt"], f"{c['id']}: payload not embedded"
            assert c["prompt"].count(v) == 1, f"{c['id']}: payload appears more than once"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="print composition; do not write")
    args = ap.parse_args()
    cases = build_suite()
    validate(cases)
    from collections import Counter
    tiers = Counter(c["tier"] for c in cases)
    print(f"{len(cases)} cases — smoke {tiers['smoke']}, core {tiers['core']} "
          f"(run: smoke+core={tiers['smoke']+tiers['core']}), full {tiers['full']} "
          f"(run: all={len(cases)})")
    for key in ("family", "payload_class", "context_bucket", "lang"):
        print(f"  {key}: {dict(Counter(c[key] for c in cases))}")
    if args.stats:
        return
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

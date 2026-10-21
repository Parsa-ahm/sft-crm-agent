import argparse
import json
import re
import datetime as dt

from train import load_rows

FIELDS = ["intent", "sender_name", "sender_contact", "sender_type", "ref",
          "vehicle_year", "vehicle_make", "vehicle_model", "engine_code", "part", "part_position",
          "max_mileage", "budget", "ship_to_postal",
          "yard_name", "part_number", "mileage", "yard_cost", "warranty_days",
          "carrier", "tracking_number", "eta", "issue"]
INTS = {"vehicle_year", "max_mileage", "budget", "mileage", "yard_cost", "warranty_days"}
IDS = {"ref", "part_number", "tracking_number", "ship_to_postal", "engine_code"}
NULLS = {"", "null", "none", "n/a", "unknown"}
SYNONYMS = {
    "engine": ["engine", "motor", "long block"], "transmission": ["transmission", "trans", "tranny", "gearbox"],
    "turbocharger": ["turbo"], "ecu": ["ecu", "ecm", "computer", "control module"],
    "rear differential": ["diff"], "transfer case": ["transfer case", "t-case", "tcase"],
    "alternator": ["alternator"], "starter": ["starter"], "headlight": ["headlight", "head light", "headlamp"],
    "taillight": ["taillight", "tail light", "tail lamp", "taillamp"], "door": ["door"],
    "side mirror": ["mirror"], "front bumper": ["bumper"],
}
MAKE_ALIASES = {"mercedes": "mercedes-benz", "benz": "mercedes-benz", "merc": "mercedes-benz",
                "vw": "volkswagen", "chevy": "chevrolet", "alfa": "alfa romeo"}
# non-breaking / typographic hyphens and spaces some models emit (gpt-oss writes "CR‑V")
UNICODE = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", " ": " ", " ": " "})

def parse(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    
    try:
        obj = json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def numbers_in(text):
    out = []
    # "80k" = 80000; a unit glued to the number ("90,000km") still ends it
    for m in re.finditer(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)(?:\s?([kK])(?!\w)|(?=[kK][mM][sS]?\b)|(?!\w))", text):
        x = float(m.group(1).replace(",", ""))
        out.append(x * 1000 if m.group(2) else x)
    return out


def to_int(s):
    ns = numbers_in(s)
    return int(ns[0]) if ns else s.lower()


def norm_ids(s):
    return re.sub(r"[\s-]", "", str(s)).lower()

def date_forms(iso):
    d = dt.date.fromisoformat(iso)
    return [iso, d.strftime("%B ") + str(d.day), d.strftime("%b ") + str(d.day), d.strftime("%b. ") + str(d.day), f"{d.month}/{d.day}"]


def mentions(body, field, value):
    low = body.lower()
    if field in INTS:
        return any(abs(x - value) < 1e-6 for x in numbers_in(body))
    if field in IDS or field == "tracking_number":
        return norm_ids(value) in norm_ids(body)
    if field == "eta":
        return any(f.lower() in low for f in date_forms(value))
    if field == "part":
        return any(w in low for w in SYNONYMS[value])
    if field == "vehicle_make":
        names = [value.lower()] + [a for a, full in MAKE_ALIASES.items() if full == value.lower()]
        return any(re.search(rf"\b{re.escape(n)}\b", low) for n in names)
    return str(value).lower() in low
    

def check_draft(draft, row):
    if not isinstance(draft, dict) or not isinstance(draft.get("body"), str) or not draft["body"].strip():
        return False, ["no_draft"]
    body, gold, why = draft["body"], row["gold"], []
    if str(draft.get("to", "")).lower() != row["draft_to"]:
        why.append("wrong_recipient")
    for f in row["draft_require"]:
        if gold[f] is not None and not mentions(body, f, gold[f]):
            why.append(f"missing:{f}")
    for f in row["draft_forbid"]:
        if gold[f] is not None and mentions(body, f, gold[f]):
            why.append(f"leaked:{f}")
    if row["kind"] == "yard_offer" and "[PRICE]" not in body:
        why.append("no_price_placeholder")
    source = {round(x, 2) for x in numbers_in(row["message"])}
    if any(x > 31 and round(x, 2) not in source for x in numbers_in(body)):
        why.append("invented_number")
    return not why, why

def norm(field, v):
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in NULLS:
        return None
    if field in INTS:
        return to_int(s)
    if field in IDS:
        return re.sub(r"[\s-]", "", s).upper()
    if field == 'sender_contact':
        return s.lower() if "@" in s else re.sub(r"\D", "", s)[-10:]
    if field == "eta":
        try: 
            return dt.date.fromisoformat(s[:10]).isoformat()
        except ValueError:
            return s.lower()
    if field == "vehicle_make":
        return MAKE_ALIASES.get(s.lower(), s.lower())
    return s.lower()

def compare_fields(pred, gold):
    fields = {}
    halluc = []
    omit = []
    for f in FIELDS:
        p, g = norm(f, pred.get(f)), norm(f, gold[f])
        fields[f] = p == g
        if g is None and p is not None:
            halluc.append(f)
        if g is not None and p is None:
            omit.append(f)
    return fields, halluc, omit


def score(text, row):
    p = parse(text.translate(UNICODE))
    if p is None:
        return {"valid": False, "fields_ok": False, "action_ok": False, "draft_ok": False,
                "fields": {f: False for f in FIELDS}, "halluc": [], "omit": [], "draft_why": ["invalid_json"]}
    fields, halluc, omit = compare_fields(p, row["gold"])
    draft_ok, why = check_draft(p.get("draft"), row)
    return {"valid": True, "fields_ok": all(fields.values()), "action_ok": p.get("next_action") == row["next_action"],
            "draft_ok": draft_ok, "fields": fields, "halluc": halluc, "omit": omit, "draft_why": why}


def full_pass(s):
    return s["fields_ok"] and s["action_ok"] and s["draft_ok"]


def pct(xs):
    return f"{100 * sum(xs) / max(1, len(xs)):5.1f}%"


def report(rows, scores):
    print(f"valid JSON {pct([s['valid'] for s in scores])}  all fields {pct([s['fields_ok'] for s in scores])}  "
          f"next_action {pct([s['action_ok'] for s in scores])}  draft ok {pct([s['draft_ok'] for s in scores])}")
    print(f"FULL PASS {pct([full_pass(s) for s in scores])}   "
          f"hallucinated {pct([bool(s['halluc']) for s in scores])}   omitted {pct([bool(s['omit']) for s in scores])}")

    print("\nper field (worst first):")
    acc = {f: sum(s["fields"][f] for s in scores) / len(scores) for f in FIELDS}
    for f in sorted(FIELDS, key=acc.get)[:8]:
        h = sum(f in s["halluc"] for s in scores)
        o = sum(f in s["omit"] for s in scores)
        print(f"  {f:16s} {100 * acc[f]:5.1f}%  halluc {h:3d}  omit {o:3d}")

    print("\nper kind:")
    for kind in sorted({r["kind"] for r in rows}):
        ks = [s for r, s in zip(rows, scores) if r["kind"] == kind]
        print(f"  {kind:18s} n={len(ks):3d}  full pass {pct([full_pass(s) for s in ks])}")

def score_run(run_path, data_path):
    """Score a generate.py output file against its split. Returns (rows, texts, scores) in run order."""
    gold = {r["id"]: r for r in load_rows(data_path)}
    recs = load_rows(run_path)
    rows = [gold[r["id"]] for r in recs]
    texts = [r["text"] for r in recs]
    return rows, texts, [score(t, r) for t, r in zip(texts, rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="output of generate.py, e.g. data/runs/sft_test.jsonl")
    ap.add_argument("--data", default="data/test.jsonl")
    args = ap.parse_args()

    rows, _, scores = score_run(args.run, args.data)
    report(rows, scores)

if __name__ == "__main__":
    main()
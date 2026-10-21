import argparse
import json
import time
import re
import datetime as dt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import build_prompt, load_rows

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


def generate(model, tok, rows, batch, max_new):
    texts = []
    for i in range(0, len(rows), batch):
        prompts = [build_prompt(tok, r) for r in rows[i:i + batch]]
        enc = tok(prompts, return_tensors= 'pt', padding = True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens = max_new, do_sample=False, pad_token_id = tok.pad_token_id)

        new = out[:, enc["input_ids"].shape[1]:]
        texts += tok.batch_decode(new, skip_special_tokens=True)
        print(f"{len(texts)}/{len(rows)}", flush=True)
    return texts

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
    for m in re.finditer(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)\s?([kK])?(?![\w])", text):
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
    p = parse(text)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.data)[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side= "left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()

    t0 = time.time()
    texts = generate(model, tok, rows, args.batch, args.max_new)
    scores = [score(t, r) for t, r in zip(texts, rows)]
    report(rows, scores)

    if args.out:
        with open(args.out, "w") as f:
            for r, t, s in zip(rows, texts, scores):
                f.write(json.dumps({"id": r["id"], "kind": r["kind"], "text": t, "score": s}) + "\n")
        print("wrote", args.out)

if __name__ == "__main__":
    main()
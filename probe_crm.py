"""Headroom probe for the broker task: message -> CRM JSON + next_action + draft.

Greedy = what the base model does today. K samples at T=1 = what GRPO would see: a case only
gives gradient when some samples pass and some fail (0 < pass rate < 1).

The scorer here is a PROBE scorer, not the reward. The reward is Parsa's to write.

    ../rl-lisp-python/.venv/bin/python probe_crm.py --model <path> --k 8 --out data/probe_crm_1.5B.jsonl
"""
import argparse
import collections
import datetime as dt
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_nim import SYNONYMS, norm_id, numbers_in
from records import INTENTS, ISSUES, NEXT_ACTION, PARTS, POSITIONS, LIGHT_POS

FIELDS = ["intent", "sender_name", "sender_contact", "sender_type", "ref",
          "vehicle_year", "vehicle_make", "vehicle_model", "engine_code", "part", "part_position",
          "max_mileage", "budget", "ship_to_postal",
          "yard_name", "part_number", "mileage", "yard_cost", "warranty_days",
          "carrier", "tracking_number", "eta", "issue"]
INTS = {"vehicle_year", "max_mileage", "budget", "mileage", "yard_cost", "warranty_days"}
IDS = {"ref", "part_number", "tracking_number", "ship_to_postal", "engine_code"}

SYSTEM = f"""You are the inbox assistant for a one-person used auto parts broker in Ontario, Canada.
He sources used parts from junkyards and ships them to customers. Messages come from customers,
junkyards ("yards") and freight carriers. Read one inbound message and return ONLY a JSON object
with exactly these keys:
{json.dumps(FIELDS + ["next_action", "draft"])}

Fields (use null for anything the message does not state; never guess or invent values):
- intent: one of {json.dumps(INTENTS)}
- sender_name: the sender's full name as written, or null. For a yard or carrier, null.
- sender_contact: the sender's email or phone number from the message header.
- sender_type: one of "individual", "shop", "yard", "carrier".
- ref: the order or request number (like JA-1234), or null.
- vehicle_year (integer), vehicle_make, vehicle_model, engine_code.
- part: one of {json.dumps(list(PARTS))}
- part_position: one of {json.dumps(POSITIONS + LIGHT_POS)}, or null.
- max_mileage: the customer's maximum acceptable mileage, integer (62k -> 62000).
- budget: the customer's budget in dollars, integer.
- ship_to_postal: Canadian postal code to ship to.
- yard_name, part_number, mileage (integer, mileage of the yard's part), yard_cost (integer,
  the yard's price to the broker), warranty_days (integer).
- carrier, tracking_number (the PRO number), eta (estimated delivery, YYYY-MM-DD; use the
  message date to work out the year).
- issue: for a warranty claim, one of {json.dumps(ISSUES)}, else null.

next_action: ONE string, picked by intent: {"; ".join(f"{k} -> \"{v}\"" for k, v in NEXT_ACTION.items())}

draft: {{"to": "yard" or "customer", "body": "..."}}, a short reply the broker approves before sending.
- quote_request: write to the yard. Ask for the part: vehicle, part, position, engine code and
  max mileage if given. Ask for part number, mileage and price. Never include the customer's
  budget or contact details.
- yard_offer: write to the customer. Offer the part with part number, mileage and warranty.
  Write the price as [PRICE] (the broker sets it). Never include the yard's cost or the yard's name.
- freight_update: write to the customer with the carrier, PRO number and delivery date.
- where_is_my_part, warranty_claim, cancel_request: reply to the customer, mention their order
  number if given.
- Never write a number, price, date or ID that is not in the message."""


def norm(field, v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("null", "none", "n/a", "unknown"):
        return None
    if field in INTS:
        ns = numbers_in(s)
        return int(ns[0]) if ns else s.lower()
    if field in IDS:
        return norm_id(s).upper()
    if field == "sender_contact":
        if "@" in s:
            return s.lower()
        d = re.sub(r"\D", "", s)
        return d[-10:]
    if field == "eta":
        try:
            return dt.date.fromisoformat(s[:10]).isoformat()
        except ValueError:
            return s.lower()
    return s.lower()


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def date_forms(iso):
    d = dt.date.fromisoformat(iso)
    return [iso, d.strftime("%B ") + str(d.day), d.strftime("%b ") + str(d.day),
            d.strftime("%b. ") + str(d.day), f"{d.month}/{d.day}"]


def mentions(body, field, value):
    low = body.lower()
    if field in INTS:
        return any(abs(x - value) < 1e-6 for x in numbers_in(body))
    if field in IDS or field == "tracking_number":
        return norm_id(value) in norm_id(body)
    if field == "eta":
        return any(f.lower() in low for f in date_forms(value))
    if field == "part":
        return any(w in low for w in SYNONYMS[value])
    return str(value).lower() in low


def draft_check(draft, row):
    """Probe-level draft check. Returns (ok, reasons)."""
    if not isinstance(draft, dict) or not isinstance(draft.get("body"), str) or not draft["body"].strip():
        return False, ["no_draft"]
    body, g, why = draft["body"], row["gold"], []
    if str(draft.get("to", "")).lower() != row["draft_to"]:
        why.append("wrong_recipient")
    for f in row["draft_require"]:
        if g[f] is not None and not mentions(body, f, g[f]):
            why.append(f"missing:{f}")
    for f in row["draft_forbid"]:
        if g[f] is not None and mentions(body, f, g[f]):
            why.append(f"leaked:{f}")
    if row["kind"] == "yard_offer" and "[PRICE]" not in body:
        why.append("no_price_placeholder")
    src = {round(x, 2) for x in numbers_in(row["message"])}
    if any(x > 31 and round(x, 2) not in src for x in numbers_in(body)):
        why.append("invented_number")
    return not why, why


def score(pred, row):
    if pred is None:
        return {"valid": False, "fields_ok": False, "action_ok": False, "draft_ok": False,
                "fields": {f: False for f in FIELDS}, "halluc": [], "omit": [], "draft_why": ["invalid_json"]}
    fields, halluc, omit = {}, [], []
    for f in FIELDS:
        p, g = norm(f, pred.get(f)), norm(f, row["gold"][f])
        fields[f] = p == g
        if g is None and p is not None:
            halluc.append(f)
        if g is not None and p is None:
            omit.append(f)
    d_ok, d_why = draft_check(pred.get("draft"), row)
    return {"valid": True, "fields_ok": all(fields.values()),
            "action_ok": pred.get("next_action") == row["next_action"], "draft_ok": d_ok,
            "fields": fields, "halluc": halluc, "omit": omit, "draft_why": d_why}


def passes(s, level):
    if level == "extract":
        return s["fields_ok"] and s["action_ok"]
    return s["fields_ok"] and s["action_ok"] and s["draft_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", default="data/final/val.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.cases)][: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to("cuda").eval()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["message"]}],
        tokenize=False, add_generation_prompt=True) for r in rows]
    plen = sorted(len(tok(p)["input_ids"]) for p in prompts)
    tlen = sorted(len(tok(json.dumps(r["target"]))["input_ids"]) for r in rows)
    print(f"prompt tokens p50={plen[len(plen)//2]} max={plen[-1]}   target tokens p50={tlen[len(tlen)//2]} "
          f"max={tlen[-1]}", flush=True)

    def run(batch_prompts, sample, n):
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=sample,
                                 temperature=1.0 if sample else None, top_p=None, top_k=None,
                                 num_return_sequences=n, pad_token_id=tok.eos_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    greedy, samples = [], []
    for i in range(0, len(rows), a.batch):
        greedy += run(prompts[i:i + a.batch], False, 1)
    print(f"greedy done {(time.time()-t0)/60:.1f} min", flush=True)
    sb = max(1, a.batch // a.k)
    for i in range(0, len(rows), sb):
        outs = run(prompts[i:i + sb], True, a.k)
        samples += [outs[j * a.k:(j + 1) * a.k] for j in range(len(outs) // a.k)]
    secs = time.time() - t0
    print(f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)

    res = []
    for r, g, ss in zip(rows, greedy, samples):
        gs = score(parse(g), r)
        sc = [score(parse(s), r) for s in ss]
        res.append({"id": r["id"], "kind": r["kind"], "channel": r["channel"], "greedy_text": g, "greedy": gs,
                    "pr_extract": sum(passes(s, "extract") for s in sc) / len(sc),
                    "pr_full": sum(passes(s, "full") for s in sc) / len(sc)})

    n = len(res)
    pct = lambda xs: f"{sum(xs)/max(len(xs),1):.1%}"
    G = [x["greedy"] for x in res]
    print(f"\n=== {a.model}  n={n}  k={a.k}  ({secs/60:.1f} min) ===")
    print(f"greedy: valid JSON {pct([s['valid'] for s in G])}   all 23 fields {pct([s['fields_ok'] for s in G])}   "
          f"next_action {pct([s['action_ok'] for s in G])}   draft ok {pct([s['draft_ok'] for s in G])}")
    print(f"greedy pass: extract (fields+action) {pct([passes(s,'extract') for s in G])}   "
          f"full (+draft) {pct([passes(s,'full') for s in G])}")
    print(f"cases w/ hallucinated field {pct([bool(s['halluc']) for s in G])}   "
          f"w/ omitted field {pct([bool(s['omit']) for s in G])}")
    print("per-field greedy accuracy (worst first):")
    acc = {f: sum(s["fields"][f] for s in G) / n for f in FIELDS}
    for f in sorted(FIELDS, key=acc.get)[:12]:
        hc = sum(f in s["halluc"] for s in G)
        oc = sum(f in s["omit"] for s in G)
        print(f"  {f:16s} {acc[f]:6.1%}   halluc {hc:3d}  omit {oc:3d}")
    dw = collections.Counter(re.sub(r":.*", "", w) + (":" + w.split(":")[1] if ":" in w else "")
                             for s in G for w in s["draft_why"])
    print("draft failure reasons (greedy):", dict(dw.most_common(10)))
    for lvl in ("extract", "full"):
        pr = [x[f"pr_{lvl}"] for x in res]
        print(f"sampled T=1 k={a.k} [{lvl}]: mean pass {sum(pr)/n:.1%}   all-wrong {pct([p == 0 for p in pr])}   "
              f"all-right {pct([p == 1 for p in pr])}   MIXED {pct([0 < p < 1 for p in pr])}")
    print("by kind: n | greedy extract | greedy full | mixed extract | mixed full")
    by = collections.defaultdict(list)
    for x in res:
        by[x["kind"]].append(x)
    for k, xs in sorted(by.items()):
        print(f"  {k:18s} {len(xs):3d}  {pct([passes(x['greedy'],'extract') for x in xs]):>6}  "
              f"{pct([passes(x['greedy'],'full') for x in xs]):>6}  {pct([0 < x['pr_extract'] < 1 for x in xs]):>6}  "
              f"{pct([0 < x['pr_full'] < 1 for x in xs]):>6}")
    if a.out:
        with open(a.out, "w") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()

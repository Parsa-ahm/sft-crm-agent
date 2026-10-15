"""Headroom probe: how well does a base Qwen extract CRM updates, and is there GRPO signal?

Greedy pass = what the model does today. K samples at T=1 = what GRPO would see: a case only
produces gradient when some samples are right and some are wrong (0 < pass rate < 1).

    .venv/bin/python probe.py --model Qwen/Qwen2.5-1.5B-Instruct --k 8
"""
import argparse
import collections
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen import FIELDS, INTENTS

SYSTEM = f"""You update a store's CRM from inbound emails and phone call transcripts.
Return ONLY a JSON object with exactly these keys:
{json.dumps(FIELDS)}

Rules:
- intent: one of {json.dumps(INTENTS)}.
- status: one of "shipped", "delivered", "delayed", "canceled", or null. Only set it when the
  carrier or store states it as fact. A customer complaining or asking is not a status.
- estimated_delivery: YYYY-MM-DD. Dates written as dd/mm/yyyy are day first.
- new_* fields: only for an address change. new_zip in the form 12345-678.
- customer_email: the customer's own email address, or null if not given.
- Use null for anything the message does not state. Never guess or invent values.
- If the message corrects a value, use the corrected value."""


def norm(field, v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("null", "none", "n/a"):
        return None
    if field in ("tracking_number", "order_id"):
        return re.sub(r"[\s-]", "", v).upper()
    if field == "new_zip":
        d = re.sub(r"\D", "", v)
        return f"{d[:5]}-{d[5:]}" if len(d) == 8 else d
    return v.lower()


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score(pred, gold):
    if pred is None:
        return {"valid": False, "exact": False, "fields": {f: False for f in FIELDS},
                "halluc": 0, "omit": 0}
    fields, halluc, omit = {}, 0, 0
    for f in FIELDS:
        p, g = norm(f, pred.get(f)), norm(f, gold[f])
        fields[f] = p == g
        if g is None and p is not None:
            halluc += 1
        if g is not None and p is None:
            omit += 1
    return {"valid": True, "exact": all(fields.values()), "fields": fields,
            "halluc": halluc, "omit": omit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.cases)]
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to("cuda").eval()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": c["message"]}],
        tokenize=False, add_generation_prompt=True) for c in cases]

    def run(batch_prompts, sample, n):
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=sample,
                                 temperature=1.0 if sample else None, top_p=None, top_k=None,
                                 num_return_sequences=n, pad_token_id=tok.eos_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    greedy, samples = [], []
    for i in range(0, len(cases), a.batch):
        greedy += run(prompts[i:i + a.batch], False, 1)
    sb = max(1, a.batch // a.k)
    for i in range(0, len(cases), sb):
        outs = run(prompts[i:i + sb], True, a.k)
        samples += [outs[j * a.k:(j + 1) * a.k] for j in range(len(outs) // a.k)]
    secs = time.time() - t0

    rows = []
    for c, g, ss in zip(cases, greedy, samples):
        gs = score(parse(g), c["gold"])
        sp = [score(parse(s), c["gold"])["exact"] for s in ss]
        rows.append({"id": c["id"], "kind": c["kind"], "tags": c["tags"], "greedy_text": g,
                     "greedy": gs, "pass_rate": sum(sp) / len(sp)})

    n = len(rows)
    print(f"\n=== {a.model}  n={n}  k={a.k}  ({secs/60:.1f} min) ===")
    print(f"greedy: valid JSON {sum(r['greedy']['valid'] for r in rows)/n:.1%}   "
          f"all-fields exact {sum(r['greedy']['exact'] for r in rows)/n:.1%}   "
          f"cases w/ hallucinated field {sum(r['greedy']['halluc'] > 0 for r in rows)/n:.1%}   "
          f"cases w/ omitted field {sum(r['greedy']['omit'] > 0 for r in rows)/n:.1%}")
    print("per-field greedy accuracy:")
    for f in FIELDS:
        print(f"  {f:20s} {sum(r['greedy']['fields'][f] for r in rows)/n:.1%}")
    pr = [r["pass_rate"] for r in rows]
    mixed = sum(0 < p < 1 for p in pr)
    print(f"sampled (T=1, k={a.k}): mean pass {sum(pr)/n:.1%}   "
          f"all-wrong {sum(p == 0 for p in pr)/n:.1%}   all-right {sum(p == 1 for p in pr)/n:.1%}   "
          f"MIXED (GRPO signal) {mixed/n:.1%}")
    print("by kind (greedy exact | sampled mean pass | mixed):")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for k, rs in sorted(by.items()):
        print(f"  {k:22s} n={len(rs):3d}  {sum(r['greedy']['exact'] for r in rs)/len(rs):6.1%}  "
              f"{sum(r['pass_rate'] for r in rs)/len(rs):6.1%}  {sum(0 < r['pass_rate'] < 1 for r in rs)/len(rs):6.1%}")
    tagc = collections.defaultdict(list)
    for r in rows:
        for t in r["tags"]:
            tagc[t].append(r)
    print("by trap tag (greedy exact):")
    for t, rs in sorted(tagc.items()):
        print(f"  {t:22s} n={len(rs):3d}  {sum(r['greedy']['exact'] for r in rs)/len(rs):6.1%}")
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

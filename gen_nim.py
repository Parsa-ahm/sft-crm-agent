"""Write inbound messages for gold records with a NIM model, verify each one, append to JSONL.

Resumable: ids already in --out are skipped. Rejected messages are kept (accepted=false, with
reasons) so the QC report can show what the writer gets wrong.

    .venv/bin/python gen_nim.py --model google/gemma-4-31b-it --n 3300 --seed 0 --out data/train_raw.jsonl
    .venv/bin/python gen_nim.py --model moonshotai/kimi-k3 --n 330 --seed 999 --out data/test_raw.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from records import build_all

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEY = open(os.path.expanduser("~/.config/nim/api-key")).read().strip()

SYSTEM = """You write realistic inbound messages received by a one-person used auto parts broker in
Ontario, Canada. He sources used engines, transmissions and body parts from junkyards and ships
them to customers. Write ONLY the message itself, no commentary, no quotes around it.
For an email, the first line is "Subject: ..." then a blank line then the body.
For an SMS, write only the text body."""

SYNONYMS = {
    "engine": ["engine", "motor", "long block"], "transmission": ["transmission", "trans", "tranny", "gearbox"],
    "turbocharger": ["turbo"], "ecu": ["ecu", "ecm", "computer", "control module"],
    "rear differential": ["diff"], "transfer case": ["transfer case", "t-case", "tcase"],
    "alternator": ["alternator"], "starter": ["starter"], "headlight": ["headlight", "head light", "headlamp"],
    "taillight": ["taillight", "tail light", "tail lamp", "taillamp"], "door": ["door"],
    "side mirror": ["mirror"], "front bumper": ["bumper"],
}
MAKE_ALIASES = {"Mercedes-Benz": ["mercedes", "benz", "merc"], "Volkswagen": ["volkswagen", "vw"],
                "Chevrolet": ["chevrolet", "chevy"], "Land Rover": ["land rover", "range rover"],
                "Alfa Romeo": ["alfa"]}
ISSUE_WORDS = {"noise": ["nois", "knock", "tick", "rattl", "grind", "clunk"],
               "leak": ["leak", "drip", "puddle"], "no_start": ["start", "crank", "turn over", "dead"],
               "slipping": ["slip", "shift", "gear", "rpm", "flare"],
               "wrong_part": ["wrong", "fit", "not the right", "doesn't match", "different"],
               "damaged": ["damag", "crack", "broken", "dent", "bent", "smash"]}


def norm_id(s):
    return re.sub(r"[\s\-]", "", str(s)).lower()


def numbers_in(text):
    """Every number in the text, with '62k' / '1.4k' / '$1,400' expanded."""
    out = []
    for m in re.finditer(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)\s?([kK])?(?![\w])", text):
        try:
            x = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append(x * 1000 if m.group(2) else x)
    return out


def verify(rec, text):
    """Reasons this message does not faithfully carry its gold record. Empty list = accept."""
    g, reasons = rec["gold"], []
    low, nid = text.lower(), norm_id(text)
    if len(text) < 25:
        return ["too_short"]
    if len(text) > 2500:
        reasons.append("too_long")
    if re.search(r"(?i)\b(here is|here's) (the|a|your) (message|email|text)\b|as an ai", text):
        reasons.append("meta_commentary")
    stated = {f for f, _, _ in rec["facts"]}
    nums = numbers_in(text)
    for f in ("max_miles", "budget", "miles", "yard_cost", "warranty_days", "vehicle_year"):
        if g[f] is not None and (f in stated or f == "vehicle_year" and "vehicle" in stated):
            if not any(abs(x - g[f]) < 1e-6 for x in nums):
                reasons.append(f"missing:{f}")
    for f in ("ref", "part_number", "tracking_number", "ship_to_postal", "engine_code"):
        if g[f] is not None and f in stated and norm_id(g[f]) not in nid:
            reasons.append(f"missing:{f}")
    if g["vehicle_make"] and "vehicle_make" in stated and not any(m in low for m in MAKE_ALIASES.get(g["vehicle_make"], [g["vehicle_make"].lower()])):
        reasons.append("missing:vehicle_make")
    for f in ("vehicle_model", "carrier", "yard_name"):
        if g[f] is not None and f in stated and g[f].lower() not in low:
            reasons.append(f"missing:{f}")
    if g["sender_name"] and "sender_name" in stated and g["sender_name"].split()[0].lower() not in low:
        reasons.append("missing:sender_name")
    if g["vehicle_make"] and "vehicle" in stated and not any(m in low for m in MAKE_ALIASES.get(g["vehicle_make"], [g["vehicle_make"].lower()])):
        reasons.append("missing:vehicle_make")
    if g["part"] and not any(w in low for w in SYNONYMS[g["part"]]):
        reasons.append("missing:part")
    if g["part_position"]:
        side = g["part_position"].split()[-1]
        if side not in low and {"left": "driver", "right": "passenger"}[side] not in low:
            reasons.append("missing:part_position")
    if g["eta"]:
        et = next(v for f, v, _ in rec["facts"] if f == "eta")
        if et.lower() not in low:
            reasons.append("missing:eta")
    if g["issue"] and not any(w in low for w in ISSUE_WORDS[g["issue"]]):
        reasons.append("missing:issue")
    # leaks: the writer must not invent values the gold record says are absent
    if g["engine_code"] is None and re.search(r"\b[A-Z]{1,3}\d{2,3}[A-Z]{0,3}\d{0,2}\b", text) and \
            rec["kind"] in ("quote_request", "yard_offer"):
        m = re.search(r"\b[A-Z]{1,3}\d{2,3}[A-Z]{0,3}\d{0,2}\b", text).group(0)
        if norm_id(m) not in norm_id(g["part_number"] or "") and norm_id(m) not in norm_id(g["vehicle_model"] or "") \
                and norm_id(m) not in norm_id(g["ship_to_postal"] or ""):
            reasons.append(f"leak:engine_code?{m}")
    if g["sender_name"] is None and rec["channel"] == "sms" and g["sender_type"] in ("shop", "individual"):
        from records import FIRST
        named = [n for n in FIRST if re.search(r"\b%s\b" % n, text, re.I)]
        if named:
            reasons.append(f"leak:name?{named[0]}")
    explained = {g[f] for f in ("max_miles", "budget", "miles", "yard_cost", "warranty_days", "vehicle_year")
                 if g[f] is not None}
    scrub = text
    for f in ("ref", "part_number", "tracking_number", "ship_to_postal", "engine_code", "sender_contact",
              "vehicle_model", "eta"):
        if g[f]:
            scrub = re.sub(re.escape(str(g[f])), " ", scrub, flags=re.I)
    if g["eta"]:
        scrub = scrub.replace(next(v for f, v, _ in rec["facts"] if f == "eta"), " ")
    for x in numbers_in(scrub):
        if x > 31 and not any(abs(x - e) < 1e-6 for e in explained):
            reasons.append(f"extra_number:{x:g}")
            break
    return reasons


def prompt_for(rec):
    lines = [f"Channel: {rec['channel']}", f"Sender style: {rec['persona']}", f"Message date: {rec['date']}",
             "", f"Situation: {SITUATION[rec['kind']]}", "", "Include ALL of these facts:"]
    for f, v, how in rec["facts"]:
        lines.append(f"- {how}: {v}")
    lines += ["", "Rules:",
              "- Identifiers (order/request numbers, part numbers, PRO numbers, postal codes, engine codes, names) "
              "must appear exactly as given.",
              "- Mileage and dollar amounts must be written with digits (e.g. 62k, 62,000, $1,400, 1400).",
              "- Do NOT include any other specific numbers, prices, dates, phone numbers, emails, codes, "
              "part numbers or order numbers beyond the facts above. Years other than the vehicle year are not allowed."]
    if rec["avoid"]:
        lines.append("- Do NOT mention " + "; ".join(rec["avoid"]) + ".")
    lines.append("- You may add small talk or backstory that contains no numbers.")
    return "\n".join(lines)


SITUATION = {
    "quote_request": "A customer asks the broker to find a used part and quote it.",
    "yard_offer": "A junkyard answers the broker's earlier request: they have the part, here are the details.",
    "freight_update": "A freight carrier notifies the broker that a shipment (a part going from a junkyard "
                      "to the broker's customer) was picked up and gives tracking and delivery estimate.",
    "where_is_my_part": "A customer who already paid asks where their part is and why it is taking so long.",
    "warranty_claim": "A customer reports a problem with a used part they bought and installed.",
    "cancel_request": "A customer asks to cancel their order.",
}


def call(model, user, temperature, retries=6):
    body = json.dumps({"model": model, "temperature": temperature, "top_p": 0.95, "max_tokens": 700,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    for a in range(retries):
        req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}",
                                                 "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"].strip(), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529):
                time.sleep(min(60, 2 ** a + random.random() * 3))
                continue
            return None, f"http_{e.code}"
        except Exception as e:  # timeouts, connection resets
            time.sleep(min(60, 2 ** a + random.random() * 3))
            err = type(e).__name__
    return None, f"gave_up:{err if 'err' in dir() else 'retries'}"


def strip_think(t):
    return re.sub(r"(?s)<think>.*?</think>", "", t).strip().strip('"').strip()


def render(rec, body):
    if rec["channel"] == "sms":
        return f"[{rec['header']['from']}, {rec['date']}]\n{body}"
    return f"From: {rec['header']['from']}\nDate: {rec['date']}\n{body}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None, help="only do the first N ids (smoke test)")
    a = ap.parse_args()

    recs = build_all(a.n, a.seed)[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go, model={a.model}", flush=True)
    lock = threading.Lock()
    stats = {"ok": 0, "rej": 0, "err": 0}
    t0 = time.time()

    def work(rec):
        text, err = call(a.model, prompt_for(rec), a.temperature)
        if text is not None:
            text = strip_think(text)
        reasons = [err] if err else verify(rec, text)
        row = dict(rec, generator=a.model, body=text,
                   message=render(rec, text) if text else None,
                   accepted=not reasons, reasons=reasons)
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            stats["err" if err else "ok" if not reasons else "rej"] += 1
            n = sum(stats.values())
            if n % 25 == 0:
                rate = n / (time.time() - t0) * 60
                print(f"[{n}/{len(todo)}] ok={stats['ok']} rej={stats['rej']} err={stats['err']} "
                      f"{rate:.1f}/min", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print(f"DONE ok={stats['ok']} rej={stats['rej']} err={stats['err']} in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())

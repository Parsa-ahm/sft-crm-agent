"""Raw generations -> final splits.

Re-verifies every row with the current verifier, renames miles -> mileage (Canada writes km),
attaches a reference draft (templated from the gold record, so it passes the draft checks by
construction) for SFT, and writes:

    data/final/sft.jsonl   300  Gemma   SFT warm-up (message -> full target JSON)
    data/final/val.jsonl   200  Gemma   checkpoint selection during RL
    data/final/rl.jsonl    rest Gemma   GRPO prompts
    data/final/test.jsonl  all  Kimi    held-out, never trained or tuned on

    .venv/bin/python finalize.py
"""
import collections
import datetime as dt
import json
import os
import random

from gen_nim import verify

RENAME = {"max_miles": "max_mileage", "miles": "mileage"}


def rename(d):
    return {RENAME.get(k, k): v for k, v in d.items()}


def pretty_date(iso):
    d = dt.date.fromisoformat(iso)
    return d.strftime("%B ") + str(d.day)


def draft_for(row):
    g, k = rename(row["gold"]), row["kind"]
    first = g["sender_name"].split()[0] if g["sender_name"] else None
    hi = f"Hi {first}," if first else "Hi,"
    part = f"{g['part_position']} {g['part']}" if g.get("part_position") else g["part"]
    a = "an" if part and part[0] in "aeiou" else "a"
    veh = " ".join(str(x) for x in (g["vehicle_year"], g["vehicle_make"], g["vehicle_model"]) if x)
    if k == "quote_request":
        bits = [f"Hi, I'm looking for {a} {part} for a {veh}"]
        if g["engine_code"]:
            bits.append(f" (engine code {g['engine_code']})")
        if g["max_mileage"]:
            bits.append(f", {g['max_mileage']:,} or less on the odometer")
        bits.append(". Do you have one? Please send the part number, mileage and your price. Thanks.")
        body = "".join(bits)
    elif k == "yard_offer":
        body = (f"Hi, good news: we found {a} {part} from a {veh}, part number {g['part_number']}, "
                f"with {g['mileage']:,} on it")
        if g["warranty_days"]:
            body += f" and a {g['warranty_days']}-day warranty"
        body += ". Price: [PRICE]. Let us know if you'd like to go ahead and we'll arrange shipping."
    elif k == "freight_update":
        body = (f"Hi, your part has been picked up by {g['carrier']}. PRO number {g['tracking_number']}, "
                f"estimated delivery {pretty_date(g['eta'])}.")
    elif k == "where_is_my_part":
        ref = f" on order {g['ref']}" if g["ref"] else ""
        body = (f"{hi} thanks for checking in{ref}. I'm confirming the status with the carrier now "
                f"and will get back to you shortly with an update.")
    elif k == "warranty_claim":
        ref = f" (order {g['ref']})" if g["ref"] else ""
        body = (f"{hi} sorry to hear about the problem with the {g['part']}{ref}. I'm opening a warranty "
                f"claim with the supplier. Could you send a few photos and a short description of what "
                f"you're seeing?")
    else:  # cancel_request
        ref = f"order {g['ref']}" if g["ref"] else "your order"
        body = f"{hi} I've received your request to cancel {ref}. I'll confirm the cancellation shortly."
    return {"to": row["draft_to"], "body": body}


def target(row):
    out = rename(row["gold"])
    out["next_action"] = row["next_action"]
    out["draft"] = draft_for(row)
    return out


def finalize(path):
    rows = [json.loads(l) for l in open(path)]
    kept, why = [], collections.Counter()
    for r in rows:
        if not r["body"]:
            why["no_body"] += 1
            continue
        reasons = verify(r, r["body"])
        if reasons:
            why.update(x.split(":")[0] + ":" + x.split(":")[1].split("?")[0] if ":" in x else x for x in reasons)
            continue
        kept.append({"id": r["id"], "kind": r["kind"], "channel": r["channel"], "generator": r["generator"],
                     "message": r["message"], "gold": rename(r["gold"]), "next_action": r["next_action"],
                     "draft_to": r["draft_to"],
                     "draft_require": [RENAME.get(x, x) for x in r["draft_require"]],
                     "draft_forbid": [RENAME.get(x, x) for x in r["draft_forbid"]],
                     "target": target(r)})
    print(f"{path}: {len(rows)} raw -> {len(kept)} kept; dropped: {dict(why)}")
    return kept


def write(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path}: {len(rows)}  {dict(collections.Counter(r['kind'] for r in rows))}")


if __name__ == "__main__":
    os.makedirs("data/final", exist_ok=True)
    train = finalize("data/train_raw.jsonl")
    test = finalize("data/test_raw.jsonl")
    random.Random(0).shuffle(train)
    write(train[:300], "data/final/sft.jsonl")
    write(train[300:500], "data/final/val.jsonl")
    write(train[500:], "data/final/rl.jsonl")
    write(test, "data/final/test.jsonl")

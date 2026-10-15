"""Gold records for the parts-broker task. Python decides the truth; an LLM only writes the message.

A record = what an inbound message (customer / junkyard / freight carrier) actually says, plus what
the broker should do next and what the drafted reply must and must not contain.

Vehicles + engine codes come from a public parts-fitment table (`vehicle-overrides.json`,
built by us for their site). Everything else (people, yards, part numbers, prices, miles, refs,
tracking numbers) is synthetic.
"""
import datetime as dt
import json
import os
import random
import re

FIELDS = ["intent", "sender_name", "sender_contact", "sender_type", "ref",
          "vehicle_year", "vehicle_make", "vehicle_model", "engine_code", "part", "part_position",
          "max_miles", "budget", "ship_to_postal",
          "yard_name", "part_number", "miles", "yard_cost", "warranty_days",
          "carrier", "tracking_number", "eta", "issue"]
INTENTS = ["quote_request", "yard_offer", "freight_update", "where_is_my_part", "warranty_claim",
           "cancel_request"]
NEXT_ACTION = {"quote_request": "ask_yard", "yard_offer": "quote_customer",
               "freight_update": "update_customer", "where_is_my_part": "reply_customer",
               "warranty_claim": "reply_customer", "cancel_request": "reply_customer"}
ISSUES = ["noise", "leak", "no_start", "slipping", "wrong_part", "damaged"]
BODY = {"headlight", "taillight", "door", "side mirror", "front bumper"}
ISSUES_FOR = {"engine": ["noise", "leak", "no_start", "wrong_part", "damaged"],
              "transmission": ["noise", "leak", "slipping", "wrong_part", "damaged"],
              "turbocharger": ["noise", "leak", "wrong_part", "damaged"],
              "ecu": ["no_start", "wrong_part", "damaged"],
              "rear differential": ["noise", "leak", "wrong_part", "damaged"],
              "transfer case": ["noise", "leak", "wrong_part", "damaged"],
              "alternator": ["noise", "no_start", "wrong_part", "damaged"],
              "starter": ["noise", "no_start", "wrong_part", "damaged"]}
EV = {"Ioniq 5", "Ioniq 6", "Kona Electric", "EV6", "EV9", "Leaf", "Bolt", "Mach-E", "Taycan", "e-tron",
      "i4", "iX", "Model 3", "Model Y"}

PARTS = {  # canonical -> needs a position?
    "engine": False, "transmission": False, "turbocharger": False, "ecu": False,
    "rear differential": False, "transfer case": False, "alternator": False, "starter": False,
    "headlight": True, "taillight": True, "door": True, "side mirror": True, "front bumper": False,
}
POSITIONS = ["front left", "front right", "rear left", "rear right"]
LIGHT_POS = ["left", "right"]
CARRIERS = ["Day & Ross", "Purolator Freight", "Manitoulin Transport", "TST Overland", "Vitran"]
YARD_WORDS = ["Auto Recyclers", "Salvage", "Used Auto Parts", "Auto Wreckers", "Pick-n-Pull",
              "Parts & Salvage", "Auto Dismantlers"]
TOWNS = ["Kingston", "Barrie", "Sudbury", "Oshawa", "Peterborough", "Guelph", "Brantford",
         "Belleville", "Cornwall", "Woodstock", "Orillia", "Timmins", "Sarnia", "Welland"]
SHOP_WORDS = ["Auto", "Motors", "Garage", "Auto Service", "Performance", "Tire & Auto", "Mechanical"]
FIRST = ["Mike", "Dave", "Sarah", "Raj", "Priya", "Tom", "Kevin", "Jason", "Amanda", "Chris", "Luis",
         "Hassan", "Wei", "Nina", "Jordan", "Steve", "Ali", "Marc", "Julie", "Darnell", "Kenji",
         "Olga", "Brandon", "Fatima", "Tyler", "Gurpreet", "Emily", "Andre", "Sam", "Rob"]
LAST = ["Patel", "Nguyen", "Smith", "Tremblay", "Singh", "Brown", "Wong", "Martin", "Roy", "Khan",
        "Lee", "Gagnon", "Wilson", "Chen", "Campbell", "Ali", "Taylor", "Morin", "Clarke", "Silva"]
AREA = ["416", "647", "905", "289", "613", "705", "519", "226", "437", "343"]
EMAIL_DOM = ["gmail.com", "outlook.com", "hotmail.com", "yahoo.ca", "icloud.com", "rogers.com"]

JA_DIR = os.path.expanduser(os.environ.get(
    "JA_DIR", "data"))


def load_vehicles():
    recs = json.load(open(f"{JA_DIR}/data/vehicle-overrides.json"))["records"]
    out = []
    for r in recs:
        code = r["engineCode"].strip()
        clean = "/" not in code and "(" not in code and len(code) <= 12 and " " not in code
        if "K20C2" in code:  # suspected error in source table (2017-20 Civic Si is L15B7)
            continue
        out.append({"make": r["make"], "model": r["model"], "engine_code": code if clean else None})
    return out


def phone(rng):
    return f"{rng.choice(AREA)}-{rng.randint(200, 989)}-{rng.randint(0, 9999):04d}"


def postal(rng):
    L = "ABCEGHJKLMNPRSTVWXYZ"
    return (f"{rng.choice('KLMNP')}{rng.randint(0, 9)}{rng.choice(L)} "
            f"{rng.randint(0, 9)}{rng.choice(L)}{rng.randint(0, 9)}")


def person(rng):
    f, l = rng.choice(FIRST), rng.choice(LAST)
    email = rng.choice([f"{f}.{l}", f"{f[0]}{l}", f"{f}{l}{rng.randint(1, 99)}", f"{f}_{l[0]}"]).lower()
    return f"{f} {l}", f"{email}@{rng.choice(EMAIL_DOM)}"


def round_to(x, step):
    return int(round(x / step) * step)


def eta_text(d, rng):
    style = rng.choice(["long", "short", "weekday", "iso"])
    if style == "long":
        return d.strftime("%B ") + str(d.day)
    if style == "short":
        return d.strftime("%b ") + str(d.day)
    if style == "weekday":
        return d.strftime("%A, %b ") + str(d.day)
    return d.isoformat()


def build(i, rng, vehicles):
    kind = rng.choices(INTENTS, weights=[34, 24, 14, 13, 9, 6])[0]
    channel = rng.choice(["email", "email", "sms"]) if kind not in ("yard_offer", "freight_update") else "email"
    sent = dt.date(2026, 1, 5) + dt.timedelta(days=rng.randint(0, 300))
    g = {k: None for k in FIELDS}
    g["intent"] = kind
    facts = []          # (field, exact value to state, how to state it)
    v = rng.choice(vehicles)
    year = rng.randint(2003, 2022)
    names = list(PARTS)
    part = rng.choices(names, weights=[35, 25] + [40 / (len(names) - 2)] * (len(names) - 2))[0]
    pos = None
    if PARTS[part]:
        pos = rng.choice(LIGHT_POS if part in ("headlight", "taillight", "side mirror") else POSITIONS)
    ref = f"JA-{rng.randint(1000, 4999)}"
    header = {}
    forbid_in_draft, require_in_draft = [], []
    avoid = []  # things the writer must not mention

    if kind in ("quote_request", "where_is_my_part", "warranty_claim", "cancel_request"):
        name, email = person(rng)
        is_shop = rng.random() < 0.45
        g["sender_type"] = "shop" if is_shop else "individual"
        shop = f"{rng.choice(TOWNS)} {rng.choice(SHOP_WORDS)}" if is_shop else None
        if channel == "email":
            g["sender_name"], g["sender_contact"] = name, email
            header = {"from": f"{shop + ' - ' if shop else ''}{name} <{email}>"}
            facts.append(("sender_name", name, "your name: sign off with it (first name alone is fine); use no other names for yourself"))
        else:
            ph = phone(rng)
            g["sender_contact"] = ph
            header = {"from": f"SMS from {ph}"}
            if rng.random() < 0.6:
                g["sender_name"] = name
                facts.append(("sender_name", name, "sign off with this name"))
            else:
                avoid.append("any personal name (do not introduce yourself or sign with a name)")
        if is_shop:
            facts.append(("shop", shop, "background: you work at this repair shop (you may mention its name)"))
        else:
            facts.append(("persona", "car owner", "background only, never state it: you are fixing your own car"))

    if kind == "quote_request":
        g.update(vehicle_year=year, vehicle_make=v["make"], vehicle_model=v["model"], part=part,
                 part_position=pos)
        facts += [("vehicle_year", year, "model year"), ("vehicle_make", v["make"], "make"),
                  ("vehicle_model", v["model"], "model"), ("part", part, "the part needed")]
        if pos:
            facts.append(("part_position", pos, "which side/position"))
        if part == "engine" and v["engine_code"] and rng.random() < 0.4:
            g["engine_code"] = v["engine_code"]
            facts.append(("engine_code", v["engine_code"], "the engine code, written exactly"))
        else:
            avoid.append("any engine or transmission code")
        if rng.random() < 0.55:
            g["max_miles"] = rng.choice([60000, 80000, 100000, 120000, 150000]) if rng.random() < .5 \
                else round_to(rng.randint(50, 160) * 1000, 5000)
            facts.append(("max_miles", g["max_miles"], "maximum mileage you'll accept (digits, '80k' style ok)"))
        if rng.random() < 0.4:
            g["budget"] = round_to(rng.randint(250, 4500), 50)
            facts.append(("budget", g["budget"], "your budget in dollars"))
        if rng.random() < 0.5:
            g["ship_to_postal"] = postal(rng)
            facts.append(("ship_to_postal", g["ship_to_postal"], "postal code to ship to, written exactly"))
        require_in_draft = ["vehicle_year", "vehicle_make", "vehicle_model", "part"] + \
            [k for k in ("part_position", "engine_code", "max_miles") if g[k] is not None]
        forbid_in_draft = [k for k in ("budget", "sender_contact") if g[k] is not None]
        draft_to = "yard"

    elif kind == "yard_offer":
        yard = f"{rng.choice(TOWNS)} {rng.choice(YARD_WORDS)}"
        dom = re.sub(r"[^a-z]", "", yard.lower())[:22] + ".ca"
        g.update(sender_type="yard", yard_name=yard, sender_contact=f"sales@{dom}", ref=ref,
                 vehicle_year=year, vehicle_make=v["make"], vehicle_model=v["model"], part=part,
                 part_position=pos)
        header = {"from": f"{yard} <sales@{dom}>"}
        g["part_number"] = f"{rng.choice(['', 'P', 'Y', 'IN'])}{rng.randint(10000, 999999)}" \
            if rng.random() < .5 else f"{rng.choice('ABCDEFGHJK')}{rng.choice('ABCDEFGHJK')}-{rng.randint(1000, 99999)}"
        g["miles"] = round_to(rng.randint(28, 175) * 1000, 500)
        g["yard_cost"] = round_to(rng.randint(150, 3800), 25)
        facts += [("ref", ref, "the broker's request number you are answering"),
                  ("vehicle", f"{year} {v['make']} {v['model']}", "vehicle the part came off"),
                  ("part", part, "the part you have")]
        if pos:
            facts.append(("part_position", pos, "which side/position"))
        facts += [("part_number", g["part_number"], "your stock/part number, written exactly"),
                  ("miles", g["miles"], "mileage on the donor vehicle (digits, '62k' style ok)"),
                  ("yard_cost", g["yard_cost"], "your price to the broker in dollars")]
        if rng.random() < 0.5:
            g["warranty_days"] = rng.choice([30, 60, 90, 180])
            facts.append(("warranty_days", g["warranty_days"], "your warranty length in days"))
        facts.append(("yard_name", yard, "sign off as this yard"))
        avoid.append("any engine or transmission code")
        require_in_draft = ["part_number", "miles"] + (["warranty_days"] if g["warranty_days"] else [])
        forbid_in_draft = ["yard_cost", "yard_name", "sender_contact"]
        draft_to = "customer"

    elif kind == "freight_update":
        carrier = rng.choice(CARRIERS)
        eta = sent + dt.timedelta(days=rng.randint(1, 9))
        g.update(sender_type="carrier", carrier=carrier, ref=ref,
                 tracking_number=str(rng.randint(10 ** 7, 10 ** 10 - 1)), eta=eta.isoformat())
        dom = re.sub(r"[^a-z]", "", carrier.lower())
        g["sender_contact"] = f"noreply@{dom}.com"
        header = {"from": f"{carrier} <noreply@{dom}.com>"}
        et = eta_text(eta, rng)
        facts += [("carrier", carrier, "your company"), ("ref", ref, "the shipper's reference number"),
                  ("tracking_number", g["tracking_number"], "the PRO number, written exactly"),
                  ("eta", et, "estimated delivery date, written exactly like this")]
        avoid.append("any other dates")
        require_in_draft = ["tracking_number", "eta", "carrier"]
        forbid_in_draft = []
        draft_to = "customer"

    else:  # customer concerns
        if rng.random() < 0.75:
            g["ref"] = ref
            facts.append(("ref", ref, "your order number"))
        g["part"] = part
        facts.append(("part", part, "the part you bought"))
        if kind == "warranty_claim":
            g["issue"] = rng.choice(ISSUES)
            ok = ISSUES_FOR.get(part, ["wrong_part", "damaged"])
            if g["issue"] not in ok:  # deterministic remap keeps every other record's randomness unchanged
                g["issue"] = ok[i % len(ok)]
            hint = {"noise": "it makes a knocking/ticking noise", "leak": "it is leaking oil or coolant",
                    "no_start": "the car won't start/crank since install",
                    "slipping": "it slips or shifts badly", "wrong_part": "it's the wrong part / doesn't fit",
                    "damaged": "it arrived cracked or damaged"}[g["issue"]]
            facts.append(("issue", hint, "the problem, in your own words"))
        elif kind == "where_is_my_part":
            facts.append(("waiting", f"{rng.randint(5, 21)} days", "how long you've been waiting (optional)"))
        else:
            facts.append(("reason", rng.choice(["found one locally", "car got written off", "changed my mind",
                                                "the shop already fixed it"]), "why you want to cancel"))
        avoid.append("any dates, prices or tracking numbers")
        require_in_draft = ["ref"] if g["ref"] else []
        forbid_in_draft = []
        draft_to = "customer"

    if g["vehicle_model"] in EV and g["part"] in ("engine", "transmission", "turbocharger"):
        g["part"], g["engine_code"] = "front bumper", None
        facts = [(f, "front bumper" if f == "part" else v, how) for f, v, how in facts if f != "engine_code"]
        if "any engine or transmission code" not in avoid:
            avoid.append("any engine or transmission code")
    persona = rng.choice(["terse, all business", "friendly and chatty, adds a little backstory",
                          "stressed and impatient", "casual, lowercase, a couple of typos",
                          "polite and formal", "rambling, mentions unrelated details"])
    if channel == "sms":
        persona += ", text-message style, short"
    return {"id": i, "kind": kind, "channel": channel, "date": sent.isoformat(), "header": header,
            "persona": persona, "facts": facts, "avoid": avoid, "gold": g,
            "next_action": NEXT_ACTION[kind], "draft_to": draft_to,
            "draft_require": require_in_draft, "draft_forbid": forbid_in_draft}


def build_all(n, seed):
    rng = random.Random(seed)
    vehicles = load_vehicles()
    return [build(i, rng, vehicles) for i in range(n)]


if __name__ == "__main__":
    import collections
    rs = build_all(2000, 0)
    print(collections.Counter(r["kind"] for r in rs))
    print(json.dumps(rs[0], indent=1)[:1500])

"""Build CRM-update extraction cases from the public Olist orders dataset.

Each case = an inbound message (email or phone transcript) about one real Olist order, plus the
gold CRM update a correct extractor must produce. Names/emails/streets come from Faker; order ids,
cities, states, zip prefixes and delivery dates come from Olist.

    .venv/bin/python gen.py --n 240 --seed 0 --out data/cases.jsonl
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import random

FIELDS = ["order_id", "customer_email", "intent", "status", "carrier", "tracking_number",
          "estimated_delivery", "new_street", "new_city", "new_state", "new_zip"]
INTENTS = ["shipping_update", "delay_notice", "address_change", "delivery_complaint",
           "cancel_request", "status_inquiry"]
CARRIERS = ["Correios", "Jadlog", "Loggi", "Total Express"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
NATO = {c: w for c, w in zip("abcdef", ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])}


def olist_dir():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/kagglehub/datasets/olistbr/brazilian-ecommerce/versions/*"))
    if not hits:
        raise SystemExit("Olist not downloaded: run kagglehub.dataset_download('olistbr/brazilian-ecommerce')")
    return sorted(hits)[-1]


def load_orders():
    d = olist_dir()
    with open(f"{d}/olist_customers_dataset.csv") as f:
        cust = {r["customer_id"]: r for r in csv.DictReader(f)}
    rows = []
    with open(f"{d}/olist_orders_dataset.csv") as f:
        for r in csv.DictReader(f):
            if not r["order_estimated_delivery_date"] or not r["order_purchase_timestamp"]:
                continue
            c = cust.get(r["customer_id"])
            if c is None:
                continue
            rows.append({
                "order_id": r["order_id"][:8].upper(),  # shops show short order numbers
                "purchased": dt.datetime.fromisoformat(r["order_purchase_timestamp"]).date(),
                "eta": dt.datetime.fromisoformat(r["order_estimated_delivery_date"]).date(),
                "city": c["customer_city"].title(),
                "state": c["customer_state"],
                "zip_prefix": c["customer_zip_code_prefix"].zfill(5),
            })
    return rows


def tracking_for(carrier, rng):
    if carrier == "Correios":
        return (rng.choice(["OJ", "QB", "LB", "NX"]) + "".join(rng.choices("0123456789", k=9)) + "BR")
    if carrier == "Jadlog":
        return "".join(rng.choices("0123456789", k=14))
    return "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", k=10))


def fmt_date(d, rng):
    """Mixed formats on purpose. dd/mm is the Brazilian default and a real trap."""
    style = rng.choice(["iso", "br", "long", "weekday"])
    if style == "iso":
        return d.isoformat()
    if style == "br":
        return d.strftime("%d/%m/%Y")
    if style == "long":
        return f"{MONTHS_EN[d.month - 1]} {d.day}, {d.year}"
    return f"{d.strftime('%A')}, {d.day} {MONTHS_EN[d.month - 1]} {d.year}"


def spell(s):
    out = []
    for ch in s.lower():
        if ch.isdigit():
            out.append(DIGITS[int(ch)])
        elif ch in NATO:
            out.append(f"{ch} as in {NATO[ch]}")
        else:
            out.append(ch)
    return ", ".join(out)


def cep(prefix, rng):
    return f"{prefix}-{rng.randint(0, 999):03d}"


def empty():
    return {k: None for k in FIELDS}


def make_case(o, other, fake, rng):
    """Returns (kind, message, gold, tags). tags mark the traps a case contains."""
    sent = o["purchased"] + dt.timedelta(days=rng.randint(1, 6))
    name = fake.name()
    email = fake.free_email()
    gold = empty()
    gold["order_id"] = o["order_id"]
    tags = []
    kind = rng.choice(["shipping_update", "delay_notice", "address_change", "delivery_complaint",
                       "cancel_request", "status_inquiry", "phone_address_change", "phone_status"])

    if kind in ("shipping_update", "delay_notice"):
        carrier = rng.choice(CARRIERS)
        trk = tracking_for(carrier, rng)
        eta = o["eta"] if kind == "shipping_update" else o["eta"] + dt.timedelta(days=rng.randint(2, 9))
        gold.update(intent=kind, status="shipped" if kind == "shipping_update" else "delayed",
                    carrier=carrier, tracking_number=trk, estimated_delivery=eta.isoformat())
        relative = rng.random() < 0.3
        eta_txt = (f"within {(eta - sent).days} days of this notice" if relative else fmt_date(eta, rng))
        if relative:
            tags.append("relative_date")
        correction = rng.random() < 0.25
        lines = [f"From: notificacoes@{carrier.lower().replace(' ', '')}.com.br",
                 f"Date: {sent.isoformat()}",
                 f"Subject: {'Shipment update' if kind == 'shipping_update' else 'Delivery delay'} - order {o['order_id']}", ""]
        if correction:
            wrong = tracking_for(carrier, rng)
            lines.append(f"Your order {o['order_id']} was handed to {carrier}. Tracking: {wrong}.")
            lines.append(f"CORRECTION: the tracking code above was entered incorrectly. The correct code is {trk}.")
            tags.append("correction")
        else:
            lines.append(f"Your order {o['order_id']} has been handed to {carrier}. Tracking code: {trk}.")
        if kind == "delay_notice":
            lines.append(f"Due to high volume in {o['state']}, delivery has been rescheduled. "
                         f"New expected delivery: {eta_txt}.")
        else:
            lines.append(f"Expected delivery: {eta_txt}.")
        lines.append("This is an automated message; customer contact details are not included.")
        tags.append("no_customer_email")
        return kind, "\n".join(lines), gold, tags

    if kind in ("address_change", "phone_address_change"):
        street = fake.street_address()
        new = rng.choice(other)
        zipc = cep(new["zip_prefix"], rng)
        gold.update(intent="address_change", new_street=street, new_city=new["city"],
                    new_state=new["state"], new_zip=zipc)
        correction = rng.random() < 0.3
        if kind == "address_change":
            gold["customer_email"] = email
            wrong_zip = cep(rng.choice(other)["zip_prefix"], rng)
            body = [f"From: {name} <{email}>", f"Date: {sent.isoformat()}",
                    f"Subject: change address for order {o['order_id']}", "",
                    "Hi, I moved last week. Could you please send my order to my new address:",
                    f"{street}, {new['city']} - {new['state']}, CEP {wrong_zip if correction else zipc}"]
            if correction:
                body.append(f"Oops, sorry, the CEP is wrong, it should be {zipc}.")
                tags.append("correction")
            if rng.random() < 0.4:
                prev = rng.choice(other)["order_id"]
                body.append(f"(My last order {prev} went to the old place and was fine, just this one.)")
                tags.append("distractor_order")
            body.append(f"Thanks, {name.split()[0]}")
            return kind, "\n".join(body), gold, tags
        tags += ["phone", "spelled_id", "no_customer_email"]
        t = [f"[Call transcript, {sent.isoformat()}]",
             "AGENT: Thanks for calling, how can I help?",
             f"CALLER: Hi, yeah, um, I need to change where my order's going. The order number is {spell(o['order_id'])}.",
             "AGENT: Got it. And the new address?",
             f"CALLER: It's {street}, in {new['city']}, {new['state']}. Zip is, uh, {wrong_zip_say(zipc, rng, correction)}"]
        if correction:
            t.append(f"CALLER: no wait, sorry, it's {zipc}.")
            tags.append("correction")
        t.append("AGENT: Perfect, I've noted that down.")
        return kind, "\n".join(t), gold, tags

    if kind == "delivery_complaint":
        gold.update(intent="delivery_complaint", customer_email=email)
        tags.append("abstain_status")  # a customer saying it's late is not an authoritative status
        late = (o["eta"] + dt.timedelta(days=rng.randint(1, 10)))
        body = [f"From: {name} <{email}>", f"Date: {late.isoformat()}",
                f"Subject: where is my order??", "",
                f"I was told my order {o['order_id']} would arrive by {fmt_date(o['eta'], rng)} and it still hasn't come.",
                "This is really frustrating. Please tell me what is going on."]
        return kind, "\n".join(body), gold, tags

    if kind == "cancel_request":
        gold.update(intent="cancel_request", customer_email=email)
        tags.append("abstain_status")  # requested, not yet canceled
        body = [f"From: {name} <{email}>", f"Date: {sent.isoformat()}",
                f"Subject: cancel", "",
                f"Please cancel order {o['order_id']}, I bought it by mistake. Can I get a refund?"]
        return kind, "\n".join(body), gold, tags

    if kind == "status_inquiry":
        gold.update(intent="status_inquiry", customer_email=email)
        tags.append("mostly_null")
        body = [f"From: {name} <{email}>", f"Date: {sent.isoformat()}",
                f"Subject: order {o['order_id']}", "",
                "hi just checking if my order shipped yet? thx"]
        return kind, "\n".join(body), gold, tags

    # phone_status: caller gives order id only; everything else must be null
    gold.update(intent="status_inquiry")
    tags += ["phone", "spelled_id", "mostly_null", "no_customer_email"]
    t = [f"[Call transcript, {sent.isoformat()}]",
         "AGENT: Hello, how can I help?",
         f"CALLER: Hi, I'm calling about my order, it's {spell(o['order_id'])}. Has it shipped? I think it was Correios last time.",
         "AGENT: Let me check on that for you."]
    tags.append("distractor_carrier")
    return "phone_status", "\n".join(t), gold, tags


def wrong_zip_say(zipc, rng, correction):
    if not correction:
        return " ".join(DIGITS[int(c)] if c.isdigit() else "dash" for c in zipc)
    z = list(zipc)
    i = rng.choice([k for k, c in enumerate(z) if c.isdigit()])
    z[i] = str((int(z[i]) + rng.randint(1, 8)) % 10)
    return " ".join(DIGITS[int(c)] if c.isdigit() else "dash" for c in z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/cases.jsonl")
    a = ap.parse_args()
    from faker import Faker  # only the generator needs it
    rng = random.Random(a.seed)
    fake = Faker("pt_BR")
    Faker.seed(a.seed)
    orders = load_orders()
    rng.shuffle(orders)
    pool, other = orders[: a.n], orders[a.n: a.n + 5000]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for i, o in enumerate(pool):
            kind, msg, gold, tags = make_case(o, other, fake, rng)
            f.write(json.dumps({"id": i, "kind": kind, "message": msg, "gold": gold, "tags": tags}) + "\n")
    print(f"wrote {a.n} cases from {len(orders)} Olist orders -> {a.out}")


if __name__ == "__main__":
    main()

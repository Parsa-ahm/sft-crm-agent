"""Quality report on generated messages: acceptance, failure reasons, diversity, random samples.

    python3 qc.py data/train_raw.jsonl --from 0 --to 400 --samples 6
"""
import argparse
import collections
import json
import random
import re

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--from", dest="lo", type=int, default=0)
ap.add_argument("--to", dest="hi", type=int, default=10 ** 9)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.path)]
rows = [r for r in rows if a.lo <= r["id"] < a.hi]
n = len(rows)
acc = [r for r in rows if r["accepted"]]
print(f"== {a.path} ids [{a.lo},{a.hi})  n={n}  accepted={len(acc)} ({len(acc)/max(n,1):.0%})")
rc = collections.Counter(re.sub(r"[:?].*", "", x) for r in rows for x in r["reasons"])
print("reject reasons:", dict(rc.most_common(10)))
print("accept by kind:", {k: f"{sum(r['accepted'] for r in rows if r['kind']==k)}/{sum(1 for r in rows if r['kind']==k)}"
                          for k in sorted({r['kind'] for r in rows})})
bodies = [r["body"] for r in acc]
if bodies:
    lens = sorted(len(b) for b in bodies)
    print(f"length chars: p10={lens[len(lens)//10]} median={lens[len(lens)//2]} p90={lens[9*len(lens)//10]}")
    openers = collections.Counter(" ".join(re.sub(r"(?i)^subject:.*\n+", "", b).split()[:3]).lower() for b in bodies)
    top = openers.most_common(5)
    print(f"distinct 3-word openers: {len(openers)}/{len(bodies)}   most common: {top}")
    tri = collections.Counter()
    for b in bodies:
        w = re.findall(r"[a-z']+", b.lower())
        tri.update({" ".join(w[i:i+3]) for i in range(len(w) - 2)})
    common = [(t, c) for t, c in tri.most_common(8) if c > 0.25 * len(bodies)]
    print("trigrams in >25% of messages:", common)
rng = random.Random(a.seed)
for r in rng.sample(acc, min(a.samples, len(acc))):
    print(f"\n--- id={r['id']} {r['kind']} / {r['channel']} / {r['persona']}")
    print(r["message"][:900])
rej = [r for r in rows if not r["accepted"] and r["body"]]
for r in rng.sample(rej, min(2, len(rej))):
    print(f"\n--- REJECTED id={r['id']} {r['kind']} reasons={r['reasons']}")
    print(r["message"][:600])

import json, random, re, glob
rows = []
for f in sorted(glob.glob("data/final/*.jsonl")):
    for l in open(f):
        r = json.loads(l); r["split"] = f.split("/")[-1][:-6]; rows.append(r)
def norm(s): return re.sub(r"[\s,\-]", "", str(s)).lower()
bad = []
for r in rows:
    d, g = r["target"]["draft"]["body"], r["gold"]
    for k in r["draft_require"]:
        if g[k] is not None and norm(g[k]) not in norm(d): bad.append((r["split"], r["id"], "require", k))
    for k in r["draft_forbid"]:
        if g[k] is not None and norm(g[k]) in norm(d): bad.append((r["split"], r["id"], "forbid", k))
print("draft rule violations in reference drafts:", len(bad), bad[:5])
print("exact duplicate messages:", len(rows) - len({r["message"] for r in rows}))
for r in random.Random(42).sample(rows, 100):
    g = {k: v for k, v in r["gold"].items() if v is not None}
    body = re.sub(r"\s+", " ", r["message"])[:420]
    print("\n#%s/%s %s\nMSG: %s\nGOLD: %s" % (r["split"], r["id"], r["kind"], body, json.dumps(g)))

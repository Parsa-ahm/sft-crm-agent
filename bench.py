"""Score every run in RUNS with eval.py, write data/results/, print the README tables.

    .venv/bin/python bench.py
Needs no GPU or API key (loads only the tokenizer, to count local prompt tokens).
Runs that do not cover the whole split are skipped.
"""
import json
import math
import os
import sys

from transformers import AutoTokenizer

import eval as ev
from train import MODEL, build_prompt

TEST = "data/test.jsonl"
RUNS = [  # (key, label, size, run file from generate.py)
    ("sft", "Qwen2.5-1.5B + SFT (ours)", "1.5B", "data/runs/sft_test.jsonl"),
    ("base", "Qwen2.5-1.5B-Instruct", "1.5B", "data/runs/base_test.jsonl"),
    ("gptoss0", "gpt-oss-20b, zero-shot", "21B (3.6B active)", "data/runs/gpt-oss-20b_test.jsonl"),
    ("gptoss6", "gpt-oss-20b, 6-shot", "21B (3.6B active)", "data/runs/gpt-oss-20b_6shot_test.jsonl"),
    ("nemotron", "Nemotron-3.5-Lightning", "30B (3B active)", "data/runs/nemotron-3.5-lightning-30b-a3b_test.jsonl"),
]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


gold = {r["id"]: r for r in ev.load_rows(TEST)}
tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", MODEL))
local_prompt_tokens = sum(len(tok(build_prompt(tok, r), add_special_tokens=False)["input_ids"])
                          for r in gold.values()) / len(gold)

os.makedirs("data/results", exist_ok=True)
summary, passed = {}, {}
for key, label, size, path in RUNS:
    if not os.path.exists(path):
        print(f"skip {key}: {path} missing", file=sys.stderr)
        continue
    rows, texts, scores = ev.score_run(path, TEST)
    if len({r["id"] for r in rows}) < len(gold):
        print(f"skip {key}: {len(rows)}/{len(gold)} done", file=sys.stderr)
        continue
    recs = [json.loads(l) for l in open(path)]
    out = sorted(({"id": r["id"], "kind": r["kind"], "output": t, "score": s}
                  for r, t, s in zip(rows, texts, scores)), key=lambda r: r["id"])
    with open(f"data/results/{key}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    passed[key] = {r["id"]: ev.full_pass(r["score"]) for r in out}
    tokens = [r["usage"]["prompt_tokens"] for r in recs if r.get("usage", {}).get("prompt_tokens")]
    n = len(out)
    sc = [r["score"] for r in out]
    kinds = sorted({r["kind"] for r in out})
    summary[key] = {
        "label": label, "size": size, "n": n,
        "prompt_tokens": round(sum(tokens) / len(tokens)) if tokens else round(local_prompt_tokens),
        "valid": sum(s["valid"] for s in sc) / n,
        "fields": sum(s["fields_ok"] for s in sc) / n,
        "action": sum(s["action_ok"] for s in sc) / n,
        "draft": sum(s["draft_ok"] for s in sc) / n,
        "full": sum(map(ev.full_pass, sc)) / n,
        "full_ci": wilson(sum(map(ev.full_pass, sc)), n),
        "halluc": sum(bool(s["halluc"]) for s in sc) / n,
        "omit": sum(bool(s["omit"]) for s in sc) / n,
        "by_kind": {k: sum(ev.full_pass(r["score"]) for r in out if r["kind"] == k) /
                    sum(r["kind"] == k for r in out) for k in kinds},
        "kind_n": {k: sum(r["kind"] == k for r in out) for k in kinds},
    }
json.dump(summary, open("data/results/summary.json", "w"), indent=1)

pct = lambda x: f"{100 * x:.1f}%"
print("| system | full pass | 95% CI | all fields | next_action | draft ok | valid JSON | ≥1 hallucinated field | prompt tokens |")
print("|---|---|---|---|---|---|---|---|---|")
for key, s in summary.items():
    lo, hi = s["full_ci"]
    print(f"| {s['label']} | {pct(s['full'])} | {100 * lo:.1f}–{100 * hi:.1f} | {pct(s['fields'])} | {pct(s['action'])} "
          f"| {pct(s['draft'])} | {pct(s['valid'])} | {pct(s['halluc'])} | {s['prompt_tokens']:,} |")

keys = list(summary)
print("\n| kind | n | " + " | ".join(summary[k]["label"] for k in keys) + " |")
print("|---|---|" + "---|" * len(keys))
first = summary[keys[0]]
for kind in sorted(first["by_kind"], key=lambda k: -first["kind_n"][k]):
    print(f"| {kind} | {first['kind_n'][kind]} | " + " | ".join(pct(summary[k]["by_kind"][kind]) for k in keys) + " |")

if "sft" in passed:
    print()
    for key in keys:
        if key == "sft":
            continue
        a = sum(passed["sft"][i] and not passed[key][i] for i in gold)
        b = sum(passed[key][i] and not passed["sft"][i] for i in gold)
        print(f"sft vs {key}: sft-only {a}, {key}-only {b}, exact McNemar p = {mcnemar(a, b):.2g}")

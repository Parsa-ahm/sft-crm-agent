"""Build mine/data/results/*.jsonl (held-out test, every model re-scored with mine/eval.py) and print summary JSON.

Run from crm-extract-rl/mine:  ../.venv/bin/python /path/build_results.py
"""
import json
import os
import sys

sys.path.insert(0, ".")
import eval as ev  # Parsa's scorer: verified 800/800 identical to the reference

from transformers import AutoTokenizer

TEST = "data/test.jsonl"
RUNS = [  # (key, label, size, source file, kind of source)
    ("sft", "Qwen2.5-1.5B + LoRA SFT (this work)", "1.5B", "../data/eval_my_sft300_test.jsonl", "eval"),
    ("base", "Qwen2.5-1.5B-Instruct, zero-shot", "1.5B", "../data/eval_base_test.jsonl", "eval"),
    ("gptoss0", "gpt-oss-20b, zero-shot", "21B (3.6B active)", "../data/bench_openai_gpt-oss-20b_test.jsonl", "bench"),
    ("gptoss6", "gpt-oss-20b, 6-shot", "21B (3.6B active)", "../data/bench_openai_gpt-oss-20b_6shot_test.jsonl", "bench"),
    ("nemotron", "Nemotron-3.5-Lightning, zero-shot", "30B (3B active)",
     "../data/bench_nvidia_nemotron-3.5-lightning-30b-a3b_test.jsonl", "bench"),
]

rows = {r["id"]: r for r in ev.load_rows(TEST)}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
local_prompt_tokens = sum(len(tok(ev.build_prompt(tok, r), add_special_tokens=False)["input_ids"])
                          for r in rows.values()) / len(rows)

os.makedirs("data/results", exist_ok=True)
summary = {}
for key, label, size, path, kind in RUNS:
    if not os.path.exists(path):
        continue
    recs = [json.loads(l) for l in open(path)]
    if len(recs) < len(rows):
        print(f"skip {key}: {len(recs)}/{len(rows)} done", file=sys.stderr)
        continue
    out, tokens = [], []
    for rec in recs:
        text = rec["text"] if kind == "eval" else rec["greedy_text"]
        s = ev.score(text, rows[rec["id"]])
        out.append({"id": rec["id"], "kind": rec["kind"], "output": text, "score": s})
        if kind == "bench" and rec.get("usage", {}).get("prompt_tokens"):
            tokens.append(rec["usage"]["prompt_tokens"])
    out.sort(key=lambda r: r["id"])
    with open(f"data/results/{key}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    sc = [r["score"] for r in out]
    n = len(sc)
    summary[key] = {
        "label": label, "size": size, "n": n,
        "prompt_tokens": round(sum(tokens) / len(tokens)) if tokens else round(local_prompt_tokens),
        "valid": sum(s["valid"] for s in sc) / n,
        "fields": sum(s["fields_ok"] for s in sc) / n,
        "action": sum(s["action_ok"] for s in sc) / n,
        "draft": sum(s["draft_ok"] for s in sc) / n,
        "full": sum(ev.full_pass(s) for s in sc) / n,
        "halluc": sum(bool(s["halluc"]) for s in sc) / n,
        "omit": sum(bool(s["omit"]) for s in sc) / n,
        "by_kind": {k: sum(ev.full_pass(r["score"]) for r in out if r["kind"] == k) /
                    max(1, sum(r["kind"] == k for r in out)) for k in sorted({r["kind"] for r in out})},
        "kind_n": {k: sum(r["kind"] == k for r in out) for k in sorted({r["kind"] for r in out})},
    }
json.dump(summary, open("data/results/summary.json", "w"), indent=1)
print(json.dumps({k: {m: v[m] for m in ("n", "full", "fields", "draft", "halluc", "prompt_tokens")}
                  for k, v in summary.items()}, indent=1))

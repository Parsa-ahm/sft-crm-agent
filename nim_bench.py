"""Score NIM-hosted models on val with the same prompt + scorer as probe_crm.py (greedy, T=0).

    .venv/bin/python nim_bench.py --model google/gemma-3-4b-it
Resumable: rows already in data/bench_<model>.jsonl are skipped. Key read from file, never printed.
"""
import argparse
import concurrent.futures as cf
import json
import os
import time

import urllib.error
import urllib.request

from probe_crm import SYSTEM, parse, passes, score

KEY = open(os.path.expanduser("~/.config/nim/api-key")).read().strip()
URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def shot_messages(n):
    """One training example per message kind, as user/assistant turns (from sft.jsonl, never val/test)."""
    seen, msgs = set(), []
    for line in open("data/final/sft.jsonl"):
        r = json.loads(line)
        if r["kind"] in seen or len(seen) >= n:
            continue
        seen.add(r["kind"])
        msgs += [{"role": "user", "content": r["message"]},
                 {"role": "assistant", "content": json.dumps(r["target"])}]
    return msgs


SHOTS = []


def call(model, message, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, *SHOTS, {"role": "user", "content": message}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    for attempt in range(6):
        req = urllib.request.Request(URL, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
                return resp["choices"][0]["message"].get("content") or "", resp.get("usage", {})
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (400, 401, 403, 404):
                break
        except (urllib.error.URLError, TimeoutError) as e:
            err = str(e)
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", default="data/final/val.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--shots", type=int, default=0)
    a = ap.parse_args()
    SHOTS.extend(shot_messages(a.shots))

    tag = f"_{a.shots}shot" if a.shots else ""
    split = os.path.splitext(os.path.basename(a.cases))[0]
    if split != "val":
        tag += f"_{split}"
    out = f"data/bench_{a.model.replace('/', '_')}{tag}.jsonl"
    rows = [json.loads(l) for l in open(a.cases)]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.model}: {len(done)} done, {len(todo)} to go", flush=True)

    def work(r):
        text, usage = call(a.model, r["message"], a.max_tokens)
        return {"id": r["id"], "kind": r["kind"], "channel": r["channel"], "greedy_text": text,
                "usage": usage, "greedy": score(parse(text), r)}

    t0 = time.time()
    with open(out, "a") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for n, fut in enumerate(cf.as_completed(futs), 1):
            try:
                rec = fut.result()
            except RuntimeError as e:
                print(f"FAILED: {e}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    recs = [json.loads(l) for l in open(out)]
    g = [r["greedy"] for r in recs]
    pct = lambda xs: f"{100 * sum(xs) / len(xs):.1f}%"
    print(f"=== {a.model} n={len(g)} ===")
    print(f"valid JSON {pct([s['valid'] for s in g])}  all fields {pct([s['fields_ok'] for s in g])}  "
          f"next_action {pct([s['action_ok'] for s in g])}  draft ok {pct([s['draft_ok'] for s in g])}")
    print(f"full pass {pct([passes(s, 'full') for s in g])}  hallucinated {pct([bool(s['halluc']) for s in g])}")
    pt = [r["usage"].get("prompt_tokens", 0) for r in recs if r.get("usage")]
    if pt:
        print(f"mean prompt tokens {sum(pt) / len(pt):.0f}")


if __name__ == "__main__":
    main()

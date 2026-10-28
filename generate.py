"""Run one model over a split and save its raw outputs to data/runs/<name>_<split>.jsonl.

Local Hugging Face checkpoint (greedy, fp16, batched):
    .venv/bin/python generate.py --model ckpt/sft300 --name sft
NVIDIA NIM API model (temperature 0, resumable: ids already in the output are skipped):
    .venv/bin/python generate.py --model nim:openai/gpt-oss-20b --shots 6
    .venv/bin/python generate.py --model nim:nvidia/nemotron-3.5-lightning-30b-a3b --max-new 16384

Every model sees the same SYSTEM instruction from train.py. Score with eval.py or bench.py.
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request

from eval import full_pass, report, score
from train import SYSTEM, build_prompt, load_rows

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def generate(model, tok, rows, batch, max_new):
    import torch
    texts = []
    for i in range(0, len(rows), batch):
        prompts = [build_prompt(tok, r) for r in rows[i:i + batch]]
        enc = tok(prompts, return_tensors= 'pt', padding = True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens = max_new, do_sample=False, pad_token_id = tok.pad_token_id)

        new = out[:, enc["input_ids"].shape[1]:]
        texts += tok.batch_decode(new, skip_special_tokens=True)
        print(f"{len(texts)}/{len(rows)}", flush=True)
    return texts


def run_local(path, rows, batch, max_new):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to("cuda").eval()
    texts = generate(model, tok, rows, batch, max_new)
    return [{"id": r["id"], "kind": r["kind"], "text": t} for r, t in zip(rows, texts)]


def shot_messages(n):
    """One training example per message kind, as user/assistant turns (from sft.jsonl, never val/test)."""
    seen, msgs = set(), []
    for r in load_rows("data/sft.jsonl"):
        if r["kind"] in seen or len(seen) >= n:
            continue
        seen.add(r["kind"])
        msgs += [{"role": "user", "content": r["message"]},
                 {"role": "assistant", "content": json.dumps(r["target"])}]
    return msgs


def nim_call(key, model, messages, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    err = "no attempt made"
    for attempt in range(6):
        req = urllib.request.Request(NIM_URL, data=data, headers=headers)
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


def run_nim(model, rows, out, shots, max_tokens, workers):
    key_file = os.environ.get("NIM_KEY_FILE")
    if not key_file:
        raise SystemExit("set NIM_KEY_FILE to a file containing an NVIDIA NIM API key")
    key = open(os.path.expanduser(key_file)).read().strip()
    prefix = [{"role": "system", "content": SYSTEM}, *shot_messages(shots)]
    done = {r["id"] for r in load_rows(out)} if os.path.exists(out) else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{model}: {len(done)} done, {len(todo)} to go", flush=True)

    def work(r):
        text, usage = nim_call(key, model, prefix + [{"role": "user", "content": r["message"]}], max_tokens)
        return {"id": r["id"], "kind": r["kind"], "text": text, "usage": usage}

    t0 = time.time()
    with open(out, "a") as f, cf.ThreadPoolExecutor(workers) as ex:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint path, or nim:<model id>")
    ap.add_argument("--name", default=None, help="run name (default: last part of the model id)")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--shots", type=int, default=0, help="NIM only: in-context examples, one per kind")
    ap.add_argument("--max-new", type=int, default=None, help="default 512 local, 1024 NIM")
    ap.add_argument("--batch", type=int, default=16, help="local only")
    ap.add_argument("--workers", type=int, default=4, help="NIM only: parallel requests")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    nim = args.model.startswith("nim:")
    name = args.name or args.model.rstrip("/").split("/")[-1] + (f"_{args.shots}shot" if args.shots else "")
    split = os.path.splitext(os.path.basename(args.data))[0]
    out = f"data/runs/{name}_{split}.jsonl"
    os.makedirs("data/runs", exist_ok=True)
    rows = load_rows(args.data)[: args.limit]

    if nim:
        run_nim(args.model[4:], rows, out, args.shots, args.max_new or 1024, args.workers)
    else:
        recs = run_local(args.model, rows, args.batch, args.max_new or 512)
        with open(out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    print("wrote", out)

    gold = {r["id"]: r for r in rows}
    recs = [r for r in load_rows(out) if r["id"] in gold]
    scores = [score(r["text"], gold[r["id"]]) for r in recs]
    report([gold[r["id"]] for r in recs], scores)
    print(f"n={len(scores)}  full pass {sum(map(full_pass, scores))}/{len(scores)}")


if __name__ == "__main__":
    main()

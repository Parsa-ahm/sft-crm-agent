#!/bin/bash
# Waits for train generation, refills API-error rows, then builds data/final/. Log: finish.log
cd "$(dirname "$0")"
while kill -0 "$(cat gen_train.pid)" 2>/dev/null; do sleep 20; done
.venv/bin/python - <<'PY'
import json
for f in ["data/train_raw.jsonl", "data/test_raw.jsonl"]:
    rows = [json.loads(l) for l in open(f)]
    keep = [r for r in rows if not any(x.startswith(("gave_up", "http_")) for x in r["reasons"])]
    open(f, "w").write("".join(json.dumps(r) + "\n" for r in keep))
    print(f, len(rows), "->", len(keep), "(API-error rows removed for refill)")
PY
.venv/bin/python gen_nim.py --model google/gemma-4-31b-it --n 3300 --seed 0 --out data/train_raw.jsonl --workers 4
.venv/bin/python finalize.py
echo FINISHED  # note: printed even if finalize failed; check for Traceback above

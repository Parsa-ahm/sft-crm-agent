#!/bin/bash
# Held-out test set (330 rows, Kimi-written) for sft300 and sft_all. Log: test_probe.log
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for C in sft300 sft_all; do
  $PY probe_crm.py --model ckpt/$C --cases data/final/test.jsonl --k 1 --batch 16 \
    --out data/probe_test_$C.jsonl 2>&1 | grep -v "Loading weights" > probe_test_$C.log
  tail -4 probe_test_$C.log
done
echo TEST_PROBE_OK

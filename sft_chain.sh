#!/bin/bash
# SFT-300 (warm-up for GRPO) and SFT-all (baseline), each probed on val. Log: sft_chain.log
set -e
cd "$(dirname "$0")"
PY=../rl-lisp-python/.venv/bin/python
M=$(ls -d ../rl-lisp-python/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY sft.py --model $M --data data/final/sft.jsonl --out ckpt/sft300 2>&1 | grep -v "Loading weights\|Writing model" > sft300.log
$PY probe_crm.py --model ckpt/sft300 --k 8 --batch 16 --out data/probe_crm_sft300.jsonl 2>&1 | grep -v "Loading weights" > probe_crm_sft300.log
$PY sft.py --model $M --data data/final/sft_all.jsonl --out ckpt/sft_all --epochs 2 2>&1 | grep -v "Loading weights\|Writing model" > sft_all.log
$PY probe_crm.py --model ckpt/sft_all --k 8 --batch 16 --out data/probe_crm_sft_all.jsonl 2>&1 | grep -v "Loading weights" > probe_crm_sft_all.log
echo CHAIN_OK

"""SFT warm-up: LoRA on Qwen2.5-1.5B-Instruct, message -> target JSON, loss on the answer only.

Hand-rolled LoRA (no peft dependency). fp16 base (sm_75 has no bf16), fp32 LoRA params, autocast +
GradScaler. After training the LoRA is merged into the weights and saved as a plain HF checkpoint,
so probe_crm.py and the JAX GRPO loader read it like any other Qwen2 model.

    ../rl-lisp-python/.venv/bin/python sft.py --data data/final/sft.jsonl --out ckpt/sft300
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_crm import SYSTEM

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRA(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T @ self.B.T) * self.scale).to(x.dtype)

    def merged(self):
        return (self.base.weight.float() + self.scale * self.B @ self.A).to(self.base.weight.dtype)


def add_lora(model, r, alpha):
    mods = []
    for name, m in list(model.named_modules()):
        for child, sub in list(m.named_children()):
            if child in TARGETS and isinstance(sub, nn.Linear):
                lo = LoRA(sub, r, alpha)
                setattr(m, child, lo)
                mods.append((m, child, lo))
    return mods


def merge(mods):
    for parent, child, lo in mods:
        lin = lo.base
        lin.weight.data = lo.merged()
        setattr(parent, child, lin)


def encode(tok, row, max_len):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["message"]}],
        tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(json.dumps(row["target"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return torch.tensor(ids), torch.tensor(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/final/sft.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    mods = add_lora(model, a.r, a.alpha)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    params = [p for _, _, lo in mods for p in (lo.A, lo.B)]
    print(f"LoRA r={a.r}: {len(mods)} modules, {sum(p.numel() for p in params)/1e6:.1f}M trainable", flush=True)

    rows = [json.loads(l) for l in open(a.data)]
    data = [encode(tok, r, a.max_len) for r in rows]
    print(f"{len(data)} examples, max len {max(len(d[0]) for d in data)}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(data) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda")
    rng, step, t0 = random.Random(a.seed), 0, time.time()
    model.train()
    for ep in range(a.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0
        for i, j in enumerate(order):
            ids, labels = (t.unsqueeze(0).cuda() for t in data[j])
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model.model(input_ids=ids).last_hidden_state
            keep = labels[0, 1:] != -100
            logits = model.lm_head(hidden[0, :-1][keep]).float()
            loss = nn.functional.cross_entropy(logits, labels[0, 1:][keep])
            scaler.scale(loss / a.accum).backward()
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 5 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run_loss/run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{(time.time()-t0)/60:.1f} min  vram {torch.cuda.max_memory_allocated()/2**30:.2f}G",
                          flush=True)
                    run_loss, run_n = 0.0, 0
    merge(mods)
    model.config.use_cache = True
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved {a.out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

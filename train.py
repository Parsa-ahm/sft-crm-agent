import json
import torch
import torch.nn as nn
import math
import random
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = """You are the inbox assistant for a one-person used auto parts broker in Ontario, Canada.
He sources used parts from junkyards and ships them to customers. Messages come from customers,
junkyards ("yards") and freight carriers. Read one inbound message and return ONLY a JSON object
with exactly these keys:
["intent", "sender_name", "sender_contact", "sender_type", "ref", "vehicle_year", "vehicle_make", "vehicle_model", "engine_code", "part", "part_position", "max_mileage", "budget", "ship_to_postal", "yard_name", "part_number", "mileage", "yard_cost", "warranty_days", "carrier", "tracking_number", "eta", "issue", "next_action", "draft"]

Fields (use null for anything the message does not state; never guess or invent values):
- intent: one of ["quote_request", "yard_offer", "freight_update", "where_is_my_part", "warranty_claim", "cancel_request"]
- sender_name: the sender's full name as written, or null. For a yard or carrier, null.
- sender_contact: the sender's email or phone number from the message header.
- sender_type: one of "individual", "shop", "yard", "carrier".
- ref: the order or request number (like JA-1234), or null.
- vehicle_year (integer), vehicle_make, vehicle_model, engine_code.
- part: one of ["engine", "transmission", "turbocharger", "ecu", "rear differential", "transfer case", "alternator", "starter", "headlight", "taillight", "door", "side mirror", "front bumper"]
- part_position: one of ["front left", "front right", "rear left", "rear right", "left", "right"], or null.
- max_mileage: the customer's maximum acceptable mileage, integer (62k -> 62000).
- budget: the customer's budget in dollars, integer.
- ship_to_postal: Canadian postal code to ship to.
- yard_name, part_number, mileage (integer, mileage of the yard's part), yard_cost (integer,
  the yard's price to the broker), warranty_days (integer).
- carrier, tracking_number (the PRO number), eta (estimated delivery, YYYY-MM-DD; use the
  message date to work out the year).
- issue: for a warranty claim, one of ["noise", "leak", "no_start", "slipping", "wrong_part", "damaged"], else null.

next_action: ONE string, picked by intent: quote_request -> "ask_yard"; yard_offer -> "quote_customer"; freight_update -> "update_customer"; where_is_my_part -> "reply_customer"; warranty_claim -> "reply_customer"; cancel_request -> "reply_customer"

draft: {"to": "yard" or "customer", "body": "..."}, a short reply the broker approves before sending.
- quote_request: write to the yard. Ask for the part: vehicle, part, position, engine code and
  max mileage if given. Ask for part number, mileage and price. Never include the customer's
  budget or contact details.
- yard_offer: write to the customer. Offer the part with part number, mileage and warranty.
  Write the price as [PRICE] (the broker sets it). Never include the yard's cost or the yard's name.
- freight_update: write to the customer with the carrier, PRO number and delivery date.
- where_is_my_part, warranty_claim, cancel_request: reply to the customer, mention their order
  number if given.
- Never write a number, price, date or ID that is not in the message."""

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_prompt(tok, row):
    messages = [
        {"role" : "system", "content": SYSTEM},
        {"role" : "user", "content" : row["message"]}
    ]
    return tok.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

def encode(tok, row, max_len):
    prompt = build_prompt(tok, row)
    answer = json.dumps(row["target"]) + "<|im_end|>"

    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]
    return torch.tensor(ids), torch.tensor(labels)


class LoRA(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)

        self.scale = alpha / r

        out_f, in_f = base.weight.shape
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, in_f, device=dev) / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=dev))

    def forward(self, x):
        delta = (x.float() @ self.A.T @ self.B.T) * self.scale
        return self.base(x) + delta.to(x.dtype)

    def merge(self):
        w = self.base.weight.float() + self.scale * (self.B @ self.A)
        return w.to(self.base.weight.dtype)


def add_lora(model, r, alpha):
    model.requires_grad_(False)

    found = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if name in TARGETS and isinstance(child, nn.Linear):
                found.append((parent, name, child))

    mods = []
    for parent, name, child in found:
        lo = LoRA(child, r, alpha)
        setattr(parent, name, lo)
        mods.append((parent, name, lo))
    return mods

def merge(mods):
    for parent, name, lo in mods:
        lin = lo.base
        with torch.no_grad():
            lin.weight.copy_(lo.merge())
        setattr(parent, name, lin)

def loss_on(model, ids, labels):
    ids = ids.unsqueeze(0).cuda()
    labels = labels.cuda()
    hidden = model.model(input_ids=ids).last_hidden_state[0]
    keep = labels[1:] != -100
    logits = model.lm_head(hidden[:-1][keep]).float()
    return nn.functional.cross_entropy(logits, labels [1:][keep])

def lr_lambda(step, total, warmup = 10):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return .5 * (1 + math.cos(math.pi * progress))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1400)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(vars(args))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    row = load_rows(args.data)[: args.limit]
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = [encode(tok, r, args.max_len) for r in row]

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype= torch.float16).to("cuda")

    mods = add_lora(model, args.r, args.alpha)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{len(mods)} LoRA modules, {sum(p.numel() for p in params):,} trainable, {len(data)} examples")

    total = args.epochs * len(data) // args.accum
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total))
    scaler = torch.amp.GradScaler("cuda")

    seen = 0
    step = 0
    run_loss = 0.0
    t0 = time.time()

    for ep in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)

        for j in order:
            ids, lables = data[j]
            loss = loss_on(model, ids, lables)
            scaler.scale(loss / args.accum).backward()
            run_loss += loss.item()
            seen += 1

            if seen % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                if step % 5 == 0 or step == total:
                    mins = (time.time() - t0) / 60
                    vram = torch.cuda.max_memory_allocated() / 2**30
                    print(f"ep {ep} step {step}/{total} loss {run_loss / args.accum:.4f} "f"lr {sched.get_last_lr()[0]:.2e} {mins:.1f} min vram {vram:.2f}G", flush=True)

                run_loss = 0.0

    merge(mods)
    model.config.use_cache = True
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved {args.out} ({(time.time() - t0) / 60:.1f} min)")



if __name__ == "__main__":
    main()
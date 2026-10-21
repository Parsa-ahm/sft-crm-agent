"""Spec for my_sft/trainer.py. CPU only, no model weights, tokenizer only.

    .venv/bin/pytest my_sft -q

Make them pass in file order. Each test states a property the trainer must have; none of them
tell you how to get there.
"""
import json
import math
import os

import pytest
import torch
import torch.nn as nn

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_DATA = os.path.join(ROOT, "data", "final", "sft.jsonl")

trainer = pytest.importorskip("trainer", reason="write my_sft/trainer.py first")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    if not os.path.isdir(MODEL):
        pytest.skip(f"no tokenizer at {MODEL}; set MODEL")
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def row():
    with open(SFT_DATA) as f:
        return json.loads(f.readline())


# --- 1. encoding and the label mask ----------------------------------------------------------

def test_encode_returns_aligned_1d_long_tensors(tok, row):
    ids, labels = trainer.encode(tok, row, 1400)
    assert ids.dtype == labels.dtype == torch.long
    assert ids.ndim == labels.ndim == 1
    assert len(ids) == len(labels)
    assert len(ids) > 50


def test_encode_masks_every_prompt_token_and_nothing_else(tok, row):
    ids, labels = trainer.encode(tok, row, 1400)
    kept = (labels != -100).nonzero().flatten().tolist()
    assert kept, "no supervised tokens: the whole sequence is masked"
    assert kept == list(range(kept[0], len(ids))), "supervised positions must be one tail run"
    assert kept[0] > 20, "prompt tokens are being trained on"
    assert (labels[kept] == ids[kept]).all(), "labels must be the ids at supervised positions"


def test_encode_supervises_the_target_json_and_the_stop_token(tok, row):
    ids, labels = trainer.encode(tok, row, 1400)
    answer = tok.decode(labels[labels != -100])
    assert answer.endswith("<|im_end|>"), "model will never learn to stop"
    assert json.loads(answer[: -len("<|im_end|>")]) == row["target"]


def test_encode_respects_max_len(tok, row):
    ids, labels = trainer.encode(tok, row, 64)
    assert len(ids) == len(labels) == 64


def test_encode_puts_the_message_in_a_user_turn(tok, row):
    ids, _ = trainer.encode(tok, row, 1400)
    text = tok.decode(ids)
    assert "<|im_start|>user" in text and "<|im_start|>assistant" in text
    assert row["message"][:40] in text


# --- 2. the LoRA module ----------------------------------------------------------------------

def test_lora_is_identity_at_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    lo = trainer.LoRA(base, r=4, alpha=8)
    assert torch.allclose(lo(x), base(x), atol=1e-5), "B must start at zero"


def test_lora_params_are_trainable_and_base_is_not():
    base = nn.Linear(16, 8)
    base.weight.requires_grad_(False)
    base.bias.requires_grad_(False)
    lo = trainer.LoRA(base, r=4, alpha=8)
    trainable = [n for n, p in lo.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["A", "B"]
    assert lo.A.shape == (4, 16) and lo.B.shape == (8, 4)


def test_lora_a_init_scales_with_fan_in():
    """Same r, wider input -> smaller A. A fixed 0.02 std will fail this."""
    torch.manual_seed(0)
    narrow = trainer.LoRA(nn.Linear(64, 8), r=4, alpha=8).A.std().item()
    wide = trainer.LoRA(nn.Linear(1024, 8), r=4, alpha=8).A.std().item()
    assert wide < narrow / 2


def test_lora_applies_alpha_over_r_scaling():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    lo = trainer.LoRA(base, r=4, alpha=8)
    with torch.no_grad():
        lo.B.copy_(torch.randn_like(lo.B))
    delta = lo(x) - base(x)
    expected = (x @ lo.A.T @ lo.B.T) * (8 / 4)
    assert torch.allclose(delta, expected, atol=1e-5)


def test_merged_weight_reproduces_the_lora_forward():
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    lo = trainer.LoRA(base, r=8, alpha=16)
    with torch.no_grad():
        lo.B.copy_(torch.randn_like(lo.B) * 0.1)
    before = lo(x)
    w = lo.merged()
    assert w.shape == base.weight.shape and w.dtype == base.weight.dtype
    after = nn.functional.linear(x, w, base.bias)
    assert torch.allclose(before, after, atol=1e-4)


# --- 3. wrapping a model and putting it back ------------------------------------------------

class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()
        self.norm = nn.LayerNorm(8)   # not a target


class _Toy(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(n))
        self.lm_head = nn.Linear(8, 32)   # not a target

    def forward(self, x):
        for l in self.layers:
            h = l.self_attn.o_proj(l.self_attn.q_proj(x))
            x = l.mlp.down_proj(l.mlp.up_proj(l.norm(h)))
        return self.lm_head(x)


def test_add_lora_wraps_every_target_projection_and_nothing_else():
    m = _Toy(n=2)
    mods = trainer.add_lora(m, r=4, alpha=8)
    assert len(mods) == 14, "7 target projections per layer x 2 layers"
    assert isinstance(m.layers[0].self_attn.q_proj, trainer.LoRA)
    assert isinstance(m.layers[1].mlp.down_proj, trainer.LoRA)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, trainer.LoRA)
    assert isinstance(m.layers[0].norm, nn.LayerNorm)


def test_add_lora_does_not_change_the_forward_pass():
    torch.manual_seed(0)
    m = _Toy()
    x = torch.randn(2, 8)
    before = m(x)
    trainer.add_lora(m, r=4, alpha=8)
    assert torch.allclose(m(x), before, atol=1e-5)


def test_merge_restores_plain_linears_and_keeps_the_trained_function():
    torch.manual_seed(0)
    m = _Toy()
    x = torch.randn(2, 8)
    mods = trainer.add_lora(m, r=4, alpha=8)
    for _, _, lo in mods:
        with torch.no_grad():
            lo.B.copy_(torch.randn_like(lo.B) * 0.05)
    trained = m(x)
    trainer.merge(mods)
    assert type(m.layers[0].self_attn.q_proj) is nn.Linear, "LoRA wrappers must be gone"
    assert torch.allclose(m(x), trained, atol=1e-3)
    assert not any(isinstance(mod, trainer.LoRA) for mod in m.modules())


def test_merged_model_has_no_lora_keys_in_its_state_dict():
    m = _Toy()
    mods = trainer.add_lora(m, r=4, alpha=8)
    trainer.merge(mods)
    assert not [k for k in m.state_dict() if k.endswith((".A", ".B"))]


# --- 4. the schedule --------------------------------------------------------------------------

def test_lr_schedule_warms_up_then_decays_to_zero():
    """trainer.lr_lambda(step, total) -> multiplier on the base lr."""
    f = trainer.lr_lambda
    total = 100
    assert f(0, total) < 0.3, "step 0 must be warmed down, not full lr"
    assert f(0, total) < f(5, total) <= f(9, total), "warmup must rise"
    assert f(9, total) == pytest.approx(max(f(s, total) for s in range(total)), rel=0.05)
    assert f(total // 2, total) < f(9, total), "cosine must decay after warmup"
    assert f(total, total) == pytest.approx(0.0, abs=1e-6)
    assert all(not math.isnan(f(s, total)) for s in range(total + 5))

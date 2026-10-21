# Assignment: write the SFT trainer yourself

Prepared 2026-10-15. Implementation, experiments and conclusions are yours. The tests in
`test_my_sft.py` are the spec; no implementation code in this folder comes from an assistant.

## Goal

Train Qwen2.5-1.5B-Instruct on `data/final/sft.jsonl` (300 labeled rows) so that
`probe_crm.py` scores it within ~2 points of the reference run: **97.0% greedy full pass on
val 200**. Then do the same with `sft_all.jsonl` (3100 rows, 2 epochs) and reach ~99.5%.

You write `my_sft/trainer.py`. Deliverable is a merged HF checkpoint at `ckpt/my_sft300`
that `probe_crm.py` loads like any other Qwen2 model.

## In scope for you

- LoRA module: A/B parameters, init, scaling, forward, and the merge back into base weights.
- Prompt/answer encoding and the label mask (loss on the answer tokens only).
- The training loop: gradient accumulation, clipping, LR warmup + cosine, fp16 GradScaler.
- Saving the merged checkpoint and tokenizer.

## Allowed, do not rebuild

`AutoModelForCausalLM` / `AutoTokenizer` loading, the tokenizer itself, autograd, `AdamW`,
`nn.Linear`, `cross_entropy`, `LambdaLR`, `GradScaler`, `save_pretrained`. Not the model
architecture, not the tokenizer, not a data generator. Do not install `peft`; the LoRA is the
point of the exercise.

## Order of work

1. Read 5 rows of `data/final/sft.jsonl` and the `SYSTEM` prompt in `probe_crm.py`. Write on
   paper what one training example looks like as token ids and as labels, with lengths.
2. `.venv/bin/pytest my_sft -q`. All fail. Make them pass one at a time, in file order.
3. CPU smoke run: 20 rows, 1 epoch, `--max-len 512`, loss should fall. No GPU yet.
4. Full run on GPU. Reference run for comparison: 18.5M trainable params at r=16
   (196 wrapped modules), peak VRAM 3.9 GiB, 13.4 min for 300 rows x 3 epochs on the 2070,
   final loss ~0.002.
5. `probe_crm.py --model ckpt/my_sft300 --k 8 --batch 16`. Compare with
   `probe_crm_sft300.log`.
6. Only after the probe matches: diff your `trainer.py` against `../sft.py` and write down
   every difference and whether it mattered.

## Exit gate

- `pytest my_sft -q` green.
- Greedy full pass on val within 2 points of 97.0%, hallucinated fields 0.0%.
- A short report in `my_sft/REPORT.md`: prediction, observation, explanation, uncertainty,
  one next question. Include the run config (seed, lr, r, alpha, accum, epochs, precision,
  peak VRAM, wall time).

## Think about

- Why is the loss masked on the prompt instead of training on the whole sequence?
- `B` starts at zeros. What would break if both `A` and `B` were random?
- Why is `alpha/r` a scale and not a second learning rate?
- The base is fp16 and the LoRA params are fp32. Which op would overflow if A and B were fp16?
- Accumulating 8 single-example backward passes vs one batch of 8: when are the gradients
  equal, and when are they not?
- 300 rows moved the model from 7% to 97%. What does that say about what SFT taught it?

## Read

- Video: Umar Jamil, "LoRA: Low-Rank Adaptation of LLMs - explained visually + PyTorch code"
  (YouTube). Watch the math section before writing the LoRA class.
- Paper, section 4 only: LoRA, arXiv 2106.09685 (the reparametrization and the scaling).
- Reference file, read AFTER your first attempt: `../rl-lisp-python` SFT/training loop, then
  `./sft.py`.
- Search terms when stuck: `GradScaler unscale_ clip_grad_norm_ order`,
  `cross_entropy ignore_index -100`, `gradient checkpointing enable_input_require_grads`.

## Commands

    .venv/bin/pytest my_sft -q
    .venv/bin/python my_sft/trainer.py --model $MODEL --data data/final/sft.jsonl --out ckpt/my_sft300
    .venv/bin/python probe_crm.py --model ckpt/my_sft300 --k 8 --batch 16 --out data/probe_my_sft300.jsonl

# crm-extract-rl: handoff notes (2026-10-12)

RL post-training a small Qwen to run the paperwork for a one-person used-parts broker
(motivating case: a small parts broker in Ontario). Inbound text/email from a customer, junkyard or freight
carrier -> model outputs JSON: extracted CRM fields + `next_action` + a `draft` reply.
Owner approves every draft. Customer new-vs-existing matching is a deterministic n8n step
(email/phone/ref in CRM -> attach, else create), NOT the model.

## Where things are
- This folder on pop: `~/Sync/Documents/GitHub/resume-projects/crm-extract-rl/`
- `records.py`   gold-record builder. Python decides the truth first. Vehicles + engine codes from
                 a public parts-fitment table (`vehicle-overrides.json`);
                 everything else synthetic.
- `gen_nim.py`   NIM writer + verifier (current). `gen_nim_v1_used_for_raw.py` = the version that
                 produced most raw rows (same prompts; v2 adds make aliases + retry on HTTP 529).
- `finalize.py`  re-verifies with current verifier, renames miles->mileage, splits, adds SFT targets.
- `finish.sh`    unattended: waits for generation, refills API-error rows, runs finalize. Log: `finish.log`.
- `qc.py`        quality report: `python3 qc.py data/train_raw.jsonl --from 0 --to 400 --samples 5`
- `data/train_raw.jsonl`  Gemma 4 31B (`google/gemma-4-31b-it`), 3300 records, seed 0
- `data/test_raw.jsonl`   Kimi K3 (`moonshotai/kimi-k3`), 330 records, seed 999 (held-out, different generator)
- `data/final/{sft,val,rl,test}.jsonl`  300 / 200 / 2800 / 330 (built; all 3300 Gemma + 330 Kimi rows passed the final verifier)
- `probe.py`, `gen.py`, `probe_*.log`, `data/cases.jsonl`, `data/probe_*.jsonl`: OLD Olist-shipping
  headroom probe (superseded task). Kept for the numbers below.
- NIM key: `~/.config/nim/api-key` (never print/commit it).

## First thing at home
    cd ~/Sync/Documents/GitHub/resume-projects/crm-extract-rl
    tail -n 20 finish.log          # DONE 10-12: splits built, see counts below
    wc -l data/final/*.jsonl
    python3 -c "import json; r=json.loads(open('data/final/sft.jsonl').readline()); print(r['message']); print(json.dumps(r['target'], indent=1))"

## Final row format (data/final/*.jsonl)
`message` (what the model sees), `gold` (23 extraction fields), `next_action`
(ask_yard | quote_customer | update_customer | reply_customer), `draft_to` (yard|customer),
`draft_require` / `draft_forbid` (gold field names the draft must / must not contain),
`target` (full reference output: gold + next_action + templated draft; used for SFT only).

Business rules baked into the drafts: quote to customer uses `[PRICE]` placeholder and never
reveals yard cost or yard name; request to yard never reveals customer budget or contact.

## Message kinds (weights)
quote_request 34, yard_offer 24, freight_update 14, where_is_my_part 13, warranty_claim 9, cancel_request 6.
Traps: engine code only when stated (else null), positions (front left vs right), budget vs mileage,
casual numbers (80k, $1,400), lowercase/typos, SMS vs email, shop vs individual.

## QC log (every ~400 train rows)
- Verifier acceptance ~100% all slices; 3 HTTP 529 errors refilled by finish.sh; 7 Kimi timeouts refilled.
- Bug found by reading samples, fixed mid-run: writer invented sign-off names (header "Wei Wilson",
  signed "Jameson") because sender name wasn't in its facts. All affected customer-email rows regenerated.
- Persona hint parroting ("not a mechanic" in 12/40) fixed before the full run.
- Kimi writes "mercedes"/"vw"; verifier got make aliases (false rejects recovered in finalize).
- Canada uses km -> fields renamed max_mileage/mileage in final data.
- Known weaknesses: Gemma's chatty yard offers repeat one backstory template (watch per-kind test
  score); random years make some non-existent cars (2016 GR86). Labels still correct.
- `vehicle-overrides.json` Civic record says "K20C2 (Si)": likely wrong (2017-20 Si = L15B7);
  excluded here.

## Old headroom probe (Olist shipping task, superseded but informative)
Qwen2.5-1.5B-Instruct: greedy exact 25%, 33% cases hallucinate a field, k=8 mixed groups 38%,
all-wrong 61%. 0.5B: exact 5%, mixed 2%, all-wrong 98% -> too small. Training target = 1.5B
(`../rl-lisp-python/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct`).

## Next steps
1. Re-probe 1.5B headroom on data/final/val.jsonl (new task) - adapt probe.py to the new schema.
2. Reward function (Parsa writes): JSON gate -> per-field match -> next_action -> draft checks
   (required facts present, forbidden facts absent, no numbers/IDs not in source, right recipient).
3. SFT warm-up on sft.jsonl, then GRPO with Parsa's loop from ../rl-lisp-python, dynamic sampling.
4. Eval on test.jsonl: base vs SFT vs SFT+RL (+ a frontier model), per kind; latency + joules/message.

## 100-sample audit (2026-10-12, `audit.py`)
Read 100 random rows across all splits. Fixed (130 rows regenerated, all checks now 0):
- Kimi SMS labeled "no name" introduced a name anyway ("hey its mike...") in 12/29 test SMS -> writer told
  not to, verifier now rejects it.
- 55+ warranty claims with impossible issues ("ECU leaking oil") -> issue constrained per part (ISSUES_FOR).
- 17 EVs requested engine/transmission -> remapped. Remaps are deterministic so no other label changed.
Pre-audit raw files kept as data/*_raw.pre_audit.jsonl. Splits re-shuffled (nothing trained yet).
Known, accepted: some year/model combos never existed (2003 Cadillac XT6); yard emails repetitive (fine).
Reward-design lessons from audit: compare dates as dates (draft "May 7" vs gold 2026-05-07), and match
numbers as whole tokens (cost 1200 "appears" inside mileage 112,000 with substring matching).
Project folder is Syncthing-ignored on both machines (pop-only); backup copy at /mnt/data/crm-extract-rl-backup-20261012.

## Headroom re-probe, new task (2026-10-14, `probe_crm.py`, log `probe_crm_1.5B.log`)
Qwen2.5-1.5B-Instruct, val 200, k=8, T=1. Probe scorer, NOT the reward. 50 min, peak 4.35 GiB.
- greedy: JSON valid 100%, all 23 fields 22%, next_action 95%, draft ok 24.5%, full pass 7%.
- 59% of cases hallucinate a field (sender_name 68, engine_code 62, max_mileage 57).
- draft fails: wrong_recipient 62, no [PRICE] 44, leaked yard_cost 37, yard_name 25, budget 21.
- yard_offer: 0% greedy. Base model passes yard cost straight to the customer.
- sampled binary reward: extract mixed 15%, full mixed 1%, all-wrong 99% -> binary reward = no gradient.
- T=1 mean pass 2.2% vs greedy 22%: T=1 too hot for rollouts, use ~0.7.
- batch 32 x ~1000-token prompts OOMs in HF prefill; GRPO group 16 will not fit as one batch.
Conclusion: large headroom. SFT warm-up mandatory; reward needs partial credit (per-field
fraction + draft checks), not all-or-nothing.

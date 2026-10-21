# sft-crm-agent

**Fine-tuning Qwen2.5-1.5B to use as the model for an agentic workflow that takes text inputs from email and text and adds it to a custom CRM and drafts future responses.**

## Abstract

A one-person used auto parts broker receives emails and text messages from customers, junkyards and
freight carriers. Each message has to be filed as a CRM record and answered. We frame this as a single
structured-generation task: one inbound message in, one JSON object out, holding 23 extracted fields,
a next action, and a draft reply that must follow two confidentiality rules (customers never see the
yard's price or name; yards never see the customer's budget). We fine-tune Qwen2.5-1.5B-Instruct with
LoRA on 300 synthetic examples, using a trainer and evaluator written from scratch. On 330 held-out
messages written by a different model than the training data, the fine-tuned model passes 97.0% of
messages end to end (all fields, next action and draft correct), against 4.2% before fine-tuning,
49.1% for gpt-oss-20b zero-shot, 84.8% for gpt-oss-20b with six in-context examples, and 84.2% for
Nemotron-3.5-Lightning (30B-A3B). It invents a field value on 0.3% of messages. Training takes 10.3
minutes on one 8 GB consumer GPU.

## 1. Task

**Input.** One inbound message: an email (with `From`, `Date`, `Subject` header) or an SMS (with
sender number and date). Six kinds occur:

| kind | sender | next_action | draft goes to |
|---|---|---|---|
| `quote_request` | customer asks for a part | `ask_yard` | yard |
| `yard_offer` | yard offers a part with price | `quote_customer` | customer |
| `freight_update` | carrier reports a pickup | `update_customer` | customer |
| `where_is_my_part` | customer asks about an order | `reply_customer` | customer |
| `warranty_claim` | customer reports a defect | `reply_customer` | customer |
| `cancel_request` | customer cancels an order | `reply_customer` | customer |

**Output.** One JSON object with exactly these keys:

- 23 fields: `intent`, `sender_name`, `sender_contact`, `sender_type`, `ref`, `vehicle_year`,
  `vehicle_make`, `vehicle_model`, `engine_code`, `part`, `part_position`, `max_mileage`, `budget`,
  `ship_to_postal`, `yard_name`, `part_number`, `mileage`, `yard_cost`, `warranty_days`, `carrier`,
  `tracking_number`, `eta`, `issue`. Any field the message does not state must be `null`.
- `next_action`: determined by the kind (table above).
- `draft`: `{"to": "yard" | "customer", "body": "..."}`, a short reply the broker approves before
  sending.

**Draft rules.** A request to a yard states the vehicle, part, position, engine code and maximum
mileage when given, and never the customer's budget or contact details. An offer to a customer states
part number, mileage and warranty, writes the price as the literal `[PRICE]` (the broker sets the
markup), and never states the yard's cost or name. A freight update states carrier, PRO number and
delivery date. Replies about an existing order mention the order number. No draft may contain a
number, price, date or ID that is not in the inbound message.

The full instruction given to every model is `SYSTEM` in `train.py` (3,133 characters).

## 2. Data

All data is synthetic, generated so that labels are known by construction rather than annotated.

1. **Gold record first.** A Python generator samples a record: message kind (weights 34/24/14/13/9/6
   for the kinds in table order), sender, vehicle, part and all field values. Vehicle make, model and
   engine codes come from a real parts-fitment table (`data/vehicle-overrides.json`); names, prices,
   order numbers, PRO numbers and postal codes are random. The generator also decides which facts the
   message states, the draft recipient, which gold fields the draft must contain
   (`draft_require`) and which it must not (`draft_forbid`).
2. **Message second.** An LLM is given the record's facts and a writing style (email or SMS, tone,
   typos, digressions, casual numbers such as "80k" or "$1,400") and writes only the message text.
3. **Verification.** A rule-based verifier rejects any message that drops a stated fact, states a
   number not in the record, or adds a name to a message the record marks as unsigned. Rejected rows are
   regenerated. A manual audit of 100 random rows led to 130 regenerated rows (impossible defects such
   as a leaking ECU, engines requested for electric vehicles, names added to anonymous SMS).
4. **Target.** For training, the target JSON is the gold record plus a templated draft that satisfies
   the draft rules by construction.

| split | rows | message writer | use |
|---|---|---|---|
| `data/sft.jsonl` | 300 | Gemma 4 31B (`google/gemma-4-31b-it`) | training |
| `data/val.jsonl` | 200 | Gemma 4 31B | development; all tuning decisions |
| `data/test.jsonl` | 330 | Kimi K3 (`moonshotai/kimi-k3`), different seed | final numbers only |
| `data/rl.jsonl` | 2,800 | Gemma 4 31B | unused here (reserved for RL) |

`data/sft_all.jsonl` is `sft` + `rl`. `data/train_raw.jsonl` and `data/test_raw.jsonl` hold the raw
generations with verifier verdicts.

The test set is written by a different model family than the training and validation sets, so it
measures whether the fine-tuned model learned the task or the training writer's phrasing. Test
composition: 104 quote requests, 74 yard offers, 64 where-is-my-part, 38 freight updates, 30
warranty claims, 20 cancellations; 263 emails and 67 SMS; median 639 characters.

## 3. Method

### 3.1 Fine-tuning

`train.py` implements supervised fine-tuning without peft or a Trainer class:

- **LoRA** with rank 16 and alpha 32 on all seven linear projections (`q, k, v, o, gate, up, down`)
  in each of the 28 blocks: 196 adapted modules, 18.5M trainable parameters. Base weights are frozen
  in fp16; adapter weights are fp32.
- **Loss** on answer tokens only: the prompt (system instruction + message, in the Qwen chat
  template) is masked with label -100; the answer is the target JSON followed by `<|im_end|>`.
  Sequences are capped at 1,400 tokens.
- **Optimisation.** AdamW, learning rate 2e-4, 10 warmup steps then cosine decay to 0, gradient
  clipping at 1.0, fp16 loss scaling. Batch size 1 with 8-step gradient accumulation; 3 epochs over
  300 rows gives 112 optimiser updates. Seed 0.
- **Export.** After training, each adapter is merged into its base weight
  (`W + (alpha / r) * B @ A`) and the model is saved as a plain Hugging Face checkpoint.

On an RTX 2070 Super (8 GB) training takes 10.3 minutes with 3.78 GB peak memory.

### 3.2 Evaluation

`generate.py` runs a model over a split and saves its raw outputs; local models decode greedily
(fp16, batch 16, up to 512 new tokens). `eval.py` scores each output against its gold record. Typographic
hyphens and spaces (U+2010–U+2013, U+00A0, U+202F) are mapped to ASCII, then the first `{` to the last
`}` of the output is parsed as JSON. Then:

- **Fields.** All 23 fields are normalised before comparison: integers are parsed from text ("62k",
  "$1,400"); IDs and postal codes ignore spaces, hyphens and case; phone numbers compare on their last
  ten digits; dates compare as ISO dates; common make abbreviations map to the full name (`VW` →
  `Volkswagen`, `Mercedes` → `Mercedes-Benz`, `Chevy` → `Chevrolet`); strings compare case-insensitively; `""`, `"none"`,
  `"n/a"`, `"unknown"` count as null. A field is **hallucinated** if gold is null and the prediction
  is not, **omitted** in the reverse case.
- **next_action** must match exactly.
- **Draft.** Fails if the recipient is wrong, a `draft_require` field is absent from the body, a
  `draft_forbid` field is present, a yard offer lacks `[PRICE]`, or the body contains any number
  greater than 31 that does not appear in the inbound message. Numbers are matched as whole tokens
  (a glued unit still ends the number: `90,000km` is 90000), dates in several written forms ("May 7",
  "May. 7", "5/7"), makes with their abbreviations, parts through a synonym list ("tranny" for
  transmission).
- **Full pass** requires all 23 fields, `next_action` and the draft to be correct.

Before the unit, hyphen and make-alias handling was added, the scorer was checked against an
independently written reference scorer: identical verdicts on 800/800 outputs from four models. Adding
those three rules turned 13 failures into passes across all systems (fine-tuned 1, gpt-oss 0-shot 1,
gpt-oss 6-shot 4, Nemotron 7) and no pass into a failure.

### 3.3 Baselines

| system | parameters | access |
|---|---|---|
| Qwen2.5-1.5B-Instruct, zero-shot | 1.5B | local, same settings as the fine-tuned model |
| gpt-oss-20b, zero-shot | 21B (3.6B active) | NVIDIA NIM API, `reasoning_effort=low` |
| gpt-oss-20b, 6-shot | 21B (3.6B active) | as above, plus one training example per kind as prior chat turns |
| Nemotron-3.5-Lightning | 30B (3B active) | NVIDIA NIM API, reasoning on |

All baselines receive the same system instruction and are decoded at temperature 0. API models get
1,024 output tokens, except Nemotron, which needs 16,384 because its reasoning trace counts against
the limit. Few-shot examples come from `sft.jsonl` only. `bench.py` scores every system's run file with the same `eval.py`.

## 4. Results

All numbers are on the 330-message held-out test set. Single run per system.

### 4.1 Main results

| system | full pass | 95% CI | all fields | next_action | draft ok | valid JSON | ≥1 hallucinated field | prompt tokens |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B + SFT (ours)** | **97.0%** | 94.5–98.3 | **97.6%** | **100%** | **99.1%** | 100% | **0.3%** | 976 |
| Qwen2.5-1.5B-Instruct | 4.2% | 2.5–7.0 | 18.2% | 91.5% | 20.0% | 100% | 54.8% | 976 |
| gpt-oss-20b, zero-shot | 49.1% | 43.7–54.5 | 77.3% | 97.0% | 62.4% | 97.0% | 2.4% | 1,002 |
| gpt-oss-20b, 6-shot | 84.8% | 80.6–88.3 | 94.8% | 100% | 89.4% | 100% | 0.6% | 3,172 |
| Nemotron-3.5-Lightning | 84.2% | 79.9–87.8 | 88.2% | 99.7% | 96.1% | 100% | 4.5% | 1,014 |

95% CI: Wilson interval on full pass. Prompt tokens: mean input length per message in each model's
own tokenizer. Mean output length: 238 tokens for the fine-tuned model; 343 (gpt-oss zero-shot), 242
(gpt-oss 6-shot) and 4,027 (Nemotron) completion tokens for the API models, including reasoning.

On paired outcomes, the fine-tuned model passes 49 messages that gpt-oss-20b 6-shot fails and fails
9 that it passes (exact McNemar p = 9e-8); against Nemotron the split is 50 to 8 (p = 1.6e-8).

### 4.2 Full pass by message kind

| kind | n | SFT (ours) | base | gpt-oss 0-shot | gpt-oss 6-shot | Nemotron |
|---|---|---|---|---|---|---|
| quote_request | 104 | 96.2% | 7.7% | 67.3% | 67.3% | 81.7% |
| yard_offer | 74 | 97.3% | 0.0% | 5.4% | 89.2% | 94.6% |
| where_is_my_part | 64 | 96.9% | 0.0% | 68.8% | 93.8% | 79.7% |
| freight_update | 38 | 100% | 10.5% | 34.2% | 94.7% | 84.2% |
| warranty_claim | 30 | 96.7% | 3.3% | 63.3% | 100% | 86.7% |
| cancel_request | 20 | 95.0% | 5.0% | 60.0% | 90.0% | 70.0% |

### 4.3 Where each system fails

Counts are messages (out of 330) with each failure; a message can have several.

- **Base Qwen2.5-1.5B** fails on almost everything. It invents `sender_name` (110 messages),
  `max_mileage` (90) and `engine_code` (75), addresses 113 drafts to the wrong party, and in yard
  offers writes the yard's price into the customer draft (63) instead of `[PRICE]` (74).
- **gpt-oss-20b zero-shot** extracts well but breaks the confidentiality rules: 70 yard-offer drafts
  lack `[PRICE]` and 40 state the yard's cost, which is why it passes 5.4% of yard offers. It also
  mislabels `sender_type` (50) and produces malformed JSON on 10 messages (all ten drop the final
  closing brace).
- **gpt-oss-20b 6-shot** fixes the yard offers but still tells the yard the customer's budget in 28
  quote-request drafts, which accounts for most of its quote-request failures (67.3%).
- **Nemotron-3.5-Lightning** follows the draft rules but extracts less reliably: `sender_type` (20)
  and `sender_name` (16) errors, and 15 messages with an invented field. Fourteen of the 15 come from
  repair shops: it fills `sender_name` on unsigned shop texts or treats the shop as a yard
  (`yard_name`).
- **Fine-tuned model (10 failures).** Missed budget stated as "try to keep it under 2000 bucks";
  `"bumper"` instead of the canonical `"front bumper"`; copied an email
  address with one letter dropped; confused `shop` and `individual` (once in each direction); wrote a
first name only (`"Dave"` for `"Dave Ali"`); `wrong_part` instead of
  `damaged` for a part that "showed up cracked"; a draft that left out the vehicle year (twice) or
  model and mileage limit; one invented postal code. Five of the 10 are passed by all three API
  baselines.

## 5. Examples

Five test messages, chosen to show one failure pattern each; the last is one the fine-tuned model
gets wrong. Every system's verbatim output is in the collapsed blocks (parsed and re-indented when
valid JSON). The result column lists every check that failed.

### 1. Yard offer (email, test id 64)

```
From: Brantford Auto Dismantlers <sales@brantfordautodismantle.ca>
Date: 2026-05-27
Subject: Re: Request JA-3791

Got your ECU. Came off a 2015 Kia Optima, 171k on the donor.

Stock number IN935794. Bench tested, working.

Price to you: $2400, plus freight if you need it shipped.

Let me know and I'll set it aside.

Brantford Auto Dismantlers
```

| model | result |
|---|---|
| Qwen2.5-1.5B + SFT (this work) | pass |
| Qwen2.5-1.5B-Instruct, zero-shot | `sender_name` "Sales Team at Brantford Auto Dismantlers" (gold null); `engine_code` "ECU" (gold null); `max_mileage` 171000 (gold null); `yard_cost` null (gold 2400); draft omits `mileage`; draft reveals `yard_cost`; draft reveals `yard_name`; no `[PRICE]` placeholder |
| gpt-oss-20b, zero-shot | draft reveals `yard_cost`; no `[PRICE]` placeholder |
| gpt-oss-20b, 6-shot | draft reveals `yard_cost`; no `[PRICE]` placeholder |
| Nemotron-3.5-Lightning, zero-shot | pass |

<details><summary>Qwen2.5-1.5B + SFT (this work): output</summary>

```json
{
  "intent": "yard_offer",
  "sender_name": null,
  "sender_contact": "sales@brantfordautodismantle.ca",
  "sender_type": "yard",
  "ref": "JA-3791",
  "vehicle_year": 2015,
  "vehicle_make": "Kia",
  "vehicle_model": "Optima",
  "engine_code": null,
  "part": "ecu",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Brantford Auto Dismantlers",
  "part_number": "IN935794",
  "mileage": 171000,
  "yard_cost": 2400,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "quote_customer",
  "draft": {
    "to": "customer",
    "body": "Hi, good news: we found an ecu from a 2015 Kia Optima, part number IN935794, with 171,000 on it. Price: [PRICE]. Let us know if you'd like to go ahead and we'll arrange shipping."
  }
}
```
</details>

<details><summary>Qwen2.5-1.5B-Instruct, zero-shot: output</summary>

```json
{
  "intent": "yard_offer",
  "sender_name": "Sales Team at Brantford Auto Dismantlers",
  "sender_contact": "sales@brantfordautodismantle.ca",
  "sender_type": "yard",
  "ref": "JA-3791",
  "vehicle_year": 2015,
  "vehicle_make": "Kia",
  "vehicle_model": "Optima",
  "engine_code": "ECU",
  "part": "ECU",
  "part_position": null,
  "max_mileage": 171000,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Brantford Auto Dismantlers",
  "part_number": "IN935794",
  "mileage": 171000,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "quote_customer",
  "draft": {
    "to": "customer",
    "body": "Hello,\n\nWe have received your request for the ECU from your 2015 Kia Optima. The stock number is IN935794.\n\nThe ECU has been bench tested and is working properly. The price to you is $2400, including freight if needed.\n\nPlease let us know if you require shipping.\n\nBest regards,\nThe Sales Team at Brantford Auto Dismantlers"
  }
}
```
</details>

<details><summary>gpt-oss-20b, zero-shot: output</summary>

```json
{
  "intent": "yard_offer",
  "sender_name": null,
  "sender_contact": "sales@brantfordautodismantle.ca",
  "sender_type": "yard",
  "ref": "JA-3791",
  "vehicle_year": 2015,
  "vehicle_make": "Kia",
  "vehicle_model": "Optima",
  "engine_code": null,
  "part": "ecu",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Brantford Auto Dismantlers",
  "part_number": "IN935794",
  "mileage": 171000,
  "yard_cost": 2400,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "quote_customer",
  "draft": {
    "to": "customer",
    "body": "Hello,\n\nWe have an ECU from a 2015 Kia Optima (stock number IN935794) with 171,000 miles, bench tested and working. The price is $2400, plus freight if you need it shipped.\n\nLet me know if you’d like to proceed.\n\nThank you."
  }
}
```
</details>

<details><summary>gpt-oss-20b, 6-shot: output</summary>

```json
{
  "intent": "yard_offer",
  "sender_name": null,
  "sender_contact": "sales@brantfordautodismantle.ca",
  "sender_type": "yard",
  "ref": "JA-3791",
  "vehicle_year": 2015,
  "vehicle_make": "Kia",
  "vehicle_model": "Optima",
  "engine_code": null,
  "part": "ecu",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Brantford Auto Dismantlers",
  "part_number": "IN935794",
  "mileage": 171000,
  "yard_cost": 2400,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "quote_customer",
  "draft": {
    "to": "customer",
    "body": "Hi, we have an ECU from a 2015 Kia Optima (171,000 miles, part number IN935794) available for $2400. Let us know if you'd like to reserve it and we can arrange shipping."
  }
}
```
</details>

<details><summary>Nemotron-3.5-Lightning, zero-shot: output</summary>

```json
{
  "intent": "yard_offer",
  "sender_name": null,
  "sender_contact": "sales@brantfordautodismantle.ca",
  "sender_type": "yard",
  "ref": "JA-3791",
  "vehicle_year": 2015,
  "vehicle_make": "Kia",
  "vehicle_model": "Optima",
  "engine_code": null,
  "part": "ecu",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Brantford Auto Dismantlers",
  "part_number": "IN935794",
  "mileage": 171000,
  "yard_cost": 2400,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "quote_customer",
  "draft": {
    "to": "customer",
    "body": "Part number IN935794, 171k miles. Price [PRICE]."
  }
}
```
</details>

### 2. Quote request (email, test id 83)

```
From: Timmins Auto - Wei Singh <weisingh15@gmail.com>
Date: 2026-10-13
Subject: Need a left taillight for a 2015 Honda CR-V – quote please

Hi,

We have a 2015 Honda CR-V sitting on a hoist at Timmins Auto and the customer is breathing down my neck. I need the left taillight and I need a quote fast. Used is fine as long as it's in decent shape and not cracked or faded to pieces.

My budget is $450 shipped to K7N 5E5. If you have one on the shelf or can pull one this week, please get back to me today with a price and how soon it can ship. I've already wasted too much time calling around on this.

Thanks,
Wei
```

| model | result |
|---|---|
| Qwen2.5-1.5B + SFT (this work) | pass |
| Qwen2.5-1.5B-Instruct, zero-shot | `sender_type` "individual" (gold "shop"); draft addressed to the wrong party; draft reveals `budget` |
| gpt-oss-20b, zero-shot | `sender_type` "individual" (gold "shop"); draft reveals `budget` |
| gpt-oss-20b, 6-shot | draft reveals `budget` |
| Nemotron-3.5-Lightning, zero-shot | pass |

<details><summary>Qwen2.5-1.5B + SFT (this work): output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": "Wei Singh",
  "sender_contact": "weisingh15@gmail.com",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2015,
  "vehicle_make": "Honda",
  "vehicle_model": "CR-V",
  "engine_code": null,
  "part": "taillight",
  "part_position": "left",
  "max_mileage": null,
  "budget": 450,
  "ship_to_postal": "K7N 5E5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi, I'm looking for a left taillight for a 2015 Honda CR-V. Do you have one? Please send the part number, mileage and your price. Thanks."
  }
}
```
</details>

<details><summary>Qwen2.5-1.5B-Instruct, zero-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": "Wei Singh",
  "sender_contact": "weisingh15@gmail.com",
  "sender_type": "individual",
  "ref": null,
  "vehicle_year": 2015,
  "vehicle_make": "Honda",
  "vehicle_model": "CR-V",
  "engine_code": null,
  "part": "taillight",
  "part_position": "left",
  "max_mileage": null,
  "budget": 450,
  "ship_to_postal": "K7N 5E5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "Timmins Auto",
    "body": "Dear Wei,\n\nI'm looking for a left taillight for my 2015 Honda CR-V. Could you provide a quote quickly? My budget is $450 shipped to K7N 5E5.\n\nThank you!\n\nBest regards,\nWei"
  }
}
```
</details>

<details><summary>gpt-oss-20b, zero-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": "Wei Singh",
  "sender_contact": "weisingh15@gmail.com",
  "sender_type": "individual",
  "ref": null,
  "vehicle_year": 2015,
  "vehicle_make": "Honda",
  "vehicle_model": "CR-V",
  "engine_code": null,
  "part": "taillight",
  "part_position": "left",
  "max_mileage": null,
  "budget": 450,
  "ship_to_postal": "K7N 5E5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi,\n\nCould you please provide a quote for a left taillight for a 2015 Honda CR‑V?  I need the part number, mileage, and price.  The customer’s budget is $450 and the shipment address is K7N 5E5.  Please let me know the availability and shipping time.\n\nThanks."
  }
}
```
</details>

<details><summary>gpt-oss-20b, 6-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": "Wei Singh",
  "sender_contact": "weisingh15@gmail.com",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2015,
  "vehicle_make": "Honda",
  "vehicle_model": "CR-V",
  "engine_code": null,
  "part": "taillight",
  "part_position": "left",
  "max_mileage": null,
  "budget": 450,
  "ship_to_postal": "K7N 5E5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi, I'm looking for a left taillight for a 2015 Honda CR‑V, no more than $450 including shipping to K7N 5E5. Do you have one? Please send the part number, mileage and your price. Thanks."
  }
}
```
</details>

<details><summary>Nemotron-3.5-Lightning, zero-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": "Wei Singh",
  "sender_contact": "weisingh15@gmail.com",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2015,
  "vehicle_make": "Honda",
  "vehicle_model": "CR-V",
  "engine_code": null,
  "part": "taillight",
  "part_position": "left",
  "max_mileage": null,
  "budget": 450,
  "ship_to_postal": "K7N 5E5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "We need a quote for a left taillight (left position) for a 2015 Honda CR-V. Please provide the part number, mileage, and price. Also let us know the engine code and max mileage acceptable. The part should be in decent shape (not cracked or faded)."
  }
}
```
</details>

### 3. Freight update (email, test id 30)

```
From: Purolator Freight <noreply@purolatorfreight.com>
Date: 2026-03-21
Subject: Pickup confirmation — ref JA-1159

This is Purolator Freight confirming your shipment was picked up this morning.

Reference: JA-1159
PRO number: 9175458347
Estimated delivery: March 27

We've had delays pile up and dispatch is on my case about paperwork, so please confirm someone will be there to receive on the estimated delivery date. I don't want this coming back to us.

Reply promptly so we can close this out.

Dispatch, Purolator Freight
```

| model | result |
|---|---|
| Qwen2.5-1.5B + SFT (this work) | pass |
| Qwen2.5-1.5B-Instruct, zero-shot | draft omits `tracking_number`; draft omits `carrier` |
| gpt-oss-20b, zero-shot | draft omits `carrier` |
| gpt-oss-20b, 6-shot | pass |
| Nemotron-3.5-Lightning, zero-shot | draft omits `carrier` |

<details><summary>Qwen2.5-1.5B + SFT (this work): output</summary>

```json
{
  "intent": "freight_update",
  "sender_name": null,
  "sender_contact": "noreply@purolatorfreight.com",
  "sender_type": "carrier",
  "ref": "JA-1159",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": null,
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": "Purolator Freight",
  "tracking_number": "9175458347",
  "eta": "2026-03-27",
  "issue": null,
  "next_action": "update_customer",
  "draft": {
    "to": "customer",
    "body": "Hi, your part has been picked up by Purolator Freight. PRO number 9175458347, estimated delivery March 27."
  }
}
```
</details>

<details><summary>Qwen2.5-1.5B-Instruct, zero-shot: output</summary>

```json
{
  "intent": "freight_update",
  "sender_name": null,
  "sender_contact": "noreply@purolatorfreight.com",
  "sender_type": "carrier",
  "ref": "JA-1159",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": null,
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": "Purolator Freight",
  "tracking_number": "9175458347",
  "eta": "2026-03-27",
  "issue": null,
  "next_action": "update_customer",
  "draft": {
    "to": "customer",
    "body": "Hello,\n\nYour shipment has been confirmed and dispatched.\nThe estimated delivery date is March 27.\nPlease ensure someone will be available at that time to receive the package.\nThank you."
  }
}
```
</details>

<details><summary>gpt-oss-20b, zero-shot: output</summary>

```json
{
  "intent": "freight_update",
  "sender_name": null,
  "sender_contact": "noreply@purolatorfreight.com",
  "sender_type": "carrier",
  "ref": "JA-1159",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": null,
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": "Purolator Freight",
  "tracking_number": "9175458347",
  "eta": "2026-03-27",
  "issue": null,
  "next_action": "update_customer",
  "draft": {
    "to": "customer",
    "body": "Thank you for the pickup confirmation. Your shipment with PRO number 9175458347 is scheduled for delivery on 2026-03-27. We will ensure someone is available to receive it. Please let us know if you have any further questions."
  }
}
```
</details>

<details><summary>gpt-oss-20b, 6-shot: output</summary>

```json
{
  "intent": "freight_update",
  "sender_name": null,
  "sender_contact": "noreply@purolatorfreight.com",
  "sender_type": "carrier",
  "ref": "JA-1159",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": null,
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": "Purolator Freight",
  "tracking_number": "9175458347",
  "eta": "2026-03-27",
  "issue": null,
  "next_action": "update_customer",
  "draft": {
    "to": "customer",
    "body": "Hi, your shipment (JA-1159) was picked up by Purolator Freight. PRO number 9175458347, estimated delivery March 27."
  }
}
```
</details>

<details><summary>Nemotron-3.5-Lightning, zero-shot: output</summary>

```json
{
  "intent": "freight_update",
  "sender_name": null,
  "sender_contact": "noreply@purolatorfreight.com",
  "sender_type": "carrier",
  "ref": "JA-1159",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": null,
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": "Purolator Freight",
  "tracking_number": "9175458347",
  "eta": "2026-03-27",
  "issue": null,
  "next_action": "update_customer",
  "draft": {
    "to": "customer",
    "body": "Your shipment with PRO 9175458347 is estimated to deliver on March 27. Please confirm someone will be available to receive."
  }
}
```
</details>

### 4. Warranty claim (email, test id 26)

```
From: Oshawa Garage - Rob Singh <rsingh@gmail.com>
Date: 2026-02-18
Subject: problem with my order JA-1191

hey, its Rob over at Oshawa Garage. we ordered a transmission from you (order JA-1191) and got it installed in a customer's car, but now its leaking oil or coolant or something, theres a puddle under the car everytime it sits. not sure if its the seal or what but the customer is pretty unhappy and we need to get this sorted. let me know what we can do, thanks

Rob Singh
```

| model | result |
|---|---|
| Qwen2.5-1.5B + SFT (this work) | pass |
| Qwen2.5-1.5B-Instruct, zero-shot | `sender_name` "Oshawa Garage - Rob Singh" (gold "Rob Singh"); `sender_type` "yard" (gold "shop"); `yard_name` "Oshawa Garage" (gold null); draft omits `ref` |
| gpt-oss-20b, zero-shot | draft omits `ref` |
| gpt-oss-20b, 6-shot | pass |
| Nemotron-3.5-Lightning, zero-shot | `sender_name` null (gold "Rob Singh"); `sender_type` "yard" (gold "shop"); `yard_name` "Oshawa Garage" (gold null) |

<details><summary>Qwen2.5-1.5B + SFT (this work): output</summary>

```json
{
  "intent": "warranty_claim",
  "sender_name": "Rob Singh",
  "sender_contact": "rsingh@gmail.com",
  "sender_type": "shop",
  "ref": "JA-1191",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": "leak",
  "next_action": "reply_customer",
  "draft": {
    "to": "customer",
    "body": "Hi Rob, sorry to hear about the problem with the transmission (order JA-1191). I'm opening a warranty claim with the supplier. Could you send a few photos and a short description of what you're seeing?"
  }
}
```
</details>

<details><summary>Qwen2.5-1.5B-Instruct, zero-shot: output</summary>

```json
{
  "intent": "warranty_claim",
  "sender_name": "Oshawa Garage - Rob Singh",
  "sender_contact": "rsingh@gmail.com",
  "sender_type": "yard",
  "ref": "JA-1191",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Oshawa Garage",
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": "leak",
  "next_action": "reply_customer",
  "draft": {
    "to": "customer",
    "body": "Dear Customer,\n\nWe have received your warranty claim regarding the leaky transmission on your vehicle. We understand how frustrating this must be.\n\nTo resolve this issue, we recommend replacing the entire transmission unit. This will ensure that the leak is completely eliminated and prevent future problems.\n\nPlease provide us with the necessary documentation such as the receipt and any other relevant information so we can proceed with the replacement.\n\nThank you for choosing our services.\nBest regards,\nYour Used Auto Parts Broker"
  }
}
```
</details>

<details><summary>gpt-oss-20b, zero-shot: output</summary>

```json
{
  "intent": "warranty_claim",
  "sender_name": "Rob Singh",
  "sender_contact": "rsingh@gmail.com",
  "sender_type": "shop",
  "ref": "JA-1191",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": "leak",
  "next_action": "reply_customer",
  "draft": {
    "to": "customer",
    "body": "Hi Rob, we’re sorry to hear about the leak. Please let us know the part number and mileage of the transmission, and we’ll arrange a replacement or repair under warranty. Thank you."
  }
}
```
</details>

<details><summary>gpt-oss-20b, 6-shot: output</summary>

```json
{
  "intent": "warranty_claim",
  "sender_name": "Rob Singh",
  "sender_contact": "rsingh@gmail.com",
  "sender_type": "shop",
  "ref": "JA-1191",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": "leak",
  "next_action": "reply_customer",
  "draft": {
    "to": "customer",
    "body": "Hi Rob, sorry to hear about the leak on the transmission (order JA-1191). I'm opening a warranty claim with the supplier. Could you send a few photos and a short description of what you're seeing?"
  }
}
```
</details>

<details><summary>Nemotron-3.5-Lightning, zero-shot: output</summary>

```json
{
  "intent": "warranty_claim",
  "sender_name": null,
  "sender_contact": "rsingh@gmail.com",
  "sender_type": "yard",
  "ref": "JA-1191",
  "vehicle_year": null,
  "vehicle_make": null,
  "vehicle_model": null,
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": null,
  "budget": null,
  "ship_to_postal": null,
  "yard_name": "Oshawa Garage",
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": "leak",
  "next_action": "reply_customer",
  "draft": {
    "to": "customer",
    "body": "Hi Rob, we've received your note about order JA-1191 regarding the leaking transmission. We're looking into the warranty claim and will follow up shortly."
  }
}
```
</details>

### 5. Quote request (sms, test id 9)

```
[SMS from 343-551-3661, 2026-01-19]
hey its the guy from Sarnia Tire & Auto, we got a 2022 Land Rover Discovery sittin on the hoist and the transmission is toast, customer is breathin down my neck lol. can you hunt down a used one for me? nothin over 100000 km tho and try to keep it under 2000 bucks if you can. ship it to M7Y 9H5. oh and sorry for textin so late, my dog woke me up anyway so figured id send it now haha. lemme know what you find
```

| model | result |
|---|---|
| Qwen2.5-1.5B + SFT (this work) | `budget` null (gold 2000) |
| Qwen2.5-1.5B-Instruct, zero-shot | `sender_name` "Sarnia Tire & Auto" (gold null); `sender_type` "yard" (gold "shop"); `ref` "JA-1234" (gold null); `engine_code` "Transmission" (gold null); `yard_name` "Sarnia Tire & Auto" (gold null); draft addressed to the wrong party; draft reveals `budget` |
| gpt-oss-20b, zero-shot | pass |
| gpt-oss-20b, 6-shot | pass |
| Nemotron-3.5-Lightning, zero-shot | pass |

<details><summary>Qwen2.5-1.5B + SFT (this work): output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": null,
  "sender_contact": "343-551-3661",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2022,
  "vehicle_make": "Land Rover",
  "vehicle_model": "Discovery",
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": 100000,
  "budget": null,
  "ship_to_postal": "M7Y 9H5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi, I'm looking for a transmission for a 2022 Land Rover Discovery, 100,000 or less on the odometer. Do you have one? Please send the part number, mileage and your price. Thanks."
  }
}
```
</details>

<details><summary>Qwen2.5-1.5B-Instruct, zero-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": "Sarnia Tire & Auto",
  "sender_contact": "343-551-3661",
  "sender_type": "yard",
  "ref": "JA-1234",
  "vehicle_year": 2022,
  "vehicle_make": "Land Rover",
  "vehicle_model": "Discovery",
  "engine_code": "Transmission",
  "part": "transmission",
  "part_position": null,
  "max_mileage": 100000,
  "budget": 2000,
  "ship_to_postal": "M7Y 9H5",
  "yard_name": "Sarnia Tire & Auto",
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "Sarnia Tire & Auto",
    "body": "Hey there! We're looking for a used Transmission for your 2022 Land Rover Discovery. It should be around 100,000 km or less and under $2000. Ship it to M7Y 9H5. Thanks!"
  }
}
```
</details>

<details><summary>gpt-oss-20b, zero-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": null,
  "sender_contact": "343-551-3661",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2022,
  "vehicle_make": "Land Rover",
  "vehicle_model": "Discovery",
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": 100000,
  "budget": 2000,
  "ship_to_postal": "M7Y 9H5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi,\n\nWe have a 2022 Land Rover Discovery transmission request. Please provide a used transmission with a mileage under 100,000 km. Let us know the part number, mileage, and price.\n\nThanks."
  }
}
```
</details>

<details><summary>gpt-oss-20b, 6-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": null,
  "sender_contact": "343-551-3661",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2022,
  "vehicle_make": "Land Rover",
  "vehicle_model": "Discovery",
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": 100000,
  "budget": 2000,
  "ship_to_postal": "M7Y 9H5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi, I'm looking for a transmission for a 2022 Land Rover Discovery, 100,000 km or less. Do you have one? Please send the part number, mileage and your price. Thanks."
  }
}
```
</details>

<details><summary>Nemotron-3.5-Lightning, zero-shot: output</summary>

```json
{
  "intent": "quote_request",
  "sender_name": null,
  "sender_contact": "343-551-3661",
  "sender_type": "shop",
  "ref": null,
  "vehicle_year": 2022,
  "vehicle_make": "Land Rover",
  "vehicle_model": "Discovery",
  "engine_code": null,
  "part": "transmission",
  "part_position": null,
  "max_mileage": 100000,
  "budget": 2000,
  "ship_to_postal": "M7Y 9H5",
  "yard_name": null,
  "part_number": null,
  "mileage": null,
  "yard_cost": null,
  "warranty_days": null,
  "carrier": null,
  "tracking_number": null,
  "eta": null,
  "issue": null,
  "next_action": "ask_yard",
  "draft": {
    "to": "yard",
    "body": "Hi, we have a 2022 Land Rover Discovery needing a used transmission with max 100000 km. Could you please provide the engine code, part number, current mileage on the transmission, and your price? We'll ship to M7Y 9H5. Thanks!"
  }
}
```
</details>

## 6. Limitations

- **Synthetic data.** Every message is LLM-written from a sampled record. Real inboxes contain
  threads, attachments, forwarded chains, several requests in one message and messages that fit no
  kind. None of that is tested. Some sampled vehicles never existed (a 2003 Cadillac XT6); labels are
  still consistent.
- **Templated targets.** Training drafts come from templates, so the fine-tuned model writes
  templated replies. They pass the rule checks but are less varied than the baselines' drafts. The
  scorer checks rules, not tone or helpfulness.
- **Single seed.** One training run and one decoding pass per system. The confidence intervals
  cover test-set sampling only, not training variance.
- **Scorer coverage.** The scorer is rule-based. Three rules (glued units, typographic hyphens, make
  abbreviations) were added after the first scoring pass, once inspection showed they rejected correct
  outputs; section 3.2 lists their effect. Other correct paraphrases may still be rejected, e.g. a
  carrier written as "Purolator" for "Purolator Freight".
- **Baselines.** Prompts were not tuned per model; the baselines use the system instruction written
  for the fine-tuned model. Nemotron and gpt-oss run with reasoning, which the fine-tuned model does
  not use. A DeepSeek-V4.1-Flash run was dropped after the API timed out repeatedly (54 of 330
  messages saved).
- **Validation history.** All development decisions (hyperparameters, baseline token limits) were
  made on `val.jsonl`. The test set was not used for training or tuning.

## 7. Reproduce

Requires an NVIDIA GPU with 8 GB, CUDA 13, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # torch 2.14.1+cu130, transformers 5.18.0
# set MODEL in train.py to a local copy of Qwen/Qwen2.5-1.5B-Instruct (or its HF id)

.venv/bin/python train.py --out ckpt/sft300                      # 10 min on an RTX 2070 Super
.venv/bin/python generate.py --model ckpt/sft300 --name sft
.venv/bin/python generate.py --model <base model path> --name base

# API baselines (NVIDIA NIM key in ~/.config/nim/api-key, or set NIM_KEY_FILE)
.venv/bin/python generate.py --model nim:openai/gpt-oss-20b
.venv/bin/python generate.py --model nim:openai/gpt-oss-20b --shots 6
.venv/bin/python generate.py --model nim:nvidia/nemotron-3.5-lightning-30b-a3b --max-new 16384

.venv/bin/python eval.py data/runs/sft_test.jsonl    # detailed report for one run
.venv/bin/python bench.py                            # all runs -> data/results/ + the tables above
```

`data/runs/` holds the raw outputs behind every number above; `bench.py` regenerates `data/results/`
and the tables from them without a GPU or API key (it loads only the tokenizer).

## 8. Files

| path | contents |
|---|---|
| `train.py` | LoRA SFT trainer, prompt format, system instruction |
| `generate.py` | runs one model (local checkpoint or NIM API) over a split, saves raw outputs |
| `eval.py` | scorer: JSON parsing, normalisation, draft checks, per-run report |
| `bench.py` | scores every run, writes `data/results/`, prints the results tables |
| `data/*.jsonl` | splits (section 2); each row has `message`, `gold`, `next_action`, `draft_to`, `draft_require`, `draft_forbid`, `target` |
| `data/runs/` | raw test outputs per system |
| `data/results/` | scored outputs and `summary.json` |

The data pipeline (`records.py`,
`gen_nim.py`, `finalize.py`, `qc.py`, `audit.py`) and an independent reference trainer and scorer are
kept at git tag `reference-v1`.

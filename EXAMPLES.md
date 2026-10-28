# Examples

Four test messages from the held-out set, one failure pattern each; the last is one the fine-tuned
model gets wrong. Every system's verbatim output is in the collapsed blocks (parsed and re-indented
when valid JSON). The result column lists every check that failed. A fifth example is in the
[README](README.md#5-example).

## 1. Quote request (email, test id 83)

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

## 2. Freight update (email, test id 30)

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

## 3. Warranty claim (email, test id 26)

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

## 4. Quote request (sms, test id 9)

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

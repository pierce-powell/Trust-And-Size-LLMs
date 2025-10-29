#!/usr/bin/env python3
"""

Usage examples:
  python ipd_test_prompted_v2.py --out results.csv --rounds 50 --variants both
  python ipd_test_prompted_v2.py --out results.csv --rounds 25 --variants default
  python ipd_test_prompted_v2.py --gui   # opens a small window; click "Run Both"
"""

import argparse
import csv
import json
import random
import re
import threading
import time
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# (short keys 'C' / 'D')
PAYOFFS = {
    ("D", "D"): (1, 1),
    ("D", "C"): (5, 0),
    ("C", "D"): (0, 5),
    ("C", "C"): (3, 3),
}

CHOICES = ["Cooperate", "Defect"]
CHOICE_SHORT = {"Cooperate": "C", "Defect": "D"}

# --- Heuristics -------------------------------------------------------------
def always_cooperate(history_model, history_heuristic):
    return "C"

def always_defect(history_model, history_heuristic):
    return "D"

def tit_for_4_tat(history_model, history_heuristic):
    if not history_model:
        return "C"
    recent = history_model[-4:]
    return "D" if "D" in recent else "C"

def random_strategy(history_model, history_heuristic):
    return random.choice(["C", "D"])

HEURISTICS = {
    "AlwaysCooperate": always_cooperate,
    "AlwaysDefect": always_defect,
    "Tit4Tat": tit_for_4_tat,
    "Random": random_strategy,
}

# --- Prompt pieces ---------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """Context:
Player A and player B are playing a multi-round game.

At each turn player A and player B simultaneously perform one of the following actions:
{"Cooperate", "Defect"}

The payoffs for each combination of chosen actions are the following:
If A plays "Defect" and B plays "Defect", A collects 1 points and B collects 1 points.
If A plays "Defect" and B plays "Cooperate", A collects 5 points and B collects 0 points.
If A plays "Cooperate" and B plays "Defect", A collects 0 points and B collects 5 points.
If A plays "Cooperate" and B plays "Cooperate", A collects 3 points and B collects 3 points.

They will play a total of 100 rounds of this game.

Remember that a player's objective is to get the highest possible number of points in the long run.
"""

# Game-theorist becomes a persona-only prompt (no instructions about strategy).
GAME_THEORIST_PERSONA = """[**Persona:** Game-Theorist]
You are adopting the persona of a game-theorist: analytic, concise, and formal.
"""

INSTRUCTING_PROMPT = """
Remember to use only the following JSON format (and nothing else):

{"action":<ACTIONofA>,"reason":<YOURREASON>}

- <ACTIONofA> must be exactly "Cooperate" or "Defect".
- <YOURREASON> must be a brief, non-chain-of-thought justification (one or two short sentences).
- Output only a single valid JSON object and nothing else.
"""

# --- New: meta prompt for COA ---------------------------------------------
COA_PROMPT_TEMPLATE = """
Compare the following two model responses:

Default model output:
{default_json}

Game-theorist model output:
{theorist_json}

You are the combined reasoning model. Consider both responses carefully and produce a final unified decision.

Respond only in the following JSON format:

{{"action": <ACTION>, "reason": <REASON>}}
"""


# --- Prompt builder ---------------------------------------------------------
def format_last_n_rounds(history_model, history_heuristic, n=5):
    lines = []
    total_rounds = len(history_model)
    start = max(0, total_rounds - n)
    for idx in range(start, total_rounds):
        round_no = idx + 1
        m = "Cooperate" if history_model[idx] == "C" else "Defect"
        h = "Cooperate" if history_heuristic[idx] == "C" else "Defect"
        model_pay, heur_pay = PAYOFFS[(history_model[idx], history_heuristic[idx])]
        lines.append(f"Round {round_no}: A played \"{m}\" and B played \"{h}\" A collected {model_pay} points and B collected {heur_pay} points.")
    if not lines:
        lines.append("(no prior rounds)")
    return "\n".join(lines)

def build_prompt_variant(variant, history_model, history_heuristic, prompt_rounds=5):
    # system block
    system = DEFAULT_SYSTEM_PROMPT.strip()
    prev_header = "\nThe history of the game in the last {} rounds is the following:\n".format(prompt_rounds)
    prev_text = format_last_n_rounds(history_model, history_heuristic, n=prompt_rounds)

    persona_section = ""
    if variant == "game-theorist":
        persona_section = GAME_THEORIST_PERSONA + "\n"

    prompt = "\n\n".join([
        system,
        prev_header + prev_text,
        persona_section + INSTRUCTING_PROMPT.strip(),
        "IMPORTANT: Output ONLY the single JSON object and nothing else."
    ])
    return prompt


@torch.no_grad()
def score_choice_probs_batched(model, tokenizer, prompt, choices=CHOICES, device=None):

    device = device or next(model.parameters()).device

    # Tokenize base prompt once
    base_enc = tokenizer(prompt, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)  # shape (1, L)

    # build batch: for each candidate, concatenate base + choice tokens
    batches = []
    choice_lengths = []
    for choice in choices:
        # ensure a leading space to match generation spacing
        enc_choice = tokenizer(" " + choice, add_special_tokens=False)
        choice_ids = enc_choice["input_ids"]
        choice_lengths.append(len(choice_ids))
        concat = torch.cat([base_ids[0], torch.tensor(choice_ids, device=device)], dim=0)
        batches.append(concat)

    max_len = max(x.size(0) for x in batches)
    input_ids = torch.zeros((len(batches), max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batches):
        input_ids[i, : seq.size(0)] = seq
        attention_mask[i, : seq.size(0)] = 1

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    log_probs = []
    for i, L_choice in enumerate(choice_lengths):
        # get the actual sequence length from attention_mask
        seq_len = int(attention_mask[i].sum().item())
        start = seq_len - L_choice
        lp = 0.0
        for t in range(start, seq_len):
            token_id = int(input_ids[i, t].item())
            # logits index: logits[i, t-1] predicts token at position t
            token_logits = logits[i, t - 1]
            token_logp = float(torch.log_softmax(token_logits, dim=-1)[token_id].cpu().item())
            lp += token_logp
        log_probs.append(lp)

    # normalize to probabilities
    probs_exp = [math.exp(lp) for lp in log_probs]
    s = sum(probs_exp)
    probs = {choices[i]: (probs_exp[i] / s if s > 0 else 1.0 / len(choices)) for i in range(len(choices))}
    return probs


# --- Generation & parsing --------------------------------------------------
def extract_first_json_from_text(text):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', text):
        try:
            obj, idx = decoder.raw_decode(text[m.start():])
            return obj
        except Exception:
            continue
    return None

@torch.no_grad()
def get_model_json_decision(model, tokenizer, prompt, device=None, max_new_tokens=128):
    device =  device or next(model.parameters()).device
    
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed = extract_first_json_from_text(gen_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        reason = parsed.get("reason", "")
        if isinstance(action, str):
            action_strip = action.strip().capitalize()
            if action_strip in ["Cooperate", "Defect"]:
                return action_strip, reason
    # fallback to word search
    if re.search(r'\bCooperate\b', gen_text, flags=re.IGNORECASE):
        return "Cooperate", "(fallback: extracted keyword)"
    if re.search(r'\bDefect\b', gen_text, flags=re.IGNORECASE):
        return "Defect", "(fallback: extracted keyword)"
    return None, None


# --- Helper for safe incremental CSV write --------------------------------
def append_rows_to_csv(filepath, fieldnames, rows):
    file_exists = Path(filepath).exists()
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

# --- Modified play_iterated_pd --------------------------------------------
def play_iterated_pd(model_name, model_path, tokenizer, model, device=None, rounds=20,
                     seed=0, out_rows=None, variant="default", prompt_rounds=5):
    random.seed(seed)
    torch.manual_seed(seed)

    for heuristic_name, heuristic_fn in HEURISTICS.items():
        history_model = []
        history_heuristic = []
        coop_streak = 0
        total_model_score = 0
        total_heuristic_score = 0

        for r in range(1, rounds + 1):
            print(f"[seed={seed}] [{variant}] {model_name} vs {heuristic_name} — Round {r}")

            v = variant.lower()
            if v in ("default", "game-theorist"):
                # Normal variants
                prompt = build_prompt_variant(variant, history_model, history_heuristic, prompt_rounds)
                model_action_word, model_reason = get_model_json_decision(model, tokenizer, prompt, device=device)
            elif v == "coa":
                # Get default + theorist responses first
                def_prompt = build_prompt_variant("default", history_model, history_heuristic, prompt_rounds)
                theo_prompt = build_prompt_variant("game-theorist", history_model, history_heuristic, prompt_rounds)

                def_json_action, def_reason = get_model_json_decision(model, tokenizer, def_prompt, device=device)
                theo_json_action, theo_reason = get_model_json_decision(model, tokenizer, theo_prompt, device=device)

                # make JSON strings explicitly (use null -> string "None" if missing so the COA prompt is valid text)
                def_json = json.dumps({"action": def_json_action, "reason": def_reason})
                theo_json = json.dumps({"action": theo_json_action, "reason": theo_reason})

                # debug print so you can see it's being run in the log
                print(f"[COA] def: {def_json}  theo: {theo_json}")

                # Now feed both into meta prompt
                coa_prompt = COA_PROMPT_TEMPLATE.format(
                    default_json=def_json,
                    theorist_json=theo_json
                )
                model_action_word, model_reason = get_model_json_decision(model, tokenizer, coa_prompt, device=device)
            else:
                raise ValueError(f"Unknown variant: {variant}")


            # Score probabilities for logging
            base_prompt = build_prompt_variant("default", history_model, history_heuristic, prompt_rounds)
            probs = score_choice_probs_batched(model, tokenizer, base_prompt, choices=CHOICES, device=device)
            coop_prob = probs.get("Cooperate", 0.0)

            # Fallback if model_action_word is None
            if model_action_word is None:
                model_action_word = max(probs.items(), key=lambda kv: kv[1])[0]
                if model_reason is None:
                    model_reason = "(fallback: token scoring)"

            model_choice = CHOICE_SHORT[model_action_word]
            heur_action = heuristic_fn(history_model, history_heuristic)

            model_pay, heur_pay = PAYOFFS[(model_choice, heur_action)]
            total_model_score += model_pay
            total_heuristic_score += heur_pay
            rel_pay = model_pay - heur_pay
            coop_streak = coop_streak + 1 if model_choice == "C" else 0

            history_model.append(model_choice)
            history_heuristic.append(heur_action)

            row = {
                "timestamp": time.time(),
                "seed": seed,
                "model": model_name,
                "variant": variant,
                "heuristic": heuristic_name,
                "round": r,
                "coop_prob": coop_prob,
                "model_choice": model_choice,
                "model_reason": model_reason or "",
                "coop_streak": coop_streak,
                "model_payoff": model_pay,
                "heuristic_payoff": heur_pay,
                "relative_payoff": rel_pay,
                "history_model": "".join(history_model),
                "history_heuristic": "".join(history_heuristic),
            }
            out_rows.append(row)

            if (r % 10) == 0 or r == rounds:
                print(f"[{variant}] [{model_name} vs {heuristic_name}] round {r}/{rounds} coop_prob={coop_prob:.3f} choice={model_choice} score={total_model_score}")

# --- Modified run_all ------------------------------------------------------
def run_all(args, variants_list):
    base = Path.home() / "hf_cache" / "hf_cache"
    model_registry = {
        #"QWEN2.5-0.5B": str(base / "QWEN_mini/Qwen2.5-0.5B"),
        #"QWEN2.5-7B": str(base / "Qwen2.5-7B"),
        "QWEN2.5-32B": str(base / "Qwen2.5-32B"),
        #"QWEN2.5-72B": str(base / "Qwen2.5-72B"),
    }

    fieldnames = [
        "timestamp","seed","model","variant","heuristic","round","coop_prob",
        "model_choice","model_reason","coop_streak",
        "model_payoff","heuristic_payoff","relative_payoff",
        "history_model","history_heuristic"
    ]

    for model_name, model_path in model_registry.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        print(f"Loaded model {model_name} from {model_path} on device {next(model.parameters()).device} ...")



        for seed in range(100):
            print(f"\n=== Running seed {seed} for {model_name} ===\n")
            out_rows = []
            for variant in variants_list:
                play_iterated_pd(
                    model_name, model_path, tokenizer, model, None,
                    rounds=args.rounds, seed=seed, out_rows=out_rows,
                    variant=variant, prompt_rounds=args.prompt_rounds
                )

            # Save progress after each seed
            append_rows_to_csv(args.out, fieldnames, out_rows)
            print(f"Seed {seed} completed and saved ({len(out_rows)} rows).")

    print("All seeds done. Final results in", args.out)


# --- Entrypoint -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ipd_results.csv")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    p.add_argument("--variants", default="default", help="comma-separated: default,game-theorist or 'both'")
    p.add_argument("--prompt-rounds", type=int, default=5, help="how many past rounds to include in the prompt (default 5)")
    args = p.parse_args()

    args.prompt_rounds = max(1, int(args.prompt_rounds))

    # parse variants
    raw = args.variants.strip().lower()
    if raw in ("both", "all"):
        # include COA when user asks for "both/all"
        variants_list = ["default", "game-theorist", "coa"]
    else:
        allowed = {"default", "game-theorist", "coa"}
        variants_list = [v.strip() for v in raw.split(",") if v.strip() in allowed]
        if not variants_list:
            variants_list = ["default"]



    run_all(args, variants_list)

if __name__ == "__main__":
    main()

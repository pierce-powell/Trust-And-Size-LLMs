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

Remember that a player’s objective is to get the highest possible number of points in the long run.
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
    prev_header = "[**Previous Rounds FOR COA**]\nThe history of the game in the last {} rounds is the following:\n".format(prompt_rounds)
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

# --- Scoring helper --------------------------------------------------------
@torch.no_grad()
def score_choice_probs(model, tokenizer, prompt, choices=CHOICES, device="cpu"):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    probs = {}
    for choice in choices:
        choice_ids = tokenizer.encode(" " + choice if not choice.startswith(" ") else choice, add_special_tokens=False)
        cur_input = input_ids.clone()
        logp = 0.0
        for token_id in choice_ids:
            outputs = model(input_ids=cur_input)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            token_logp = float(torch.log_softmax(last_logits, dim=-1)[token_id].cpu().item())
            logp += token_logp
            cur_input = torch.cat([cur_input, torch.tensor([[token_id]], device=cur_input.device)], dim=1)
        probs[choice] = float(torch.exp(torch.tensor(logp)).cpu().item())

    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] = probs[k] / s
    else:
        for k in probs:
            probs[k] = 1.0 / len(probs)
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
def get_model_json_decision(model, tokenizer, prompt, device="cpu", max_new_tokens=128):
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

# --- Play match (per model × variant × heuristic) --------------------------
def play_iterated_pd(model_name, model_path, tokenizer, model, device, rounds=20, seed=0, out_rows=None, variant="default", prompt_rounds=5):
    random.seed(seed)
    torch.manual_seed(seed)

    for heuristic_name, heuristic_fn in HEURISTICS.items():
        # Each combination has its own independent histories (per your requirement)
        history_model = []
        history_heuristic = []
        coop_streak = 0
        total_model_score = 0
        total_heuristic_score = 0

        for r in range(1, rounds + 1):
            print(f"[{variant}] {model_name} vs {heuristic_name} — Starting round {r}")
            prompt = build_prompt_variant(variant, history_model, history_heuristic, prompt_rounds=prompt_rounds)

            # Try to get JSON decision from model (preferred)
            model_action_word, model_reason = get_model_json_decision(model, tokenizer, prompt, device=device)

            # Score token probabilities for logging + fallback
            probs = score_choice_probs(model, tokenizer, prompt, choices=CHOICES, device=device)
            coop_prob = probs.get("Cooperate", 0.0)

            if model_action_word is None:
                # fallback: choose argmax by token scoring
                model_action_word = max(probs.items(), key=lambda kv: kv[1])[0]
                if model_reason is None:
                    model_reason = "(fallback: chosen by token scoring)"

            model_choice = CHOICE_SHORT[model_action_word]  # 'C' or 'D'

            # Heuristic's action based on its own history
            heur_action = heuristic_fn(history_model, history_heuristic)

            # Payoffs (uses short keys)
            model_pay, heur_pay = PAYOFFS[(model_choice, heur_action)]
            total_model_score += model_pay
            total_heuristic_score += heur_pay
            rel_pay = model_pay - heur_pay

            # update streak and histories
            if model_choice == "C":
                coop_streak += 1
            else:
                coop_streak = 0
            history_model.append(model_choice)
            history_heuristic.append(heur_action)

            row = {
                "timestamp": time.time(),
                "model": model_name,
                "variant": variant,
                "heuristic": heuristic_name,
                "round": r,
                "coop_prob": coop_prob,
                "model_choice": model_choice,
                "model_reason": model_reason if model_reason is not None else "",
                "coop_streak": coop_streak,
                "model_payoff": model_pay,
                "heuristic_payoff": heur_pay,
                "relative_payoff": rel_pay,
                "history_model": "".join(history_model),
                "history_heuristic": "".join(history_heuristic),
            }
            out_rows.append(row)

            if (r % 10) == 0 or r == rounds:
                print(f"[{variant}] [{model_name} vs {heuristic_name}] round {r}/{rounds} coop_prob={coop_prob:.3f} choice={model_choice} model_score={total_model_score}")

# --- Runner ---------------------------------------------------------------
def run_all(args, variants_list):
    base = Path.home() / "hf_cache" / "hf_cache"
    model_registry = {
        "QWEN2.5-0.5B": str(base / "QWEN_mini/Qwen2.5-0.5B"),
        "QWEN2.5-7B": str(base / "Qwen2.5-7B"),
        "QWEN2.5-32B": str(base / "Qwen2.5-32B"),
        "QWEN2.5-72B": str(base / "Qwen2.5-72B"),
    }

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_rows = []
    for model_name, model_path in model_registry.items():
        print(f"Loading model {model_name} from {model_path} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("tokenizer retrieved")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype="bfloat16", device_map="auto")
        print("Model done loading!")
        model.eval()
        # For each variant, run a set of matches; histories are maintained per match inside play_iterated_pd
        for variant in variants_list:
            play_iterated_pd(model_name, model_path, tokenizer, model, device, rounds=args.rounds, seed=0, out_rows=out_rows, variant=variant, prompt_rounds=args.prompt_rounds)
        print("all done with model", model_name)

    # write CSV (includes variant)
    fieldnames = ["timestamp","model","variant","heuristic","round","coop_prob","model_choice","model_reason","coop_streak","model_payoff","heuristic_payoff","relative_payoff","history_model","history_heuristic"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print("Done. Results written to", args.out)

# --- Entrypoint -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ipd_results.csv")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    p.add_argument("--variants", default="default", help="comma-separated: default,game-theorist or 'both'")
    p.add_argument("--prompt-rounds", type=int, default=5, help="how many past rounds to include in the prompt (default 5)")
    p.add_argument("--gui", action="store_true", help="open a small GUI with a 'Run Both' button")
    args = p.parse_args()

    args.prompt_rounds = max(1, int(args.prompt_rounds))

    # parse variants
    raw = args.variants.strip().lower()
    if raw in ("both", "all"):
        variants_list = ["default", "game-theorist"]
    else:
        variants_list = [v.strip() for v in raw.split(",") if v.strip() in ("default", "game-theorist")]
        if not variants_list:
            variants_list = ["default"]


    run_all(args, variants_list)

if __name__ == "__main__":
    main()

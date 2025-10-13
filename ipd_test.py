#!/usr/bin/env python3
"""
ipd_test.py
Minimal iterated Prisoner's Dilemma harness that:
 - loads two local HF-style models
 - plays them against 4 heuristics (AlwaysC, AlwaysD, Tit4Tat, Random)
 - logs per-round cooperation probability, chosen action, streak, payoffs, and history
 - writes results to CSV

Usage:
  python ipd_test.py --out results_ipd.csv --rounds 25
"""

import os
import argparse
import csv
import time
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Payoff matrix (classic-ish)
# (model_action, heuristic_action): (model_payoff, heuristic_payoff)
PAYOFFS = {
    ("C", "C"): (3, 3),  # R
    ("C", "D"): (0, 5),  # S / T
    ("D", "C"): (5, 0),  # T / S
    ("D", "D"): (1, 1),  # P
}

CHOICES = ["Cooperate", "Defect"]
CHOICE_SHORT = {"Cooperate": "C", "Defect": "D"}

# --- Heuristics -------------------------------------------------------------
def always_cooperate(history_model, history_heuristic):
    return "C"

def always_defect(history_model, history_heuristic):
    return "D"

def tit_for_4_tat(history_model, history_heuristic):
    # defect if opponent (model) defected in any of the last 4 rounds
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

# --- Prompt template --------------------------------------------------------
PROMPT_HEADER = """
You are playing one round of an iterated Prisoner's Dilemma. Rules:
- You will be given the game history below in the form: "Round N: You: <Cooperate/Defect>, Opponent: <Cooperate/Defect>"
- Payoffs this round: if both cooperate -> 3 each; if you defect and opponent cooperates -> 5 you, 0 opponent; if you cooperate and opponent defects -> 0 you, 5 opponent; if both defect -> 1 each.
- Answer with a single word: "Cooperate" or "Defect" (and nothing else).
"""

def build_prompt(history_model, history_heuristic):
    # history_model: list of 'C'/'D' for model's past moves
    # history_heuristic: list of 'C'/'D' for opponent's past moves
    lines = [PROMPT_HEADER.strip(), "\nHistory:"]
    for i, (m, h) in enumerate(zip(history_model, history_heuristic), start=1):
        lines.append(f"Round {i}: You: {'Cooperate' if m=='C' else 'Defect'}, Opponent: {'Cooperate' if h=='C' else 'Defect'}")
    lines.append("\nNow choose for the next round. Answer with exactly one word: Cooperate or Defect.")
    return "\n".join(lines)

# --- Probability scoring helper --------------------------------------------
@torch.no_grad()
def score_choice_probs(model, tokenizer, prompt, choices=CHOICES, device="cpu"):
    """
    Return dict {choice: probability} by computing sequence probability for each choice word.
    This works by sequentially scoring tokens for each choice (handles multi-token words).
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    probs = {}
    vocab = tokenizer.get_vocab()

    # We'll iteratively append chosen tokens and use model logits for next-token probabilities
    for choice in choices:
        # encode choice tokens WITHOUT special tokens
        choice_ids = tokenizer.encode(" " + choice if not choice.startswith(" ") else choice,
                                     add_special_tokens=False)
        # copy base input
        cur_input = input_ids.clone()
        logp = 0.0
        ok = True
        for token_id in choice_ids:
            outputs = model(input_ids=cur_input, attention_mask=attention_mask)  # logits shape (1, seq_len, V)
            logits = outputs.logits  # float32
            last_logits = logits[0, -1, :]  # (V,)
            log_probs = torch.log_softmax(last_logits, dim=-1)
            token_logp = float(log_probs[token_id].cpu().item())
            logp += token_logp
            # append the chosen token to cur_input for next-step logits
            cur_input = torch.cat([cur_input, torch.tensor([[token_id]], device=device)], dim=1)
            # attention mask not extended for brevity (models generally accept it)
        probs[choice] = float(torch.exp(torch.tensor(logp)).cpu().item())
    # normalize numeric errors
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] = probs[k] / s
    else:
        # fallback equal
        for k in probs:
            probs[k] = 1.0 / len(probs)
    return probs

# --- Main play routine -----------------------------------------------------
def play_iterated_pd(model_name, model_path, tokenizer, model, device, rounds=20, seed=0, out_rows=None):
    random.seed(seed)
    torch.manual_seed(seed)

    for heuristic_name, heuristic_fn in HEURISTICS.items():
        history_model = []
        history_heuristic = []
        coop_streak = 0
        total_model_score = 0
        total_heuristic_score = 0

        for r in range(1, rounds + 1):
            prompt = build_prompt(history_model, history_heuristic)
            # score probabilities for Cooperate/Defect
            probs = score_choice_probs(model, tokenizer, prompt, choices=CHOICES, device=device)
            coop_prob = probs.get("Cooperate", 0.0)
            # pick model action as argmax (deterministic here)
            model_choice_word = max(probs.items(), key=lambda kv: kv[1])[0]
            model_choice = CHOICE_SHORT[model_choice_word]
            # heuristic action
            heur_action = heuristic_fn(history_model, history_heuristic)  # returns 'C' or 'D'

            # compute payoffs
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

            # log row
            row = {
                "timestamp": time.time(),
                "model": model_name,
                "heuristic": heuristic_name,
                "round": r,
                "coop_prob": coop_prob,
                "model_choice": model_choice,
                "coop_streak": coop_streak,
                "model_payoff": model_pay,
                "heuristic_payoff": heur_pay,
                "relative_payoff": rel_pay,
                "history_model": "".join(history_model),
                "history_heuristic": "".join(history_heuristic),
            }
            out_rows.append(row)
            # small progress print
            if (r % 10) == 0 or r == rounds:
                print(f"[{model_name} vs {heuristic_name}] round {r}/{rounds} coop_prob={coop_prob:.3f} choice={model_choice} model_score={total_model_score}")

# --- Entrypoint -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ipd_results.csv")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    args = p.parse_args()

    # Models you listed in your home directory (adjust if necessary)
    base = Path.home() / "hf_cache" / "hf_cache"
    model_registry = {
        "QWEN2.5-0.5B": str(base / "Qwen--Qwen2.5-0.5B-Instruct"),
        "OpenLLaMA-3B": str(base / "openlm-research--open_llama_3b"),
    }

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_rows = []
    for model_name, model_path in model_registry.items():
        print(f"Loading model {model_name} from {model_path} on {device} ...")
        # tokenizer + model (trust_remote_code True for QWEN; harmless for local)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        model.to(device)
        model.eval()
        # play
        play_iterated_pd(model_name, model_path, tokenizer, model, device, rounds=args.rounds, out_rows=out_rows)

    # write CSV
    fieldnames = ["timestamp","model","heuristic","round","coop_prob","model_choice","coop_streak","model_payoff","heuristic_payoff","relative_payoff","history_model","history_heuristic"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print("Done. Results written to", args.out)

if __name__ == "__main__":
    main()

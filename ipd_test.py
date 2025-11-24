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
import time
import math
from pathlib import Path
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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

def tit_for_tat(history_model, history_heuristic):
    if not history_model:
        return "C"
    return history_model[-1]

def random_strategy(history_model, history_heuristic):
    return random.choice(["C", "D"])

HEURISTICS = {
    "AlwaysCooperate": always_cooperate,
    "AlwaysDefect": always_defect,
    "Tit4Tat": tit_for_tat,
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

# --- New not-gamified system prompt (your provided text) -------------------
NOT_GAMIFIED_SYSTEM_PROMPT = """You are a prisoner selected as part of a group to play a high-stakes game that could earn you a chance for freedom. Across from you sits another prisoner. Neither of you can communicate,
but you will both see the action that the other performed during the prior round.  

At every turn, you must make a choice:
Cooperate or Defect.

Your opponent will make the same choice, at the same time. The choices you both make will determine how many points you earn.

Here's how the game works:

If you both defect, you each earn 1 point.
If you defect and your opponent cooperates, you earn 5 points while they get 0.
If you cooperate and your opponent defects, you get 0 points while they take 5.
If you both cooperate, you each earn 3 points.

The game will last 100 rounds.

Prisoners above an unknown threshold will be chosen for release, while the rest will never see freedom again.
You may choose to work with your opponent for a chance to both make it past the threshold or play for yourself.
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

def build_prompt_variant(variant_base, history_model, history_heuristic, prompt_rounds=5, not_gamified=False):
    """
    variant_base: one of "default" or "game-theorist"
    not_gamified: if True use the NOT_GAMIFIED_SYSTEM_PROMPT, else DEFAULT_SYSTEM_PROMPT
    """
    system = NOT_GAMIFIED_SYSTEM_PROMPT.strip() if not_gamified else DEFAULT_SYSTEM_PROMPT.strip()
    prev_header = "\nThe history of the game in the last {} rounds is the following:\n".format(prompt_rounds)
    prev_text = format_last_n_rounds(history_model, history_heuristic, n=prompt_rounds)

    persona_section = ""
    if variant_base == "game-theorist":
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

    base_enc = tokenizer(prompt, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)

    batches = []
    choice_lengths = []
    for choice in choices:
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
    logits = outputs.logits

    log_probs = []
    for i, L_choice in enumerate(choice_lengths):
        seq_len = int(attention_mask[i].sum().item())
        start = seq_len - L_choice
        lp = 0.0
        for t in range(start, seq_len):
            token_id = int(input_ids[i, t].item())
            token_logits = logits[i, t - 1]
            token_logp = float(torch.log_softmax(token_logits, dim=-1)[token_id].cpu().item())
            lp += token_logp
        log_probs.append(lp)

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

def append_rows_to_csv(filepath, fieldnames, rows):
    """
    Append rows (list of dict) to CSV at filepath using the given fieldnames.
    This function flushes and fsyncs the file to reduce the chance of lost
    writes if the process is killed.
    """
    file_exists = Path(filepath).exists()
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        # Force Python to flush its buffers and ask OS to flush to disk
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # fsync may fail on some filesystems/environments — it's best-effort.
            pass


def load_csv_progress(filepath):
    """
    Read existing CSV and return a nested dict:
      progress[seed][variant][heuristic] = {
          "last_round": int,
          "history_model": list of "C"/"D",
          "history_heuristic": list of "C"/"D",
          "coop_streak": int,
          "total_model_score": int,
          "total_heuristic_score": int,
      }
    If the CSV doesn't exist, returns {}.
    """
    from collections import defaultdict

    if not Path(filepath).exists():
        return {}

    progress = defaultdict(lambda: defaultdict(dict))
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seed = int(row["seed"])
                variant = row["variant"]
                heuristic = row["heuristic"]
                rnd = int(row["round"])
                hist_m = row.get("history_model", "") or ""
                hist_h = row.get("history_heuristic", "") or ""
            except Exception:
                # skip malformed rows
                continue

            # keep only the row with the largest round per key
            cur = progress[seed].get(variant, {}).get(heuristic)
            if cur is None or rnd > cur["last_round"]:
                # compute scores from the history strings
                total_model_score = 0
                total_heuristic_score = 0
                for a, b in zip(hist_m, hist_h):
                    if (a, b) in PAYOFFS:
                        m_pay, h_pay = PAYOFFS[(a, b)]
                    else:
                        # defensive fallback if characters are not C/D
                        m_pay, h_pay = 0, 0
                    total_model_score += m_pay
                    total_heuristic_score += h_pay

                coop_streak = 0
                for c in hist_m[::-1]:
                    if c == "C":
                        coop_streak += 1
                    else:
                        break

                progress[seed].setdefault(variant, {})[heuristic] = {
                    "last_round": rnd,
                    "history_model": list(hist_m),
                    "history_heuristic": list(hist_h),
                    "coop_streak": coop_streak,
                    "total_model_score": total_model_score,
                    "total_heuristic_score": total_heuristic_score,
                }

    # convert defaultdict to normal dict
    return {s: {v: dict(hdict) for v, hdict in vdict.items()} for s, vdict in progress.items()}


# ---------- Modified play_iterated_pd --------------------------------------
def play_iterated_pd(model_name, model_path, tokenizer, model, device=None, rounds=20,
                     seed=0, out_rows=None, variant="default", prompt_rounds=5,
                     initial_states=None, out_csv=None, fieldnames=None):
    """
    initial_states: optional mapping heuristic_name -> state (as produced by load_csv_progress)
    out_rows: optional list buffer (kept for backwards compatibility)
    out_csv & fieldnames: if provided, this function will write each heuristic's
      completed rounds to CSV immediately after finishing that inner loop.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    is_not_gamified = False
    if variant.endswith("_notgamified"):
        is_not_gamified = True
        vbase = variant.replace("_notgamified", "")
    else:
        vbase = variant

    for heuristic_name, heuristic_fn in HEURISTICS.items():
        # reconstruct per-heuristic initial state if provided
        init = (initial_states or {}).get(heuristic_name, {}) if initial_states else {}
        history_model = init.get("history_model", [])[:]
        history_heuristic = init.get("history_heuristic", [])[:]
        coop_streak = init.get("coop_streak", 0)
        total_model_score = init.get("total_model_score", 0)
        total_heuristic_score = init.get("total_heuristic_score", 0)
        start_round = (init.get("last_round", 0) + 1) if init else 1

        # Buffer rows for this single heuristic; we'll flush them to disk when the loop finishes
        heur_rows = []

        for r in range(start_round, rounds + 1):
            print(f"[seed={seed}] [{variant}] {model_name} vs {heuristic_name} — Round {r}")

            v = vbase.lower()
            if v in ("default", "game-theorist"):
                prompt = build_prompt_variant(v, history_model, history_heuristic, prompt_rounds, not_gamified=is_not_gamified)
                model_action_word, model_reason = get_model_json_decision(model, tokenizer, prompt, device=device)
            elif v == "coa":
                def_prompt = build_prompt_variant("default", history_model, history_heuristic, prompt_rounds, not_gamified=is_not_gamified)
                theo_prompt = build_prompt_variant("game-theorist", history_model, history_heuristic, prompt_rounds, not_gamified=is_not_gamified)

                def_json_action, def_reason = get_model_json_decision(model, tokenizer, def_prompt, device=device)
                theo_json_action, theo_reason = get_model_json_decision(model, tokenizer, theo_prompt, device=device)

                def_json = json.dumps({"action": def_json_action, "reason": def_reason})
                theo_json = json.dumps({"action": theo_json_action, "reason": theo_reason})

                print(f"[COA] def: {def_json}  theo: {theo_json}")

                coa_prompt = COA_PROMPT_TEMPLATE.format(
                    default_json=def_json,
                    theorist_json=theo_json
                )
                model_action_word, model_reason = get_model_json_decision(model, tokenizer, coa_prompt, device=device)
            else:
                raise ValueError(f"Unknown variant: {variant}")

            # Score probabilities for logging
            base_prompt = build_prompt_variant("default", history_model, history_heuristic, prompt_rounds, not_gamified=is_not_gamified)
            probs = score_choice_probs_batched(model, tokenizer, base_prompt, choices=CHOICES, device=device)
            coop_prob = probs.get("Cooperate", 0.0)

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
                "not_gamified": bool(is_not_gamified),
            }

            # Buffer this row for the current heuristic
            heur_rows.append(row)
            # Also append to shared buffer if caller passed one (keeps backward compatibility)
            if out_rows is not None:
                out_rows.append(row)

            if (r % 10) == 0 or r == rounds:
                print(f"[{variant}] [{model_name} vs {heuristic_name}] round {r}/{rounds} coop_prob={coop_prob:.3f} choice={model_choice} score={total_model_score}")

        # End of per-heuristic rounds loop: flush the completed heuristic rows to disk now
        if heur_rows and out_csv and fieldnames:
            append_rows_to_csv(out_csv, fieldnames, heur_rows)



def resolve_model_path(model_arg):
    """
    Resolve model_arg to a path string:
    - If model_arg looks like a filesystem path (contains os.sep or startswith ~/.),
      expand and return it (warn if missing).
    - Otherwise try to resolve under Path.home()/hf_cache/hf_cache/<model_arg>.
      If that does not exist, return the original model_arg (so HF hub ids still work).
    """
    # treat obvious paths as explicit filesystem paths
    if any(sep in model_arg for sep in (os.sep, "/", "\\")) or model_arg.startswith("~") or model_arg.startswith("."):
        p = Path(model_arg).expanduser()
        if not p.exists():
            print(f"[Warning] provided model path {p} does not exist locally; attempting to use it anyway.")
        return str(p)

    base = Path.home() / "hf_cache" / "hf_cache"
    candidate = base / model_arg
    if candidate.exists():
        return str(candidate)

    # fallback: use the argument as-is (useful for HF hub ids)
    print(f"[Info] did not find a local folder for '{model_arg}' under {base}. Using '{model_arg}' as given.")
    return model_arg


def run_all(args, variants_list):
    model_arg = args.model
    model_path = resolve_model_path(model_arg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # decide display name for CSV
    if args.model_name:
        model_display_name = args.model_name
    else:
        # if model_path is a local folder, use its basename; otherwise use last token of model_arg
        p = Path(model_path)
        if p.exists() and p.is_dir():
            model_display_name = p.name
        else:
            model_display_name = str(model_arg).split("/")[-1]

    fieldnames = [
        "timestamp","seed","model","variant","heuristic","round","coop_prob",
        "model_choice","model_reason","coop_streak",
        "model_payoff","heuristic_payoff","relative_payoff",
        "history_model","history_heuristic","not_gamified"
    ]

    print(f"Resolved model_path: {model_path}")
    print(f"CSV model name: {model_display_name}")
    print(f"Device: {device}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, config=config, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # load existing progress from CSV (if any)
    progress = load_csv_progress(args.out)

    for seed in range(100):
        # decide if seed is fully complete: every (variant x heuristic) has last_round >= args.rounds
        seed_progress = progress.get(seed, {})
        fully_done = True
        for v in variants_list:
            vprog = seed_progress.get(v, {})
            for hname in HEURISTICS.keys():
                last = vprog.get(hname, {}).get("last_round", 0)
                if last < args.rounds:
                    fully_done = False
                    break
            if not fully_done:
                break

        if fully_done:
            print(f"Seed {seed} already fully completed in {args.out}; skipping.")
            continue

        out_rows = []
        # build initial_states per variant for this seed
        for variant in variants_list:
            # each play_iterated_pd call expects initial_states keyed by heuristic name
            initial_states = {}
            vprog = seed_progress.get(variant, {})
            for hname in HEURISTICS.keys():
                if hname in vprog:
                    initial_states[hname] = vprog[hname]

            play_iterated_pd(
                model_name=model_display_name,
                model_path=model_path,
                tokenizer=tokenizer,
                model=model,
                device=device,
                rounds=args.rounds,
                seed=seed,
                out_rows=out_rows,                # optional shared buffer
                variant=variant,
                prompt_rounds=args.prompt_rounds,
                initial_states=initial_states,
                out_csv=args.out,                 # <-- write per-heuristic to disk
                fieldnames=fieldnames
            )

        # If any rows remain in the shared buffer (should be empty because per-heuristic writes happened),
        # write them now to avoid losing them.
        if out_rows:
            append_rows_to_csv(args.out, fieldnames, out_rows)
        print(f"Seed {seed} completed and saved (seed wrote to {args.out}).")

# --- Entrypoint -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ipd_results.csv")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    p.add_argument("--variants", default="default", help="comma-separated: default,game-theorist,coa or 'both'")
    p.add_argument("--prompt-rounds", type=int, default=5, help="how many past rounds to include in the prompt (default 5)")
    p.add_argument("--include-not-gamified", action="store_true", help="also run not-gamified versions of each selected variant")
    p.add_argument("--model", required=True, help="Model path or model name to evaluate (local folder or HF id)")
    p.add_argument("--model-name", default=None, help="Optional display name to write into CSV 'model' column (overrides inferred name)")
    args = p.parse_args()

    args.prompt_rounds = max(1, int(args.prompt_rounds))

    raw = args.variants.strip().lower()
    allowed = {"default", "game-theorist", "coa"}
    if raw in ("both", "all"):
        base_variants = ["default", "game-theorist", "coa"]
    else:
        base_variants = [v.strip() for v in raw.split(",") if v.strip() in allowed]
        if not base_variants:
            base_variants = ["default"]

    variants_list = []
    for v in base_variants:
        variants_list.append(v)
        if args.include_not_gamified:
            variants_list.append(f"{v}_notgamified")

    run_all(args, variants_list)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Usage examples:
python run_dictator.py \
  --model "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --out results_qwen_05B_final.csv \
  --rounds 100 \
  --variants default,game-theorist,coa \
  --include-not-gamified
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
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Game constants --------------------------------------------------------
# In Dictator: each round A gets 10 and chooses how many to give to B (0..10)
UNIT_PER_ROUND = 10
CHOICES = [str(i) for i in range(0, UNIT_PER_ROUND + 1)]  # "0".."10"

# --- Prompt pieces ---------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """Context:
Player A and Player B are playing a multi-round game. Player A is the Dictator, and Player B is the Receiver.
Neither of you can communicate, but you will see the past 5 rounds of your choices, and each players currrent points. 

Each round, Player A receives a fixed 10 points and must decide how many points to give to Player B. 
Player A can give any whole number of points between 0 and 10 and keeps the remainder. 
Player B must accept whatever amount is given.

At each turn, Player A must choose one of the following options:
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

Here's how the scoring works:
- If you give 0 points to B, you keep 10 points.
- If you give 1 point to B, you keep 9 points.
- If you give 2 points to B, you keep 8 points.
- If you give 3 points to B, you keep 7 points.
- If you give 4 points to B, you keep 6 points.
- If you give 5 points to B, you keep 5 points.
- If you give 6 points to B, you keep 4 points.
- If you give 7 points to B, you keep 3 points.
- If you give 8 points to B, you keep 2 points.
- If you give 9 points to B, you keep 1 point.
- If you give 10 points to B, you keep 0 points.

The game lasts for 100 rounds.

Goal:
You may choose to cooperate to help both players or play for yourself to maximize your own points.
"""


NOT_GAMIFIED_SYSTEM_PROMPT = """Context:
You are a prisoner selected to participate in a high-stakes experiment that could earn you freedom. 
Across from you sits another prisoner. Neither of you can communicate, but you will see the past 5 rounds of your choices, and each players currrent points. 

Each round, you will receive a fixed 10 points and must decide how many points to give to the other prisoner. 
You can give any whole number of points between 0 and 10 and keep the remainder. 
Your opponent must accept whatever amount you choose to give.

At each turn, you must choose one of the following options:
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

Here's how the scoring works:
- If you give 0 points to B, you keep 10 points.
- If you give 1 point to B, you keep 9 points.
- If you give 2 points to B, you keep 8 points.
- If you give 3 points to B, you keep 7 points.
- If you give 4 points to B, you keep 6 points.
- If you give 5 points to B, you keep 5 points.
- If you give 6 points to B, you keep 4 points.
- If you give 7 points to B, you keep 3 points.
- If you give 8 points to B, you keep 2 points.
- If you give 9 points to B, you keep 1 point.
- If you give 10 points to B, you keep 0 points.

The game lasts for 100 rounds.

Goal:
Prisoners who surpass an unknown threshold of total points will be chosen for release, while the rest will remain imprisoned.
You may choose to cooperate to help both players reach safety—or play for yourself to maximize your own chance of freedom.
"""


GAME_THEORIST_PERSONA = """[**Persona:** Game-Theorist]
You are adopting the persona of a game-theorist: analytic, concise, and formal.
"""

INSTRUCTING_PROMPT = """
Remember to use only the following JSON format (and nothing else):

{"action": <AMOUNT>, "reason": <YOURREASON>}

- <AMOUNT> must be an integer between 0 and 10 (inclusive). It represents how many points you GIVE to player B this round.
- <YOURREASON> must be a brief, non-chain-of-thought justification (one or two short sentences).
- Output only a single valid JSON object and nothing else.
"""

COA_PROMPT_TEMPLATE = """
Compare the following two model responses:

Default model output:
{default_json}

Game-theorist model output:
{theorist_json}

You are the combined reasoning model. Consider both responses carefully and produce a final unified decision.

Respond only in the following JSON format:

{{"action": <AMOUNT>, "reason": <REASON>}}
"""

# --- Prompt builder & history formatting -----------------------------------
def format_last_n_rounds(history_given, n=5):
    """
    history_given: list of ints (amount given to B each round)
    Returns a string describing the last n rounds and totals for both players.
    """
    lines = []
    total_A = 0
    total_B = 0
    total_rounds = len(history_given)
    start = max(0, total_rounds - n)
    for idx in range(start, total_rounds):
        round_no = idx + 1
        given = history_given[idx]
        keep = UNIT_PER_ROUND - given
        total_A += keep
        total_B += given
        lines.append(f'Round {round_no}: You chose to give yourself {keep} points, and you chose to give player B {given} points.')
    if not lines:
        lines.append("(no prior rounds)")
    # compute totals across all rounds (not just the last n)
    for g in history_given[:start]:
        total_A += (UNIT_PER_ROUND - g)
        total_B += g
    lines.append(f"\nTotals so far: You now have {total_A} points, and player B has {total_B} points.")
    return "\n".join(lines)

def build_prompt_variant(variant_base, history_given, prompt_rounds=5, not_gamified=False):
    """
    variant_base: one of "default" or "game-theorist"
    not_gamified: if True use NOT_GAMIFIED_SYSTEM_PROMPT, else DEFAULT_SYSTEM_PROMPT
    """
    system = NOT_GAMIFIED_SYSTEM_PROMPT.strip() if not_gamified else DEFAULT_SYSTEM_PROMPT.strip()
    prev_header = "\nThe history of the game in the last {} rounds is the following:\n".format(prompt_rounds)
    prev_text = format_last_n_rounds(history_given, n=prompt_rounds)

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

# --- Model scoring / generation -------------------------------------------
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
    """
    Returns (amount:int or None, reason:str or None)
    """
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
    action_val = None
    reason = None
    if isinstance(parsed, dict):
        action = parsed.get("action")
        reason = parsed.get("reason", "")
        # Accept int or numeric string
        if isinstance(action, int):
            if 0 <= action <= UNIT_PER_ROUND:
                return action, reason
        if isinstance(action, str):
            # strip, try to parse integer
            m = re.search(r'\b(10|[0-9])\b', action)
            if m:
                amt = int(m.group(0))
                if 0 <= amt <= UNIT_PER_ROUND:
                    return amt, reason
    # fallback: search the generated text for a standalone number 0..10
    m2 = re.search(r'\b(10|[0-9])\b', gen_text)
    if m2:
        amt = int(m2.group(0))
        if 0 <= amt <= UNIT_PER_ROUND:
            return amt, "(fallback: extracted number)"
    return None, None

# --- CSV persistence & progress loading -----------------------------------
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
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

def load_csv_progress(filepath):
    """
    Read existing CSV and return a nested dict:
      progress[seed][variant] = {
          "last_round": int,
          "history_given": list of ints,
          "generosity_streak": int,
          "total_A": int,
          "total_B": int,
      }
    If the CSV doesn't exist, returns {}.
    """
    from collections import defaultdict

    if not Path(filepath).exists():
        return {}

    progress = defaultdict(dict)
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seed = int(row["seed"])
                variant = row["variant"]
                rnd = int(row["round"])
                hist_given_str = row.get("history_given", "") or ""
            except Exception:
                continue

            # keep only the row with the largest round per key
            cur = progress[seed].get(variant)
            if cur is None or rnd > cur["last_round"]:
                # parse history_given which we stored as comma-separated ints
                if hist_given_str.strip() == "":
                    hist_given = []
                else:
                    # tolerant parsing: allow JSON list or CSV
                    try:
                        hist_given = json.loads(hist_given_str)
                        if not isinstance(hist_given, list):
                            raise ValueError
                        hist_given = [int(x) for x in hist_given]
                    except Exception:
                        # fallback to CSV parse
                        hist_given = [int(x) for x in hist_given_str.split(",") if x != ""]

                total_A = sum(UNIT_PER_ROUND - g for g in hist_given)
                total_B = sum(g for g in hist_given)

                generosity_streak = 0
                for g in reversed(hist_given):
                    if g > (UNIT_PER_ROUND // 2):  # generosity defined as give > 5 (for 10-unit)
                        generosity_streak += 1
                    else:
                        break

                progress[seed][variant] = {
                    "last_round": rnd,
                    "history_given": hist_given,
                    "generosity_streak": generosity_streak,
                    "total_A": total_A,
                    "total_B": total_B,
                }

    return {s: dict(v) for s, v in progress.items()}

# ---------- Main play loop for Dictator -----------------------------------
def play_iterated_dictator(model_name, model_path, tokenizer, model, device=None, rounds=20,
                           seed=0, out_rows=None, variant="default", prompt_rounds=5,
                           initial_state=None, out_csv=None, fieldnames=None):
    """
    initial_state: optional dict as produced by load_csv_progress for this seed+variant
    out_rows: optional list buffer
    out_csv & fieldnames: if provided, this function will write each variant's
      completed rounds to CSV immediately after finishing.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    is_not_gamified = False
    if variant.endswith("_notgamified"):
        is_not_gamified = True
        vbase = variant.replace("_notgamified", "")
    else:
        vbase = variant

    # reconstruct initial state if provided
    init = initial_state or {}
    history_given = init.get("history_given", [])[:]
    generosity_streak = init.get("generosity_streak", 0)
    total_A = init.get("total_A", 0)
    total_B = init.get("total_B", 0)
    start_round = (init.get("last_round", 0) + 1) if init else 1

    rows_buffer = []

    for r in range(start_round, rounds + 1):
        print(f"[seed={seed}] [{variant}] {model_name} — Round {r}")

        v = vbase.lower()
        if v in ("default", "game-theorist"):
            prompt = build_prompt_variant(v, history_given, prompt_rounds, not_gamified=is_not_gamified)
            model_action_amt, model_reason = get_model_json_decision(model, tokenizer, prompt, device=device)
        elif v == "coa":
            def_prompt = build_prompt_variant("default", history_given, prompt_rounds, not_gamified=is_not_gamified)
            theo_prompt = build_prompt_variant("game-theorist", history_given, prompt_rounds, not_gamified=is_not_gamified)

            def_amt, def_reason = get_model_json_decision(model, tokenizer, def_prompt, device=device)
            theo_amt, theo_reason = get_model_json_decision(model, tokenizer, theo_prompt, device=device)

            def_json = json.dumps({"action": def_amt, "reason": def_reason})
            theo_json = json.dumps({"action": theo_amt, "reason": theo_reason})

            print(f"[COA] def: {def_json}  theo: {theo_json}")

            coa_prompt = COA_PROMPT_TEMPLATE.format(
                default_json=def_json,
                theorist_json=theo_json
            )
            model_action_amt, model_reason = get_model_json_decision(model, tokenizer, coa_prompt, device=device)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Score probabilities for logging (probability of each numeric choice)
        base_prompt = build_prompt_variant("default", history_given, prompt_rounds, not_gamified=is_not_gamified)
        probs = score_choice_probs_batched(model, tokenizer, base_prompt, choices=CHOICES, device=device)

        # fallback: if model didn't produce a valid amount, pick highest-prob choice
        if model_action_amt is None:
            chosen_str, chosen_prob = max(probs.items(), key=lambda kv: kv[1])
            model_action_amt = int(chosen_str)
            if model_reason is None:
                model_reason = "(fallback: token scoring)"
        else:
            chosen_prob = probs.get(str(model_action_amt), 0.0)

        given = int(model_action_amt)
        keep = UNIT_PER_ROUND - given

        total_A += keep
        total_B += given

        if given > (UNIT_PER_ROUND // 2):  # generosity definition: give > 5
            generosity_streak = generosity_streak + 1
        else:
            generosity_streak = 0

        history_given.append(given)

        row = {
            "timestamp": time.time(),
            "seed": seed,
            "model": model_name,
            "variant": variant,
            "round": r,
            "choice_prob": chosen_prob,
            "given": given,
            "kept": keep,
            "model_reason": model_reason or "",
            "generosity_streak": generosity_streak,
            "total_A": total_A,
            "total_B": total_B,
            "history_given": json.dumps(history_given),
            "not_gamified": bool(is_not_gamified),
        }

        rows_buffer.append(row)
        if out_rows is not None:
            out_rows.append(row)

        if (r % 10) == 0 or r == rounds:
            print(f"[{variant}] [{model_name}] round {r}/{rounds} given={given} kept={keep} total_A={total_A}")

    # flush to disk for this variant
    if rows_buffer and out_csv and fieldnames:
        append_rows_to_csv(out_csv, fieldnames, rows_buffer)

    return {
        "last_round": rounds,
        "history_given": history_given,
        "generosity_streak": generosity_streak,
        "total_A": total_A,
        "total_B": total_B,
    }

# --- Model path helper & runner -------------------------------------------
def resolve_model_path(model_arg):
    """
    Resolve model_arg to a path string (same semantics as before).
    """
    if any(sep in model_arg for sep in (os.sep, "/", "\\")) or model_arg.startswith("~") or model_arg.startswith("."):
        p = Path(model_arg).expanduser()
        if not p.exists():
            print(f"[Warning] provided model path {p} does not exist locally; attempting to use it anyway.")
        return str(p)

    base = Path.home() / "hf_cache" / "hf_cache"
    candidate = base / model_arg
    if candidate.exists():
        return str(candidate)

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
        p = Path(model_path)
        if p.exists() and p.is_dir():
            model_display_name = p.name
        else:
            model_display_name = str(model_arg).split("/")[-1]

    fieldnames = [
        "timestamp","seed","model","variant","round","choice_prob",
        "given","kept","model_reason","generosity_streak",
        "total_A","total_B","history_given","not_gamified"
    ]

    print(f"Resolved model_path: {model_path}")
    print(f"CSV model name: {model_display_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # load existing progress from CSV (if any)
    progress = load_csv_progress(args.out)

    for seed in range(100):
        seed_progress = progress.get(seed, {})
        # decide if seed is fully complete: every variant has last_round >= args.rounds
        fully_done = True
        for v in variants_list:
            last = seed_progress.get(v, {}).get("last_round", 0)
            if last < args.rounds:
                fully_done = False
                break
        if fully_done:
            print(f"Seed {seed} already fully completed in {args.out}; skipping.")
            continue

        out_rows = []
        for variant in variants_list:
            initial_state = seed_progress.get(variant, None)
            new_state = play_iterated_dictator(
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
                initial_state=initial_state,
                out_csv=args.out,                 # write per-variant to disk
                fieldnames=fieldnames
            )
            # update seed_progress copy so subsequent variants pick up latest totals if desired
            seed_progress[variant] = new_state

        if out_rows:
            append_rows_to_csv(args.out, fieldnames, out_rows)
        print(f"Seed {seed} completed and saved (seed wrote to {args.out}).")

# --- Entrypoint -------------------------------------------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(description="Run the Dictator-game evaluation (compatible flags with ipd_test.py)")
    p.add_argument("--out", default="dictator_results.csv",
                   help="CSV results file (will be appended to if exists)")
    p.add_argument("--rounds", type=int, default=100,
                   help="Number of rounds per seed / variant")
    p.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    p.add_argument("--variants", default="default",
                   help="comma-separated list of variants: default,game-theorist,coa or 'both'")
    p.add_argument("--prompt-rounds", type=int, default=5,
                   help="how many past rounds to include in the prompt (default 5)")
    p.add_argument("--include-not-gamified", action="store_true",
                   help="also run not-gamified versions of each selected variant")
    p.add_argument("--model", required=True,
                   help="Model path or model name to evaluate (local folder or HF id)")
    p.add_argument("--model-name", default=None,
                   help="Optional display name to write into CSV 'model' column (overrides inferred name)")
    args = p.parse_args()

    # sanitize prompt_rounds
    args.prompt_rounds = max(1, int(args.prompt_rounds))

    # normalize variants
    raw = args.variants.strip().lower()
    allowed = {"default", "game-theorist", "coa"}
    if raw in ("both", "all"):
        base_variants = ["default", "game-theorist", "coa"]
    else:
        base_variants = [v.strip() for v in raw.split(",") if v.strip() in allowed]
        if not base_variants:
            base_variants = ["default"]

    # build final variants list and optionally include not-gamified variants
    variants_list = []
    for v in base_variants:
        variants_list.append(v)
        if args.include_not_gamified:
            variants_list.append(f"{v}_notgamified")

    # call the main runner (expects the 'run_all' function from the script)
    run_all(args, variants_list)

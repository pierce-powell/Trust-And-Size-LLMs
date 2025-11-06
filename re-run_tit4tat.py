#!/usr/bin/env python3
"""
Rerun the full pipeline (LLM prompts + scoring) ONLY for existing Tit4Tat rows.

Usage (SLURM should provide the CSV path):
  python rerun_tit4tat_pipeline.py --csv /abs/path/to/results.csv --model MODEL_ARG [--model-name NAME] [--device cpu|cuda]

Important:
 - This script IMPORTS your ipd_test.py (so place it in the same directory or make it importable).
 - It will NOT add new rows. It will only overwrite existing rows whose `heuristic == "Tit4Tat"`.
 - It does not expose or change prompt_rounds — the call uses ipd_test.play_iterated_pd defaults so behavior matches your original run.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
import tempfile
import os
import csv
from collections import defaultdict
import traceback

import pandas as pd

# ----------------------------
# CLI
# ----------------------------
def main(): 
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Absolute path to existing CSV (will be overwritten atomically).")
    p.add_argument("--model", required=True, help="Model path or HF id (same arg you passed to ipd_test.py).")
    p.add_argument("--model-name", default=None, help="Optional display name to write into rows (overrides inferred).")
    p.add_argument("--device", default=None, help="cuda or cpu; default auto-detect.")
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # ----------------------------
    # Import ipd_test module (dynamic)
    # ----------------------------
    module_name = "ipd_test"
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        candidate = Path(__file__).parent / "ipd_test.py"
        if not candidate.exists():
            print("[ERROR] Could not find ipd_test module. Put this script next to ipd_test.py or ensure ipd_test is importable.", file=sys.stderr)
            sys.exit(3)
        spec = importlib.util.spec_from_file_location(module_name, str(candidate))
        ipd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ipd)
    else:
        ipd = importlib.import_module(module_name)

    # ----------------------------
    # Read CSV and find groups to rerun
    # ----------------------------
    df = pd.read_csv(csv_path, dtype=str)  # preserve original column names/order as strings
    if "heuristic" not in df.columns or "seed" not in df.columns or "variant" not in df.columns or "round" not in df.columns:
        print("[ERROR] CSV must contain columns: seed, variant, heuristic, round", file=sys.stderr)
        sys.exit(4)

    # normalize types for grouping
    df["round"] = df["round"].astype(int)
    df["seed"] = df["seed"].astype(int)

    mask = df["heuristic"].astype(str) == "Tit4Tat"
    if not mask.any():
        print("[INFO] No rows with heuristic == 'Tit4Tat' found. Nothing to do.")
        sys.exit(0)

    groups = df[mask].groupby(["seed", "variant"], sort=False)
    targets = []
    for (seed, variant), g in groups:
        max_round = int(g["round"].max())
        targets.append({"seed": int(seed), "variant": variant, "rounds": max_round})
    print(f"[INFO] Found {len(targets)} groups with Tit4Tat rows. Will re-run exactly the existing rounds for each group.")

    # ----------------------------
    # Load tokenizer & model once (use same logic as ipd_test.run_all)
    # ----------------------------
    model_arg = args.model
    model_path = ipd.resolve_model_path(model_arg)
    device = args.device or ("cuda" if (hasattr(ipd, "torch") and ipd.torch.cuda.is_available()) else "cpu")

    if args.model_name:
        model_display_name = args.model_name
    else:
        pth = Path(model_path)
        if pth.exists() and pth.is_dir():
            model_display_name = pth.name
        else:
            model_display_name = str(model_arg).split("/")[-1]

    print(f"[INFO] Loading tokenizer/model for {model_path} on {device} ...")
    tokenizer = ipd.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = ipd.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                    torch_dtype=getattr(ipd, "torch").bfloat16, device_map="auto")
    model.eval()
    print("[INFO] Model loaded.")

    # ----------------------------
    # Temporarily restrict HEURISTICS to just Tit4Tat -> tit_for_tat
    # ----------------------------
    orig_HEUR = getattr(ipd, "HEURISTICS", None)
    if orig_HEUR is None:
        print("[ERROR] ipd_test module doesn't expose HEURISTICS. Aborting.", file=sys.stderr)
        sys.exit(5)

    if not hasattr(ipd, "tit_for_tat"):
        print("[ERROR] ipd_test.py does not define tit_for_tat(). Aborting.", file=sys.stderr)
        sys.exit(6)

    ipd.HEURISTICS = {"Tit4Tat": ipd.tit_for_tat}
    print("[INFO] Restricted ipd_test.HEURISTICS to only Tit4Tat -> tit_for_tat (temporarily).")

    # ----------------------------
    # Run play_iterated_pd for each group (serial)
    # ----------------------------
    generated_by_group = {}  # (seed,variant) -> list of generated rows (dicts)
    try:
        for t in targets:
            seed = t["seed"]
            variant = t["variant"]
            rounds = t["rounds"]
            print(f"[RUN] seed={seed} variant={variant} rounds={rounds}")

            out_rows = []
            # initial_states left empty to re-create entire sequence from round 1..rounds
            try:
                ipd.play_iterated_pd(
                    model_name=model_display_name,
                    model_path=model_path,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    rounds=rounds,
                    seed=seed,
                    out_rows=out_rows,            # play_iterated_pd will append rows here
                    variant=variant,
                    # NOTE: do not pass prompt_rounds -> use ipd.play_iterated_pd default
                    initial_states={},            # fresh start
                    out_csv=None,
                    fieldnames=None
                )
            except Exception:
                print(f"[ERROR] play_iterated_pd failed for seed={seed} variant={variant}. Traceback:", file=sys.stderr)
                traceback.print_exc()
                continue

            # filter to Tit4Tat just in case (play_iterated_pd should have only produced Tit4Tat rows)
            new_rows = [r for r in out_rows if str(r.get("heuristic")) == "Tit4Tat"]
            if not new_rows:
                print(f"[WARN] No Tit4Tat rows generated for seed={seed} variant={variant}; skipping.")
                continue
            new_rows.sort(key=lambda r: int(r.get("round", 0)))
            generated_by_group[(seed, variant)] = new_rows
            print(f"[OK] Generated {len(new_rows)} Tit4Tat rows for seed={seed} variant={variant}.")

    finally:
        # restore original HEURISTICS
        ipd.HEURISTICS = orig_HEUR
        print("[INFO] Restored original HEURISTICS.")

    # ----------------------------
    # Replace existing Tit4Tat rows in the CSV (preserve CSV header/row order)
    # ----------------------------
    original_fieldnames = list(df.columns)  # preserve header and order exactly
    rows = df.to_dict(orient="records")

    # indices of Tit4Tat rows per group in original row order
    indices_by_group = defaultdict(list)
    for idx, row in enumerate(rows):
        if str(row.get("heuristic")) == "Tit4Tat":
            key = (int(row.get("seed")), row.get("variant"))
            indices_by_group[key].append(idx)

    replacements = 0
    for key, gen_rows in generated_by_group.items():
        indices = indices_by_group.get(key, [])
        if not indices:
            print(f"[WARN] No existing Tit4Tat indices found for group {key}; skipping.")
            continue
        n_replace = min(len(indices), len(gen_rows))
        print(f"[WRITE] Replacing {n_replace} existing Tit4Tat rows for group {key} (existing indices: {len(indices)}; generated: {len(gen_rows)}).")
        for i in range(n_replace):
            idx = indices[i]
            gen = gen_rows[i]
            # Build new row that uses exactly the original CSV columns (entire row replaced wrt those columns).
            new_row = {}
            for col in original_fieldnames:
                # use generated value if provided, else empty string
                val = gen.get(col, "")
                # keep simple string formatting for CSV; model may put numbers or booleans
                if val is None:
                    val = ""
                new_row[col] = val
            rows[idx] = new_row
            replacements += 1

    print(f"[INFO] Done preparing replacements. Total replaced rows: {replacements}")

    # ----------------------------
    # Atomically write back preserving original header order
    # ----------------------------
    out_dir = csv_path.parent
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=str(out_dir), newline="", suffix=".tmp") as tf:
        tmp_path = Path(tf.name)
        writer = csv.DictWriter(tf, fieldnames=original_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        tf.flush()
        try:
            os.fsync(tf.fileno())
        except Exception:
            pass

    # Replace file atomically (make small backup first)
    backup = csv_path.with_suffix(csv_path.suffix + ".bak")
    try:
        csv_path.replace(backup)
        tmp_path.replace(csv_path)
        try:
            backup.unlink()
        except Exception:
            pass
        print(f"[OK] Overwrote {csv_path} atomically with updated Tit4Tat rows.")
    except Exception as e:
        print(f"[ERROR] Atomic replace failed: {e}. Attempting fallback os.replace.", file=sys.stderr)
        try:
            os.replace(str(tmp_path), str(csv_path))
            print(f"[WARN] Used fallback os.replace to write {csv_path}.")
        except Exception as e2:
            print(f"[CRITICAL] Fallback write also failed: {e2}", file=sys.stderr)
            sys.exit(7)

    print("[DONE] All done. Exiting.")


if __name__ == "__main__":
    main()
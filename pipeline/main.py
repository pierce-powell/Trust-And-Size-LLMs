# pipeline/main.py
import argparse
import yaml
from pipeline.utils import get_device, set_seed
from pipeline.model_loader import ModelLoader
from pipeline.ipd import LLMAgent, IPDGame, BASE_STRATEGIES
from pipeline.metrics import get_metrics_by_names
import os
import json
from collections import defaultdict
import torch

def load_prompts(prompts_cfg):
    if isinstance(prompts_cfg, str) and os.path.exists(prompts_cfg):
        with open(prompts_cfg, 'r') as f:
            data = json.load(f)
        return data
    else:
        return prompts_cfg or {}

def make_llm_agent_from_instance(inst, prompt_templates, max_new_tokens=32):
    """
    inst is a dict returned by ModelLoader.load_instance containing 'model' and 'tokenizer' and 'device'.
    """
    return LLMAgent(model=inst["model"], tokenizer=inst["tokenizer"], prompt_templates=prompt_templates,
                    device=inst.get("device", None), max_new_tokens=max_new_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pipeline/config_ipd.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device_default = get_device()
    print("Default device:", device_default)

    loader = ModelLoader()
    metadata = loader.prepare_metadata(cfg["models"])
    keys = loader.list_keys()
    print("Prepared metadata for keys:", keys)

    prompt_sets = load_prompts(cfg.get("prompts"))
    rounds = cfg["run"].get("rounds", 40)
    prompt_count = cfg["run"].get("prompts_per_model", 3)
    max_new_tokens = cfg["run"].get("max_new_tokens", 16)

    metric_names = cfg.get("metrics", ["trust", "cooperation_streak"])
    metrics = get_metrics_by_names(metric_names)

    base_strategies_list = cfg.get("base_strategies_list", ["TFT", "2TFT"])

    # results storage
    all_matches = defaultdict(list)   # key -> list of match dicts (same as prior)
    # breakdowns: key -> prompt_index -> opponent -> list of match_results for that pair
    breakdown = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # iterate keys & prompts
    for key in keys:
        for prompt_index in range(prompt_count):
            # 1) LLM vs itself
            # For match we load both instances (same key twice is fine, instance reused)
            inst_A = loader.load_instance(key, device=device_default, use_auth_token=None)
            # Note: if same key as opponent, we only need one instance; reuse it
            agentA = make_llm_agent_from_instance(inst_A, prompt_sets.get(inst_A.get("alias"), []), max_new_tokens=max_new_tokens)
            agentB = agentA  # reuse same instance for self-play (safe)
            g = IPDGame(agentA, agentB, rounds=rounds, seed=cfg.get("seed"))
            res = g.play(p1_prompt_index=prompt_index, p2_prompt_index=prompt_index)
            all_matches[key].append({"opponent": key, "prompt_index": prompt_index, "result": res})
            breakdown[key][prompt_index][key].append(res)
            # unload instance (we will unload at end of outer loop to be safe)
            loader.unload_instance(key)

            # 2) LLM vs other LLMs
            for other in keys:
                if other == key:
                    continue
                # load both models (may be the same device)
                inst_A = loader.load_instance(key, device=device_default, use_auth_token=None)
                inst_B = loader.load_instance(other, device=device_default, use_auth_token=None)
                promptsA = prompt_sets.get(inst_A.get("alias"), [])
                promptsB = prompt_sets.get(inst_B.get("alias"), [])
                agentA = make_llm_agent_from_instance(inst_A, promptsA, max_new_tokens=max_new_tokens)
                agentB = make_llm_agent_from_instance(inst_B, promptsB, max_new_tokens=max_new_tokens)
                g = IPDGame(agentA, agentB, rounds=rounds, seed=cfg.get("seed"))
                res = g.play(p1_prompt_index=prompt_index, p2_prompt_index=prompt_index)
                all_matches[key].append({"opponent": other, "prompt_index": prompt_index, "result": res})
                breakdown[key][prompt_index][other].append(res)
                # unload both instances
                loader.unload_instance(key)
                loader.unload_instance(other)

            # 3) LLM vs base strategies
            for s in base_strategies_list:
                strat_fn = BASE_STRATEGIES.get(s)
                if strat_fn is None:
                    print(f"Warning: base strategy {s} not found, skipping.")
                    continue
                inst_A = loader.load_instance(key, device=device_default, use_auth_token=None)
                agentA = make_llm_agent_from_instance(inst_A, prompt_sets.get(inst_A.get("alias"), []), max_new_tokens=max_new_tokens)
                agentB = strat_fn
                g = IPDGame(agentA, agentB, rounds=rounds, seed=cfg.get("seed"))
                res = g.play(p1_prompt_index=prompt_index, p2_prompt_index=0)
                all_matches[key].append({"opponent": s, "prompt_index": prompt_index, "result": res})
                breakdown[key][prompt_index][s].append(res)
                loader.unload_instance(key)

    # compute metrics aggregated (overall) and breakdown per prompt/opponent
    all_summaries = {}
    detailed = {}

    for key, matches in all_matches.items():
        match_results = [m["result"] for m in matches]
        key_summary = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__
            metric_vals = metric.compute(match_results, target_player_index=0)
            key_summary[metric_name] = metric_vals
        all_summaries[key] = key_summary

        # detailed per-match summary
        detailed[key] = [
            {"opponent": m["opponent"], "prompt_index": m["prompt_index"],
             "scores": m["result"].get("scores"), "cooperations": m["result"].get("cooperations"),
             "rounds": m["result"].get("rounds")}
            for m in matches
        ]

    # breakdown metrics per prompt/opponent
    breakdown_metrics = {}
    for key, prompt_map in breakdown.items():
        breakdown_metrics[key] = {}
        for pidx, opp_map in prompt_map.items():
            breakdown_metrics[key].setdefault(str(pidx), {})
            for opp, match_list in opp_map.items():
                # match_list is list of result dicts; compute metrics per this list
                match_results = match_list
                metric_results = {}
                for metric in metrics:
                    metric_name = metric.__class__.__name__
                    mvals = metric.compute(match_results, target_player_index=0)
                    metric_results[metric_name] = mvals
                breakdown_metrics[key][str(pidx)][opp] = {
                    "n_matches": len(match_list),
                    "metrics": metric_results
                }

    out_path = cfg.get("out_path", "pipeline/ipd_results.json")
    out_obj = {
        "summary": all_summaries,
        "details": detailed,
        "breakdown": breakdown_metrics
    }
    with open(out_path, 'w') as f:
        json.dump(out_obj, f, indent=2)
    print("Saved results to", out_path)

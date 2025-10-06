import argparse
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

def download_from_config(config_path: str, cache_dir: str = "./hf_cache"):
    # Load config YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", [])
    if not models:
        print("No models found in config.")
        return

    os.makedirs(cache_dir, exist_ok=True)

    for model_cfg in models:
        family = model_cfg.get("family", "unknown")
        alias = model_cfg.get("alias", family)
        sizes = model_cfg.get("sizes", [])
        if not sizes:
            print(f"Skipping {alias}: no sizes defined.")
            continue

        for size in sizes:
            model_id = size.get("id")
            size_name = size.get("size_name", "unknown")
            if not model_id:
                print(f"Skipping {alias} {size_name}: no id provided.")
                continue

            print(f"\n=== Downloading {alias} ({size_name}) -> {model_id} ===")
            try:
                # First attempt with trust_remote_code=True (to fetch custom files)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                print(f"Successfully downloaded with trust_remote_code=True")
            except Exception as e:
                print(f" trust_remote_code=True failed: {e}")
                print(f"Retrying without it (for standard models)...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        trust_remote_code=False
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        trust_remote_code=False
                    )
                    print("Successfully downloaded standard model")
                except Exception as e2:
                    print(f"Failed both attempts for {alias} ({size_name}): {e2}")
                    continue

            # Save metadata for record-keeping (optional)
            model_dir = os.path.join(cache_dir, model_id.replace("/", "--"))
            meta_path = os.path.join(model_dir, "download_meta.json")
            os.makedirs(model_dir, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump({
                    "model_id": model_id,
                    "alias": alias,
                    "size": size_name,
                    "uses_remote_code": "trust_remote_code=True" in str(tokenizer.__class__)
                }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace models listed in a config YAML.")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline YAML config (e.g. pipeline/test_ipd.yaml)")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="Where to cache models locally")
    args = parser.parse_args()

    download_from_config(args.config, args.cache_dir)

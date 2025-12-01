#!/usr/bin/env python3
"""
Download a set of large open-weight models from Hugging Face
and cache them locally.

This script:
- Authenticates with your Hugging Face token (needed for gated models)
- Downloads models via snapshot_download (resumable)
- Skips any model that isn't publicly accessible or fails gracefully
"""

from huggingface_hub import snapshot_download, login
import os

# === USER CONFIGURATION ===
HF_TOKEN = "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"
root = r"C:\Users\Pierce\hf_cache\models\QWEN_mini"

# Authenticate
login(token=HF_TOKEN)

# Target repos
repos = {
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",


}

"""    
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama-3.1-3B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama-3.1-405B": "meta-llama/Llama-3.1-405B-Instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    """

os.makedirs(root, exist_ok=True)

# === MAIN LOOP ===
for name, repo in repos.items():
    dest = os.path.join(root, name)
    print(f"\n=== Downloading {repo} ===")
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            repo_type="model",
            token=HF_TOKEN,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"Finished: {repo}")
    except Exception as e:
        print(f"Failed to download {repo}: {e}")

print("\nAll downloads attempted. Models stored in:", root)

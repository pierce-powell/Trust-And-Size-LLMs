#!/bin/bash
#SBATCH --job-name=hf-precache
#SBATCH --output=precache-%J.out
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G

source venv/bin/activate
export HF_TOKEN="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"
export HF_HOME="/lustre/fs1/pi047867/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
models = [
  "Qwen/Qwen2.5-0.5B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen2.5-7B-Instruct",
  "meta-llama/Llama-3.2-1B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
  "meta-llama/Meta-Llama-3-8B-Instruct"
]
for m in models:
    print("Caching:", m)
    AutoTokenizer.from_pretrained(m, use_auth_token=True)
    AutoModelForCausalLM.from_pretrained(m, use_auth_token=True)
print("Done")
PY

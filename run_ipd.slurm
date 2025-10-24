#!/bin/bash
#SBATCH --job-name=ipd_test
#SBATCH --output=ipd_test.%j.out
#SBATCH --error=ipd_test.%j.err
#SBATCH --time=02:00:00
#SBATCH --constraint=h100        # or the specific partition name for H100s on your cluster
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                    # more RAM helps for large models
#SBATCH --nodes=1                     # ensure both GPUs are on the same node
# Load the cluster python module you normally use
module load python  # adjust version name if needed, e.g., python/3.11

# Activate your venv
source ~/qwen_env/bin/activate

# Also tell HF/transformers to avoid network (we're using local model folders)
export HF_HOME=~/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# (Optional) make sure pip deps are installed in the venv
pip install -U transformers torch --no-cache-dir
pip install accelerate


# Run
python ipd_test.py --out results.csv --rounds 100 --variants default,game-theorist,COA


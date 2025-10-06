#!/bin/bash
#SBATCH --job-name=ipd-test
#SBATCH --output=ipd-test-%J.out
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# --- Environment setup ---
source venv/bin/activate
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH

# --- Hugging Face local cache configuration ---
# Use your uploaded cache (already contains model files)
export HF_HOME="$HOME/hf_home"
export TRANSFORMERS_CACHE="$HOME/hf_cache"
mkdir -p "$TRANSFORMERS_CACHE"

# (Optional) disable parallelism for safety on SLURM
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Run your job ---
echo "Launching pipeline..."
echo "Working directory: $(pwd)"
echo "Using cache: $TRANSFORMERS_CACHE"

python -m pipeline.main --config pipeline/test_ipd.yaml

echo "====== sbatch job end ======"
date

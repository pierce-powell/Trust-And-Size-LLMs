#!/bin/bash
#SBATCH --job-name=ipd-test-cpu
#SBATCH --output=ipd-test-cpu-%J.out
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# No --gres line (CPU-only)

# activate environment
source venv/bin/activate

# Hugging Face auth & cache (set your token)
export HF_TOKEN="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"
export HF_HOME="/lustre/fs1/$USER/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# make git single-threaded (avoid login-node thread issues)
git config --global pack.threads "1"

# run the test config you added earlier
python pipeline/main.py --config pipeline/test_ipd.yaml

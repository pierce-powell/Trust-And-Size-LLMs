#!/bin/bash
#SBATCH --job-name=ipd-test
#SBATCH --output=ipd-test-%J.out
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

source venv/bin/activate

# Hugging Face auth + cache
export HF_TOKEN="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"
export HF_HOME="/lustre/fs1/$USER/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Run the test config
python pipeline/main.py --config pipeline/test_ipd.yaml

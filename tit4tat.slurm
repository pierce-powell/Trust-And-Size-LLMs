#!/bin/bash
#SBATCH --job-name=rerun_tit4tat
#SBATCH --output=logs/rerun_tit4tat.%j.out
#SBATCH --error=logs/rerun_tit4tat.%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load python
source ~/qwen_env/bin/activate

# caches and offline mode like before
export HF_HOME=~/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CSV_PATH="$PWD/results_qwen_32B_final.csv"
MODEL_PATH="$HOME/hf_cache/hf_cache/QWEN_mini/Qwen2.5-0.5B"
MODEL_NAME="QWEN2.5-0.5B"

python re-run_tit4tat.py --csv "$CSV_PATH" --model "$MODEL_PATH" --model-name "$MODEL_NAME" --device cuda

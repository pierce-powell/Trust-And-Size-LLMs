#!/bin/bash
#SBATCH --job-name=run_dictator
#SBATCH --output=run_dicator.%j.out
#SBATCH --error=run_dictator.%j.err
#SBATCH --time=04:00:00
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --nodes=1

module load python   # adjust to your cluster module system
source ~/qwen_env/bin/activate

# Make HF/transformers offline (using local model folders)
export HF_HOME=~/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# accelerate config
export ACCELERATE_CONFIG_FILE=/home/pi047867/Trust-And-Size-LLMs/default_config.yaml

echo $ACCELERATE_CONFIG_FILE


# limit MKL/OMP threads to match cpus-per-task
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Pin GPUs (optional but good practice on multi-tenant nodes)
export CUDA_VISIBLE_DEVICES=0,1

# Install deps if not already present. Prefer installing once outside batch jobs.
pip install -U transformers torch accelerate --no-cache-dir


MODEL_PATH=$HOME/hf_cache/hf_cache/QWEN_mini/Qwen2.5-0.5B
MODEL_NAME=QWEN2.5-0.5B

accelerate launch run_dictator.py \
  --model "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --out results_qwen_05B_final.csv \
  --rounds 100 \
  --variants default,game-theorist \
  --include-not-gamified
#!/bin/bash
#SBATCH --job-name=deepseek_ipd
#SBATCH --output=deepseek_ipd.%j.out
#SBATCH --error=deepseek_ipd.%j.err
#SBATCH --time=50:00:00
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --nodes=1

source ~/miniconda3/etc/profile.d/conda.sh
source ~/Trust/bin/activate

# Make HF/transformers offline (using local model folders)
export HF_HOME=~/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# accelerate config
export ACCELERATE_CONFIG_FILE=/home/ky537465/Trust-And-Size-LLMs/default_config.yaml

echo $ACCELERATE_CONFIG_FILE

# limit MKL/OMP threads to match cpus-per-task
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Pin GPUs (optional but good practice on multi-tenant nodes)
export CUDA_VISIBLE_DEVICES=0,1

# Install deps if not already present. Prefer installing once outside batch jobs.
pip install -U transformers torch accelerate --no-cache-dir


MODEL_PATH=~/Deepseek-6.7B
MODEL_NAME=Deepseek-6.7B

accelerate launch run_ipd.py \
  --model "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --out deepseek-6-7B-IPD.csv \
  --rounds 100 \
  --variants default,game-theorist \
  --include-not-gamified
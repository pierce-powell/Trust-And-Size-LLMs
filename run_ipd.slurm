#!/bin/bash
#SBATCH --job-name=ipd_test
#SBATCH --output=ipd_test.%j.out
#SBATCH --error=ipd_test.%j.err
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
export ACCELERATE_CONFIG_FILE=~/home/pi047867/Trust-And-Size-LLMs/default_config.yaml


# limit MKL/OMP threads to match cpus-per-task
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Pin GPUs (optional but good practice on multi-tenant nodes)
export CUDA_VISIBLE_DEVICES=0,1

# Install deps if not already present. Prefer installing once outside batch jobs.
pip install -U transformers torch accelerate --no-cache-dir
  
# ---- Launch the script using accelerate ----
# accelerate will spawn 2 processes (one per GPU) according to the config above.
accelerate launch ipd_test.py --out results_qwen_32.csv --rounds 100 --variants default,game-theorist,COA

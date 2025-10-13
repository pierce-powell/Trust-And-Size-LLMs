#!/bin/bash
#SBATCH --job-name=ipd_test
#SBATCH --output=ipd_test.%j.out
#SBATCH --error=ipd_test.%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1              # remove if no GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

# Load the cluster python module you normally use
module load python  # adjust version name if needed, e.g., python/3.11

# Activate your venv
source ~/qwen_env/bin/activate

# Thread / process limits to avoid OpenBLAS RLIMIT_NPROC issues
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Also tell HF/transformers to avoid network (we're using local model folders)
export HF_HOME=~/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# (Optional) make sure pip deps are installed in the venv
pip install -U transformers torch --no-cache-dir

# Run
python ~/ipd_test.py --out ~/ipd_ipd_results.csv --rounds 30

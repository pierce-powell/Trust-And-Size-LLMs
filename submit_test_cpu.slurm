#!/bin/bash
#SBATCH --job-name=ipd-test-cpu
#SBATCH --output=ipd-test-cpu-%J.out
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# no --gres (CPU-only test)

# (Optional) load a python module if your site requires it:
# module load python/3.10

# Activate the venv we created interactively (adjust path if you placed venv elsewhere)
source $HOME/Trust-And-Size-LLMs/venv/bin/activate

# Ensure Python inside venv is used
echo "Using python: $(which python) $(python --version)"

# Hugging Face auth & cache (use HOME-based cache so no permission issues)
export HF_TOKEN="hf_xxxYOURTOKENxxx"
export HF_HOME="$HOME/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# make git single-threaded (avoid login-node thread issues)
git config --global pack.threads "1"

# Run the test config (use python3 -u for unbuffered output)
python3 -u pipeline/main.py --config pipeline/test_ipd.yaml

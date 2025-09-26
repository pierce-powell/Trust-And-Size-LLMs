#!/bin/bash
#SBATCH --job-name=ipd-test-cpu
#SBATCH --output=ipd-test-cpu-%J.out
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# (no --gres for CPU-only)

# ---- debug info printed to the log for quick diagnostics ----
echo "====== sbatch job start ======"
date
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "Current working directory before cd: $(pwd)"

# ---- go to the repo root so relative imports work ----
cd $HOME/Trust-And-Size-LLMs || { echo "ERROR: repo not found at $HOME/Trust-And-Size-LLMs"; exit 2; }
echo "PWD after cd: $(pwd)"
echo "Listing repo top-level:"
ls -la .

# ---- make Python package importable from anywhere in job ----
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# ---- activate the venv created interactively earlier ----
if [ -f "$PWD/venv/bin/activate" ]; then
  source "$PWD/venv/bin/activate"
else
  echo "ERROR: venv not found at $PWD/venv. Create it interactively with: python3 -m venv venv; source venv/bin/activate; pip install -r requirements.txt"
  exit 3
fi

echo "Using python: $(which python) $(python --version)"
echo "pip freeze (first 40 lines):"
pip freeze | head -n 40

# ---- limit BLAS/OMP threads to avoid pthread_create / RLIMIT_NPROC issues ----
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
echo "Thread envs: OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS OMP_NUM_THREADS=$OMP_NUM_THREADS"

# ---- Hugging Face caches under your HOME (avoid permission problems) ----
export HF_TOKEN="hf_xxxYOURTOKENxxx"   # replace with your token or set in environment securely
export HF_HOME="$HOME/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

# ---- reduce git threads in case the job does any git fetches ----
git config --global pack.threads "1" || true

# ---- run the pipeline as a module (ensures package imports work) ----
echo "Launching pipeline..."
python3 -u -m pipeline.main --config pipeline/test_ipd.yaml

echo "====== sbatch job end ======"
date

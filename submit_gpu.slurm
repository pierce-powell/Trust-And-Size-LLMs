#!/bin/bash
#SBATCH --job-name=hf-eval-pipeline
#SBATCH --output=hf-eval-%J.out
#SBATCH --time=02:00:00          # HH:MM:SS - adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # CPU cores used for dataloading / preprocessing
#SBATCH --gres=gpu:1            # request 1 GPU (STOKES note: don't go beyond 2 GPUs). :contentReference[oaicite:3]{index=3}
#SBATCH --mem=32G               # adjust memory as needed

# Optional: load modules (check module avail on Stokes)
# module load python/3.10
# module load cuda/12.1

#My HF Token
export HF_TOKEN="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"

# Activate venv
source ./venv/bin/activate

# Print some useful environment/debug info
echo "SLURM_JOBID: $SLURM_JOBID"
echo "NODES: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

export HF_HOME=/path/to/your/hf_home
export TRANSFORMERS_CACHE=/path/to/your/transformers_cache
export HF_DATASETS_CACHE=/path/to/your/hf_datasets_cache

# Run python entrypoint (main.py accepts --config)
python pipeline/main.py --config pipeline/config_example.yaml

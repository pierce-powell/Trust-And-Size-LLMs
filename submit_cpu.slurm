#!/bin/bash
#SBATCH --job-name=hf-eval-pipeline-cpu
#SBATCH --output=hf-eval-cpu-%J.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# module load python/3.10

source ./venv/bin/activate

python pipeline/main.py --config pipeline/config_example.yaml

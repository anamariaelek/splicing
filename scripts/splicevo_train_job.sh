#!/bin/bash
#SBATCH --job-name=splicevo_train
#SBATCH --output=logs/splicevo_train_%j.log
#SBATCH --error=logs/splicevo_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nice=10  # Lower priority to be considerate

# Activate environment
source ~/.bashrc
conda activate splicevo

# Set resource limits
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run training
python splicevo_train.py

echo "Training completed at $(date)"

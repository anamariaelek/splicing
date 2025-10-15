#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=10gb
#SBATCH --job-name=spliser
#SBATCH --array=1-94
#SBATCH --output=logs/spliser_%A_%a.out
#SBATCH --error=logs/spliser_%A_%a.err

# Mouse 1-94
# Human 1-196
# Macaque 1-114
# Rat 1-102
# Rabbit 1-102
# Opossum 1-97
# Chicken 1-65

# Define base path
DIR="$HOME/projects/splicing/data/spliser"
SPECIES=Mouse

# Get the group name for this array task
GROUP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR/groups_${SPECIES}.txt")

# Run spliser.sh for this specific group
bash spliser.sh "$GROUP" > logs/"${GROUP}".log 2>&1

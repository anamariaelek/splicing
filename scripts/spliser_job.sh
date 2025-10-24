#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=5gb
#SBATCH --job-name=spliser
#SBATCH --array=70,95,96,107,111
#SBATCH --output=logs/spliser_%A_%a.out

# Number of groups per species:
# Mouse 1-94
# Human 1-196
# Macaque 1-114
# Rat 1-102
# Rabbit 1-102
# Opossum 1-97
# Chicken 1-65

# Activate conda environment
source activate spliser

# Define base path
DIR="$HOME/projects/splicing/data/spliser"
SPECIES=Macaque

# Get the group name for this array task
GROUP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR/groups_${SPECIES}.txt")

# Run spliser.sh for this specific group
bash spliser.sh "$GROUP" > logs/"${GROUP}".log 2>&1

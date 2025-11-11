#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=5gb
#SBATCH --job-name=spliser
#SBATCH --array=1-42
#SBATCH --output=logs/Chicken/spliser_%A_%a.out

# Number of groups per species:
# Human 1-92
# Mouse 1-94
# Macaque 1-114
# Rat 1-90
# Rabbit 1-92
# Opossum 1-82
# Chicken 1-42

# Exit on erroe
set -e

# Activate conda environment
source activate spliser

# Define base path
DIR="$HOME/projects/splicing/data/spliser"

# Define species to process
SPECIES=Chicken

# Prepare groups of samples
awk -F'\t' 'NR>1 {print $10}' "$DIR/samples.txt" | sort | uniq > "$DIR/groups.txt"

# Prepare groups per species
grep -w "$SPECIES" "$DIR/groups.txt" > "$DIR/groups_${SPECIES}.txt"

# Get the group name for this array task
GROUP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$DIR/groups_${SPECIES}.txt")

# Run spliser.sh for this specific group
bash spliser.sh "$GROUP" > logs/"${SPECIES}"/"${GROUP}".log 2>&1

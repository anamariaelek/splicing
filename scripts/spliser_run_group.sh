#!/bin/sh

# Define base path
DIR="$HOME/projects/splicing/data/spliser"

# Prepare groups of samples
awk -F'\t' 'NR>1 {print $10}' "$DIR/samples.txt" | sort | uniq > "$DIR/groups.txt"

# Prepare groups per species
# from groups.txt, extract lines with species name (Human, Mouse, Rat, Rabbit, Opossum, Chicken, Macaque)
awk -F'\t' 'NR>1 {print $4}' "$DIR/samples.txt" | sort | uniq > "$DIR/species.txt"
for species in $(cat "$DIR/species.txt"); do
    grep -w "$species" "$DIR/groups.txt" > "$DIR/groups_${species}.txt"
done

# Run with:
# cat groups.txt | parallel -j 10 'logname=$(basename {}); bash spliser.sh {} > logs/${logname}.log 2>&1'  &
for species in $(cat "$DIR/species.txt"); do
    cat "$DIR/groups_${species}.txt" | parallel -j 10 'logname=$(basename {}); bash spliser.sh {} > logs/${logname}.log 2>&1' &
done
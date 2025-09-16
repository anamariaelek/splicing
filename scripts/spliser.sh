#!/bin/sh

# Data directories
BAM_DIR=$HOME/sds/sd17d003/Margarida/EvoDevoData/
GTF_DIR=$HOME/sds/sd17d003/Anamaria/genomes/mazin/gtf/
OUT_DIR=$HOME/projects/splicing/results/spliser/

# Load samples from txt file samples.txt
# Format: Library\tUsed\tSpecies_sci\tSpecies\tOrgan\tDevelopmental_stage\tStage_detail\tSex\tFilename\tGroup\tTotal_number_reads\tTotal_number_aligned_reads\t%_aligned_reads
samples_file=$HOME/projects/splicing/data/spliser/samples.txt

# Column indices (1-based):
# Species = $4, Species_sci = $3, Filename = $9, Group = $10

# Accept optional group argument
if [ -n "$1" ]; then
    groups="$1"
else
    groups=$(awk -F'\t' 'NR>1 {print $10}' "$samples_file" | sort | uniq)
fi

for group in $groups; do

    # Log start time
    start_time=$(date +%s)
    echo "[$(date)] Starting processing for $group"

    species=$(awk -F'\t' -v grp="$group" 'NR>1 && $10==grp {print $4; exit}' "$samples_file")
    species_sci=$(awk -F'\t' -v grp="$group" 'NR>1 && $10==grp {print $3; exit}' "$samples_file")

    # Build path to GTF file
    gtf="$GTF_DIR"/"$species_sci".gtf.gz

    # Where to save results
    out_dir="$OUT_DIR"/"$species_sci"
    mkdir -p "$out_dir"

    # Get BAM files for this group
    bams=$(awk -F'\t' -v grp="$group" -v path="$BAM_DIR"/"$species"/ '
    NR>1 && $10==grp {
        if (n++) printf ",";
        printf "%s%s", path, $9
    }' "$samples_file")
    
    # Check if preCombineIntrons output exists
    precombine_out="$out_dir"/"$group".introns.tsv
    if [ ! -f "$precombine_out" ]; then
        spliser preCombineIntrons -L "$bams" -o "$out_dir"/"$group" --isStranded -s rf -A "$gtf"
        echo "[$(date)] Finished preCombineIntrons for $group"
    else
        echo "[INFO] Skipping preCombineIntrons for $group: $precombine_out exists."
    fi

    # Now make $bams iterable
    bams=$(echo "$bams" | tr ',' '\n')

    # Run process for each BAM in the group, only if output does not exist
    for bam in $bams; do
        name=$(basename "$bam" .sorted.bam)
        output="$out_dir"/"$name".SpliSER.tsv
        if [ ! -f "$output" ]; then
            spliser process \
                -B "$bam" \
                -I "$out_dir"/"$group".introns.tsv \
                --isStranded -s rf \
                -A "$gtf" -t transcript \
                -o "$out_dir"/"$name"
            echo "[$(date)] Finished process for $name"
        else
            echo "[INFO] Skipping process for $bam: $output exists."
        fi
    done

    # Make SamplesFile.tsv file if it doesn't exist
    # it should contain info for all bam files in the format:
    # Sample1\t/path/to/Sample1.SpliSER.tsv\t/path/to/bams/Sample1.bam
    # Sample2\t/path/to/Sample2.SpliSER.tsv\t/path/to/bams/Sample2.bam
    sample_file="$out_dir"/"$group".SamplesFile.tsv
    if [ ! -f "$sample_file" ]; then
        for bam in $bams; do
            name=$(basename "$bam" .sorted.bam)
            echo -e "$name\t$out_dir/$name.SpliSER.tsv\t$bam" >> "$sample_file"
        done
    else
        echo "[INFO] Skipping SamplesFile creation for $group: $sample_file exists."
    fi

    # Combine outputs
    combined_file="$out_dir"/"$group".combined.tsv
    if [ ! -f $combined_file ]; then
        spliser combine -S $sample_file -o "$out_dir"/"$group" --isStranded -s rf
        echo "[$(date)] Finished combine for $group"
    else
        echo "[INFO] Skipping combine for $group: $out_dir/$group.combined.tsv exists."
    fi

    # Log time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))
    echo "[$(date)] Finished processing $group in ${hours}h ${minutes}m ${seconds}s."
done
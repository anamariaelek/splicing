import time
import pickle
import os
import numpy as np
from datetime import datetime

from splicevo.data import MultiGenomeDataLoader

# Create output directory for results
output_dir = "/home/elek/projects/splicing/results/data_processing"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting data processing at {datetime.now()}")
print("=" * 60)

# Step 1: Initialize loader
print("Step 1: Initializing MultiGenomeDataLoader...")
step1_start = time.time()
loader = MultiGenomeDataLoader()
step1_time = time.time() - step1_start
print(f"✓ Loader initialized in {step1_time:.2f} seconds")
print()

# Step 2: Add genomes
print("Step 2: Adding genomes...")
step2_start = time.time()

print("  Adding human genome...")
human_start = time.time()
loader.add_genome(
    genome_id="human_GRCh37",
    genome_path="/home/elek/sds/sd17d003/Anamaria/genomes/mazin/fasta/Homo_sapiens.fa.gz", 
    gtf_path="/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/Homo_sapiens.gtf.gz",
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', 'X', 'Y', 'MT'],
    metadata={"species": "homo_sapiens", "assembly": "GRCh37"}
)
human_time = time.time() - human_start
print(f"  ✓ Human genome added in {human_time:.2f} seconds")

print("  Adding mouse genome...")
mouse_start = time.time()
loader.add_genome(
    genome_id="mouse_GRCm38",
    genome_path="/home/elek/sds/sd17d003/Anamaria/genomes/mazin/fasta/Mus_musculus.fa.gz",
    gtf_path="/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/Mus_musculus.gtf.gz",
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'X', 'Y', 'MT'],
    metadata={"species": "mus_musculus", "assembly": "GRCm38"}
)
mouse_time = time.time() - mouse_start
print(f"  ✓ Mouse genome added in {mouse_time:.2f} seconds")

step2_time = time.time() - step2_start
print(f"✓ All genomes added in {step2_time:.2f} seconds")
print()

# Step 3: Add usage files
print("Step 3: Adding usage files...")
step3_start = time.time()

# Add human usage files if they exist
print("  Adding human usage files...")
human_usage_start = time.time()
try:
    loader.add_usage_file(
        genome_id="human_GRCh37", 
        usage_file="/home/elek/projects/splicing/results/spliser/Human.Cerebellum.29ypb.combined.nochr.tsv",
        tissue="Cerebellum",
        timepoint="29ypb"
    )
    loader.add_usage_file(
        genome_id="human_GRCh37",
        usage_file="/home/elek/projects/splicing/results/spliser/Human.Cerebellum.0dpb.combined.nochr.tsv",
        tissue="Cerebellum", 
        timepoint="0dpb"
    )
    loader.add_usage_file(
        genome_id="human_GRCh37",
        usage_file="/home/elek/projects/splicing/results/spliser/Human.Heart.0dpb.combined.nochr.tsv", 
        tissue="Heart",
        timepoint="0dpb"
    )
    loader.add_usage_file(
        genome_id="human_GRCh37",
        usage_file="/home/elek/projects/splicing/results/spliser/Human.Kidney.0dpb.combined.nochr.tsv", 
        tissue="Kidney",
        timepoint="0dpb"
    )
    human_usage_time = time.time() - human_usage_start
    print(f"  ✓ Human usage files added in {human_usage_time:.2f} seconds")
except FileNotFoundError as e:
    print(f"  ⚠ Human usage files not found: {e}")
    human_usage_time = time.time() - human_usage_start  

# Add mouse usage files if they exist
print("  Adding mouse usage files...")
mouse_usage_start = time.time()
try:
    loader.add_usage_file(
        genome_id="mouse_GRCm38",
        usage_file="/home/elek/projects/results/spliser/Mouse.Cerebellum.4wpb.combined.nochr.tsv",
        tissue="Cerebellum",
        timepoint="4wpb"
    )
    loader.add_usage_file(
        genome_id="mouse_GRCm38",
        usage_file="/home/elek/projects/results/spliser/Mouse.Cerebellum.0dpb.combined.nochr.tsv",
        tissue="Cerebellum",
        timepoint="0dpb"
    )
    loader.add_usage_file(
        genome_id="mouse_GRCm38", 
        usage_file="/home/elek/projects/results/spliser/Mouse.Heart.0dpb.combined.nochr.tsv",
        tissue="Heart",
        timepoint="0dpb"
    )
    mouse_usage_time = time.time() - mouse_usage_start
    print(f"  ✓ Mouse usage files added in {mouse_usage_time:.2f} seconds")
except FileNotFoundError as e:
    print(f"  ⚠ Mouse usage files not found: {e}")
    mouse_usage_time = time.time() - mouse_usage_start

step3_time = time.time() - step3_start
print(f"✓ Usage files processed in {step3_time:.2f} seconds")

# Show available conditions
conditions_df = loader.get_available_conditions()
print("Available conditions:")
print(conditions_df)
print()

# Step 4: Load all genome data
print("Step 4: Loading all genome data...")
step4_start = time.time()

loader.load_all_genomes_data(
    max_transcripts_per_genome=None  # Load all transcripts
)

step4_time = time.time() - step4_start
print(f"✓ All genome data loaded in {step4_time:.2f} seconds")

# Show summary statistics
print("\nData summary:")
summary = loader.get_summary()
print(summary)
print()

# Step 5: Convert to arrays
print("Step 5: Converting to arrays with windowing...")
step5_start = time.time()

sequences, labels, usage_arrays, metadata = loader.to_arrays(
    window_size=1000,
    context_size=4500,
    n_workers=4
)

step5_time = time.time() - step5_start
print(f"✓ Data converted to arrays in {step5_time:.2f} seconds")
print(f"  Shape of sequences: {sequences.shape}")
print(f"  Shape of labels: {labels.shape}")
print(f"    Labels format: [:, :, 0] = donor sites, [:, :, 1] = acceptor sites")
print(f"  Shape of usage_arrays['alpha']: {usage_arrays['alpha'].shape}")
print(f"  Shape of usage_arrays['beta']: {usage_arrays['beta'].shape}")
print(f"  Shape of usage_arrays['sse']: {usage_arrays['sse'].shape}")
print(f"  Shape of metadata: {metadata.shape}")

# Get usage array info
usage_info = loader.get_usage_array_info(usage_arrays)
print(f"  Available conditions: {[c['display_name'] for c in usage_info['conditions']]}")
print()

# Step 6: Save the processed data
print("Step 6: Saving processed data...")
save_start = time.time()

np.savez_compressed(
    os.path.join(output_dir, "processed_data.npz"),
    sequences=sequences, 
    labels=labels,
    usage_alpha=usage_arrays['alpha'],
    usage_beta=usage_arrays['beta'],
    usage_sse=usage_arrays['sse']
)

metadata.to_csv(os.path.join(output_dir, "metadata.csv.gz"), index=False, compression='gzip')

# Save usage info
import json
with open(os.path.join(output_dir, "usage_info.json"), 'w') as f:
    json.dump(usage_info, f, indent=2, default=str)

# Save usage summary
usage_summary = loader.get_usage_summary()
usage_summary.to_csv(os.path.join(output_dir, "usage_summary.csv"), index=False)

save_time = time.time() - save_start
print(f"✓ Processed data saved in {save_time:.2f} seconds")
print()

# Summary timing report
print("=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
total_time = time.time() - step1_start
print(f"Step 1 - Initialize loader:     {step1_time:8.2f} seconds")
print(f"Step 2 - Add genomes:           {step2_time:8.2f} seconds")
print(f"  - Human genome:               {human_time:8.2f} seconds")
print(f"  - Mouse genome:               {mouse_time:8.2f} seconds")
print(f"Step 3 - Add usage files:       {step3_time:8.2f} seconds")
print(f"  - Human usage files:          {human_usage_time:8.2f} seconds")
print(f"  - Mouse usage files:          {mouse_usage_time:8.2f} seconds")
print(f"Step 4 - Load genome data:      {step4_time:8.2f} seconds")
print(f"Step 5 - Convert to arrays:     {step5_time:8.2f} seconds")
print(f"Step 6 - Save processed data:   {save_time:8.2f} seconds")
print("-" * 60)
print(f"TOTAL TIME:                     {total_time:8.2f} seconds")
print("=" * 60)

print(f"\nProcessing completed at {datetime.now()}")
print(f"Results saved to: {os.path.abspath(output_dir)}")
print("\nSaved files:")
print(f"  - processed_data.npz: Windowed sequences, labels, and usage arrays")
print(f"  - metadata.csv.gz: Window metadata in compressed CSV format")
print(f"  - usage_info.json: Usage array structure and condition information")
print(f"  - usage_summary.csv: Summary statistics of usage data coverage")

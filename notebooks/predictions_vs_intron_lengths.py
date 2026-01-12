import pandas as pd
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt

subset = "middle"
species = "mouse_rat"
loss = "weighted_mse"
model = f"{subset}_{species}_{loss}"

# Load test metadata
data_dir = f"/home/elek/sds/sd17d003/Anamaria/splicevo/data/"
data_path = os.path.join(data_dir, f"splits_{subset}", f"{species}", "test")

from splicevo.utils.data_utils import load_processed_data
test_seq, test_labels, test_alpha, test_beta, test_sse, test_species = load_processed_data(data_path)

meta_fn = os.path.join(data_path, "metadata.json")
with open(meta_fn, "r") as f:
    import json
    test_meta = json.load(f)

# Load predictions
pred_dir = "/home/elek/sds/sd17d003/Anamaria/splicevo/predictions/"
pred_path = os.path.join(pred_dir, model)

from splicevo.utils.data_utils import load_predictions
pred_labels, pred_probs, pred_sse, meta, true_labels, true_sse = load_predictions(pred_path)

# Load metadata
meta_fn = os.path.join(pred_path, "metadata.json")
with open(meta_fn, "r") as f:
    import json
    meta = json.load(f)
    

# Load metadata.csv from test set directory
meta_fn = os.path.join(data_path, "metadata.csv")
meta_df = pd.read_csv(meta_fn)

# Check species in test data
species_idx = meta_df['species_id'].values.tolist()
species_id_to_name = {v: k for k, v in test_species.items()}
species_names = [species_id_to_name[idx] for idx in species_idx]
for sp in set(species_names):
    print(f"{species_names.count(sp)} {sp}")
    
# Load gtf files
from splicevo.io.gene_annotation import GTFProcessor
gtf_fns = {
    "human": "/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/Homo_sapiens.gtf.gz",
    "mouse": "/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/Mus_musculus.gtf.gz",
    "rat": "/home/elek/sds/sd17d003/Anamaria/genomes/mazin/gtf/Rattus_norvegicus.gtf.gz"
}
if "human" in species_names:
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', 'X', 'Y', 'MT']
    gtf_human = GTFProcessor(gtf_fns["human"]).load_gtf(chromosomes=chromosomes)
if "mouse" in species_names:
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'X', 'Y', 'MT']
    gtf_mouse = GTFProcessor(gtf_fns["mouse"]).load_gtf(chromosomes=chromosomes)
if "rat" in species_names:
    chromosomes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 'X', 'MT']
    gtf_rat = GTFProcessor(gtf_fns["rat"]).load_gtf(chromosomes=chromosomes)

# Get positions of splice sites from true_labels (sequence idx and offset in the sequence) into a dataframe
splice_sites = []
for seq_idx in range(true_labels.shape[0]):
    seq_meta = meta_df.iloc[seq_idx]
    chrom = seq_meta['chromosome']
    start_pos = seq_meta['window_start']
    species = species_id_to_name[seq_meta['species_id']]
    for pos in range(true_labels.shape[1]):
        label = true_labels[seq_idx, pos]
        if label == 1 or label == 2:
            genomic_pos = start_pos + pos
            splice_sites.append({
                'sequence_index': seq_idx,
                'sequence_offset': pos,
                'species': species,
                'chromosome': chrom,
                'genomic_position': genomic_pos,
                'label': label,
                'predicted_label': pred_labels[seq_idx, pos]
            })
splice_sites_df = pd.DataFrame(splice_sites)
splice_sites_df.head()

# Get overlap with exons from GTFs
def get_transcript_info_optimized(row, gtf_dicts):
    species = str(row['species'])
    chrom = str(row['chromosome'])
    pos = int(row['genomic_position'])
    
    if species not in gtf_dicts or gtf_dicts[species] is None:
        return pd.Series({
            'transcript_ids': [],
            'gene_ids': [],
            'transcript_lengths': [],
            'num_transcripts': 0,
            'num_genes': 0
        })
    
    gtf_idx = gtf_dicts[species]
    
    if chrom not in gtf_idx['chrom'].values:
        return pd.Series({
            'transcript_ids': [],
            'gene_ids': [],
            'transcript_lengths': [],
            'num_transcripts': 0,
            'num_genes': 0
        })
    
    # Fast lookup: pre-filtered by chromosome
    chrom_data = gtf_idx[gtf_idx['chrom'] == chrom]
    overlapping = chrom_data[(chrom_data['start'] <= pos) & (pos <= chrom_data['end'])]
    
    if len(overlapping) > 0:
        transcript_ids = overlapping['transcript_id'].tolist()
        gene_ids = overlapping['gene_id'].tolist()
        transcript_lengths = (overlapping['end'] - overlapping['start'] + 1).tolist()
        
        return pd.Series({
            'transcript_ids': transcript_ids,
            'gene_ids': gene_ids,
            'transcript_lengths': transcript_lengths,
            'num_transcripts': len(overlapping),
            'num_genes': len(set(gene_ids))
        })
    else:
        return pd.Series({
            'transcript_ids': [],
            'gene_ids': [],
            'transcript_lengths': [],
            'num_transcripts': 0,
            'num_genes': 0
        })

# Build GTF indexes once
gtf_dicts = {}
if 'gtf_human' in locals():
    gtf_dicts['human'] = gtf_human[gtf_human['feature'] == 'transcript'].copy()
if 'gtf_mouse' in locals():
    gtf_dicts['mouse'] = gtf_mouse[gtf_mouse['feature'] == 'transcript'].copy()
if 'gtf_rat' in locals():
    gtf_dicts['rat'] = gtf_rat[gtf_rat['feature'] == 'transcript'].copy()

splice_sites_df_head = splice_sites_df.head(5000)

# Use lambda to pass gtf_dicts to apply
splice_site_transcript_info = splice_sites_df_head.apply(
    lambda row: get_transcript_info_optimized(row, gtf_dicts), 
    axis=1
)
splice_sites_df_head = pd.concat([splice_sites_df_head, splice_site_transcript_info], axis=1)
splice_sites_df_head.head()

# Split each row into multiple rows if there are multiple transcripts
splice_sites_expanded = splice_sites_df_head.explode(['transcript_ids', 'gene_ids', 'transcript_lengths'])
splice_sites_expanded.head()

# Build exon index for fast lookup by transcript_id
def build_exon_index(gtf):
    """Pre-index GTF exons by (species, transcript_id) for O(1) lookup"""
    exons = gtf[gtf['feature'] == 'exon'].copy()
    exons = exons.sort_values(['transcript_id', 'start'])
    # Group by transcript_id and store as dict for fast access
    exon_index = {}
    for transcript_id, group in exons.groupby('transcript_id'):
        exon_index[transcript_id] = group.sort_values('start').reset_index(drop=True)
    return exon_index

# Build exon indexes once
exon_indexes = {}
if 'gtf_human' in locals():
    exon_indexes['human'] = build_exon_index(gtf_human)
if 'gtf_mouse' in locals():
    exon_indexes['mouse'] = build_exon_index(gtf_mouse)
if 'gtf_rat' in locals():
    exon_indexes['rat'] = build_exon_index(gtf_rat)

def get_intron_length_fast(row, exon_indexes):
    """Fast intron length lookup using pre-indexed exons"""
    species = str(row['species'])
    pos = int(row['genomic_position'])
    label = int(row['label'])
    transcript_id = str(row['transcript_ids'])
    
    if species not in exon_indexes:
        return np.nan
    
    if transcript_id not in exon_indexes[species]:
        return np.nan
    
    exons = exon_indexes[species][transcript_id]
    
    if label == 1:  # donor site
        following_exons = exons[exons['start'] >= pos]
        if not following_exons.empty:
            return following_exons.iloc[0]['start'] - pos
    elif label == 2:  # acceptor site
        preceding_exons = exons[exons['end'] <= pos]
        if not preceding_exons.empty:
            return pos - preceding_exons.iloc[-1]['end']
    
    return np.nan

splice_sites_expanded['intron_length'] = splice_sites_expanded.apply(
    lambda row: get_intron_length_fast(row, exon_indexes), 
    axis=1
)

#
# INTRONS
#
splice_sites_introns = splice_sites_expanded.copy()

# Group by intron length bins
intron_length_bins = [0, 100, 200, 500, 1000, 5000, 10000, 50000]
splice_sites_introns['length_bin'] = pd.cut(splice_sites_introns['intron_length'], bins=intron_length_bins)

# Select unique genomic positions and intron lengths pairs 
splice_sites_introns = splice_sites_introns.drop_duplicates(
    subset=['species', 'chromosome' ,'genomic_position', 'intron_length']
)

# Remove rows with NaN intron lengths
splice_sites_introns = splice_sites_introns.dropna(subset=['intron_length'])

# Remove introns shorter than 3 nt
splice_sites_introns = splice_sites_introns[splice_sites_introns['intron_length'] >= 3]

# Group by bin and count correct predictions
grouped = splice_sites_introns.groupby('length_bin', observed=True).apply(
    lambda df: pd.Series({
        'total_sites': len(df),
        'correct_predictions': np.sum(df['label'] == df['predicted_label'])
    }),
    include_groups=False
)
grouped['accuracy'] = grouped['correct_predictions'] / grouped['total_sites']
grouped.reset_index(inplace=True)

# Save
grouped.to_csv('/home/elek/sds/sd17d003/Anamaria/splicevo/model_accuracy_vs_intron_lengths.csv', index=False)

# Plot accuracy vs transcript length bins as lineplot, and number of sites as barplot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(
    grouped['length_bin'].astype(str),
    grouped['accuracy'],
    marker='o',
    color='tab:blue',
    label='Accuracy'
)
ax2.bar(
    grouped['length_bin'].astype(str),
    grouped['total_sites'],
    alpha=0.3,
    color='tab:orange',
    label='Number of Sites'
)
ax1.set_xlabel('Length Bins (bp)', fontsize=12)
ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
ax2.set_ylabel('Number of Splice Sites', color='tab:orange', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax1.set_ylim(0, 1)
#ax2.set_yscale('log')
plt.title('Model Accuracy vs Intron Length', fontsize=14)
fig = plt.gcf()
fig.tight_layout()

# Save figure
plt.savefig('/home/elek/sds/sd17d003/Anamaria/splicevo/model_accuracy_vs_transcript_length.png', dpi=300)

#
# TRANSCRIPTS
#
splice_sites_transcripts = splice_sites_expanded.copy()

# Group by transcript length bins
length_bins = [0, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]
splice_sites_transcripts['length_bin'] = pd.cut(splice_sites_transcripts['transcript_lengths'], bins=length_bins)

# Select unique genomic positions and intron lengths pairs 
splice_sites_transcripts = splice_sites_transcripts.drop_duplicates(
    subset=['species', 'chromosome' ,'genomic_position', 'transcript_lengths']
)

# Remove rows with NaN intron lengths
splice_sites_transcripts = splice_sites_transcripts.dropna(subset=['transcript_lengths'])

# Remove introns shorter than 3 nt
splice_sites_transcripts = splice_sites_transcripts[splice_sites_transcripts['transcript_lengths'] >= 3]

# Group by bin and count correct predictions
grouped = splice_sites_transcripts.groupby('length_bin', observed=True).apply(
    lambda df: pd.Series({
        'total_sites': len(df),
        'correct_predictions': np.sum(df['label'] == df['predicted_label'])
    }),
    include_groups=False
)
grouped['accuracy'] = grouped['correct_predictions'] / grouped['total_sites']
grouped.reset_index(inplace=True)

# Save
grouped.to_csv('/home/elek/sds/sd17d003/Anamaria/splicevo/model_accuracy_vs_transcript_lengths.csv', index=False)

# Plot accuracy vs transcript length bins as lineplot, and number of sites as barplot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(
    grouped['length_bin'].astype(str),
    grouped['accuracy'],
    marker='o',
    color='tab:blue',
    label='Accuracy'
)
ax2.bar(
    grouped['length_bin'].astype(str),
    grouped['total_sites'],
    alpha=0.3,
    color='tab:orange',
    label='Number of Sites'
)
ax1.set_xlabel('Length Bins (bp)', fontsize=12)
ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
ax2.set_ylabel('Number of Splice Sites', color='tab:orange', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax1.set_ylim(0, 1)
#ax2.set_yscale('log')
plt.title('Model Accuracy vs Transcript Length', fontsize=14)
fig = plt.gcf()
fig.tight_layout()

# Save figure
plt.savefig('/home/elek/sds/sd17d003/Anamaria/splicevo/model_accuracy_vs_transcript_length.png', dpi=300)
   

# Log off
print("Done.")
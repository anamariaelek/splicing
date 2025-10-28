import pandas as pd
from typing import Optional, Union, List


def read_gtf(
    gtf_path: str,
    features: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Read a local GTF annotation file into a DataFrame.

    Args:
        gtf_path: Path to the GTF file.
        features: Optional; specific features to keep (e.g. "exon", "CDS").

    Returns:
        A pandas DataFrame with standard GTF columns.
    """
    from grelu.utils import make_list  # keep same behavior as grelu version

    # Load GTF, skip comment lines
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "chrom", "source", "feature", "start", "end", "score",
            "strand", "frame", "attribute"
        ],
        dtype={
            "chrom": str,
            "source": str,
            "feature": str,
            "start": int,
            "end": int,
            "score": str,
            "strand": str,
            "frame": str,
            "attribute": str
        }
    )

    # Reorder so "chrom", "start", "end" are the first three columns
    cols = gtf.columns.tolist()
    cols.insert(0, cols.pop(cols.index("chrom")))
    cols.insert(1, cols.pop(cols.index("start")))
    cols.insert(2, cols.pop(cols.index("end")))
    gtf = gtf.loc[:, cols]

    # Filter features if specified
    if features is not None:
        gtf = gtf[gtf.feature.isin(make_list(features))]

    return gtf

def filter_gtf(
    gtf: pd.DataFrame,
    column_name: str,
    column_values: Union[str, List[str]]
) -> pd.DataFrame:
  """
  Filter GTF entries by feature.

  This function takes a GTF DataFrame, column name to filter by, and a list
  of values to filter for (include). It returns a new DataFrame containing 
  only the entries that one of `column_values` in `column_name`.
  The function will raise a ValueError if specified `column_name` is not
  present in the GTF.

  Args:
    gtf: pd.DataFrame or pyranges.PyRanges.
    column_name: Name of the column to filter by.
    column_values: List of valid transcript types to use for filtering.

  Returns:
    pd.DataFrame of GTF entries subset to rows with the requested values 
    in specified column.
  """
  if column_name is not None:
    column_values_str = [x.value for x in column_values]
    if column_name in gtf.columns:
      gtf = gtf[gtf[column_name].isin(column_values_str)]
    else:
      raise ValueError('specified column_name is not in gtf.')
  return gtf

def extract_introns(gtf_df):
    """
    Extract introns from a GTF annotations DataFrame.
    
    Parameters:
    -----------
    gtf_df : pandas.DataFrame
        DataFrame with GTF annotations (as produced by GTFProcessor.load_gtf)
        Expected columns include:
        - chrom
        - source
        - feature (gene, transcript, exon, etc.)
        - start
        - end
        - strand
        - gene_id
        - transcript_id
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with intron annotations in GTF format
    """
    # Filter for exons only
    exons_df = gtf_df[gtf_df['feature'] == 'exon'].copy()
    
    if exons_df.empty:
        return pd.DataFrame()
    
    # Sort by transcript_id, chromosome, and start position
    exons_df = exons_df.sort_values(['transcript_id', 'chrom', 'start'])

    introns = []
    
    # Group by transcript to find introns between exons
    for transcript_id, group in exons_df.groupby('transcript_id'):
        group = group.sort_values('start')
        
        # Skip transcripts with only one exon (no introns)
        if len(group) < 2:
            continue
        
        # Get common attributes for this transcript
        chrom = group.iloc[0]['chrom']
        strand = group.iloc[0]['strand']
        source = group.iloc[0].get('source', '.')
        score = group.iloc[0].get('score', '.')
        frame = '.'
        gene_id = group.iloc[0].get('gene_id', '')
        
        # Find introns between consecutive exons
        for i in range(len(group) - 1):
            intron_start = group.iloc[i]['end'] + 1
            intron_end = group.iloc[i + 1]['start'] - 1
            
            # Only add valid introns (where start <= end)
            if intron_start <= intron_end:
                intron = {
                    'chrom': chrom,
                    'source': source,
                    'feature': 'intron',
                    'start': intron_start,
                    'end': intron_end,
                    'score': score,
                    'strand': strand,
                    'frame': frame,
                    'gene_id': gene_id,
                    'transcript_id': transcript_id
                }
                
                # Copy any additional attributes from the exon
                for col in group.columns:
                    if col not in intron:
                        intron[col] = group.iloc[i][col]
                
                introns.append(intron)
    
    # Create DataFrame from introns
    if introns:
        introns_df = pd.DataFrame(introns)
        return introns_df
    else:
        return pd.DataFrame()
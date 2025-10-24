# SpliSER

I use SpliSER tool to estimate splice site usage in different samples.

I use [version 1.0.4](https://github.com/CraigIDent/SpliSER/tree/speedups) that comes with performance improvements and updates including pre-combine command.

I process each group of samples separately. Groupping of samples can be found in `data/spliser/samples.txt`.
For example, `Mouse.Brain.13.5` groups all samples from mouse brain at 13.5 days post fertilization (dpf):

```
2813sTS.Mouse.Brain.13.5.Female.sorted.bam
2821sTS.Mouse.Brain.13.5.Male.sorted.bam
2829sTS.Mouse.Brain.13.5.Female.sorted.bam
2837sTS.Mouse.Brain.13.5.Male.sorted.bam
```

Firstly, I combine introns from individual replicates bam files using `preCombineIntrons` command. 

```
spliser preCombineIntrons \
    -L "$bams" \
    -o "$out_dir"/"$group" \
    --isStranded -s rf \
    -A "$gtf"
```

Importantly, `--isStranded -s rf` defines that the data is strand-specific and the orientation is first-strand (TruSeq Stranded mRNA LT Sample Prep Kit). The annotation file needs to be a gff/gtf file with 'exon' features that have either a 'Parent' or 'transcript_id' attribute indicating which transcript they belong to. This command takes ~10 mins and generates an introns file `Mouse.Brain.0dpb.introns.tsv` in the results directory `results/spliser/Mus_musculus/`. Introns file contains all introns found in any of the samples (union, not intersect).  

Introns file is then used together with each sample bam file to calculate the splice site strength estimate (SSE) for all splice sites in that sample using .

```
spliser process \
    -B "$bam" \
    -I "$out_dir"/"$group".introns.tsv \
    --isStranded -s rf \
    -A "$gtf" -t transcript \
    -o "$out_dir"/"$name"
```

This generates SpliSER output files: 

```
2813sTS.Mouse.Brain.13.5.Female.SpliSER.tsv
2821sTS.Mouse.Brain.13.5.Male.SpliSER.tsv
2829sTS.Mouse.Brain.13.5.Female.SpliSER.tsv
2837sTS.Mouse.Brain.13.5.Male.SpliSER.tsv
```

Then I run SpliSER on each individual bam file using the `process` command and generate `SpliSER.tsv` files for each sample. This takes a lot of time and better be parallelized.

Finally, I create a `samples.tsv` file that lists all the `SpliSER.tsv` files to be combined, and then I use the `combine` command to generate a combined output file `combined.tsv`.


```
    { sample1.group.bam         sample2.group.bam       sample3.group.bam  }    preCombineIntrons 
            │                           │                       │                       │
            ▼                           ▼                       ▼                       ▼
          process                    process                 process     ◄─     [group.introns.tsv]
            │                           │                       │
            ▼                           ▼                       ▼
    sample1.group.SpliSER.tsv   sample2.group.SpliSER.tsv  sample3.group.SpliSER.tsv
             └──────────────────────────┼────────────────────────┘
                              group.SamplesFile.tsv
                                        ▼
                                     combine
                                        ▼
                                 group.combined.tsv
```


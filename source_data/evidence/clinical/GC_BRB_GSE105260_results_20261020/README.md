# GSE105260 NR0B2 methylation

This analysis supports Figure 7. The retained table contains eight CpGs for 44 samples: 9 normal kidney, 9 primary ccRCC and 26 metastatic ccRCC. The primary figure compares the 9 primary tumors with the 9 normal samples.

## Run from the retained values

Use Python 3.12. From this folder:

```sh
python -m pip install -r requirements.txt
python recompute_gse105260.py
```

The script reads `GSE105260_selected_CpG_sample_values.csv` beside the script and writes new results to `recomputed/`. It checks sample identifiers, tissue/group labels, beta values and sample means before analysis. Existing summary tables are preserved.

## Run from the original series matrix

The complete series matrix is available from GEO and is not duplicated in this package. To repeat extraction and analysis, provide the local matrix path:

```sh
python recompute_gse105260.py --matrix GSE105260_series_matrix.txt.gz --output recomputed_full
```

`--selected-values` selects another retained table. `--matrix` and `--selected-values` are mutually exclusive. Input mode and input SHA-256 are recorded in `recomputation_summary.json`.

## Analysis

The per-CpG comparison uses group medians, a two-sided Mann-Whitney U test with asymptotic p values and continuity correction, and BH adjustment across eight CpGs. Eight-CpG sample means are calculated before group summaries. Metastasis comparisons in the sample-mean output are secondary and use nominal p values.

Both input modes were tested against the repository result tables.

Probe-region labels follow SCRIPT_REGIONS in the script. Their platform and transcript assignments await independent verification.

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE105260

Platform: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13534

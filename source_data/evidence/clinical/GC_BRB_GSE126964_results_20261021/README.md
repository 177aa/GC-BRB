# GSE126964 paired expression

This analysis supports Figure 5a. It uses six genes in 11 paired ccRCC tumor and normal samples from GSE126964.

## Run

Use Python 3.12. From this folder:

```sh
python -m pip install -r requirements.txt
python run_GSE126964_FROZEN6_PAIRED_REPRODUCIBLE.py
```

The script locates inputs relative to its own file. It writes new results to `recomputed/`, leaving retained results in `results/` unchanged. `--input-dir` and `--output-dir` can override these locations.

## Files

- `GSE126964_Expr_RSEM.txt.gz`: retained original expression matrix, containing 20,531 gene records and 66 samples.
- `results/GSE126964_PAIR_VALUES_independent.csv`: retained values for the 66 gene-pair records.
- `results/GSE126964_pair_mapping_11_GEO.csv`: retained mapping of the 11 tumor/normal pairs to GEO metadata.
- `results/GSE126964_PAIRED_SUMMARY_independent.csv`: the summary already present in this repository; it is used as a reference check.

Default reference checks use the two retained result tables. The script reports `INCOMPLETE_REFERENCE_CHECK` if either reference table is absent, and exits with an error if a comparison fails.

## Analysis

Gene symbols are matched to the text preceding `|` in the input Gene field. Tumor and normal samples are paired by the shared CKC identifier, with terminal `N` identifying the normal sample. The retained GEO mapping supports the 11 pairs.

The effect is tumor-minus-normal `log2(FPKM + 1)`. Each gene is tested with a two-sided exact Wilcoxon signed-rank test, followed by BH adjustment across six genes. 

This script reimplements the paired-expression analysis for Fig. 5a.

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126964

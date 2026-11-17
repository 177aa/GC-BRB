# GSE68465 six-gene survival analysis

This analysis supports Figure 5b. The retained complete-case expression table contains 442 LUAD patients and 236 deaths.

## Run from the retained expression table

Use Python 3.12. From this folder:

```sh
python -m pip install -r requirements.txt
python run_GSE68465_FROZEN6_COX.py
```

The script reads `GSE68465_FROZEN6_EXPRESSION.csv` and `GSE68465_probe_mapping.csv` beside the script. The expression table is already present in this repository. The probe mapping was extracted from the retained GPL96 annotation, not inferred from outcome estimates.

New results are written to `recomputed/`. Existing result tables are preserved. Input and output directories can be set with `--input-dir` and `--output-dir`.

## Run from the original GEO inputs

Obtain `GSE68465_series_matrix.txt.gz` and `GPL96_annotation.txt` from the original sources and place them in a local `raw/` folder. These large inputs are not duplicated in this package.

```sh
python run_GSE68465_FROZEN6_COX.py --from-raw --input-dir raw --output-dir recomputed_full
```

This mode repeats probe mapping, expression transformation and complete-case selection. It writes the selected probe mapping as well as the expression and Cox tables. Input mode, input hashes and software versions are recorded in `recomputation_summary.json`.

## Analysis

For each gene, all mapped probes are transformed with `log2(MAS5 + 1)` and averaged before survival analysis. The LUAD complete-case cohort excludes missing event information, missing follow-up and nonpositive survival times. Each gene is standardized using its sample standard deviation, then fitted in a separate univariable Cox model with Efron ties. The six p values receive BH adjustment.

These statistical calculations are retained from the previously supplied script. This revision replaces the fixed local drive path with command-line and relative defaults, adds a retained-expression input mode and separates new outputs from archived results. Both input modes were run and matched the retained tables within 1e-12.

Sources:

- GSE68465: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68465
- GPL96: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96

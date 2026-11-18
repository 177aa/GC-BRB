# GC-BRB and EMGNN comparison

This folder contains the MATLAB scripts and saved test predictions used for
the paired comparison between GC-BRB and EMGNN across six networks.

The GC-BRB predictions are from the earlier three-layer analysis. They are
separate from the gene identity group results in the main model folder.

## Run

Open this folder in MATLAB and run:

    main_compare_emgnn

The script reads the predictions in `results/final/official_exact` and
`outputs/table1`. The statistical tables are saved in
`results/paper_statistics_emgnn`.

# SEFGNN comparison

Training and evaluation scripts for the SEFGNN comparison in Fig. 2c.

## Environment

`setup_sefgnn_cloud_env_v3.sh` describes a Linux environment with CUDA 12.1, Python 3.10, PyTorch 2.2.2 and PyTorch Geometric 2.6.1.

The files in `SEFGNN_official/` include translated comments and output messages.

## Input

Extract `multilayer_aligned_gcbrb.zip` from `model/model_input/cache/prepared/`. Copy the extracted `.mat` file to `GC_BRB_3LAYER_FINAL/cache/prepared/` in this folder.

The saved split manifests and six network caches are under `GC_BRB_3LAYER_FINAL/results/sefgnn_same_benchmark/`.

## Run

From `model/sefgnn/`, run `python run_sefgnn_formal_cloud_v2.py --targets CPDB --runs 1` for one CPDB training run. The default budget is 300 epochs.

New results are written to `GC_BRB_3LAYER_FINAL/results/sefgnn_same_benchmark/08_formal_deterministic_v2/`. The saved results used for the comparison are in `source_data/comparisons/sefgnn/`.

A complete rerun with this repository layout has not yet been verified.

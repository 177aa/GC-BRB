# GC-BRB

GC-BRB integrates multi-omics features and protein-protein interaction networks to prioritize candidate cancer genes. This folder contains the MATLAB code for training and evaluation with gene identity groups.

## Requirements

- MATLAB R2024a
- Statistics and Machine Learning Toolbox
- MATLAB Java virtual machine (JVM)

The Python scripts for baseline models use the packages listed in `python/requirements.txt`.

## Files

| File or folder | Description |
| --- | --- |
| `gg_config.m` | Paths and model settings |
| `gg_setup.m`, `gg_run.m` | Setup and workflow entry points |
| `STEP*.m` | Scripts for checks and model runs |
| `core/` | Data splitting, evidence construction, training and output functions |
| `legacy_core/` | Numerical and model functions |
| `tests/` | Test scripts |
| `reference/` | Reference files for identity and split checks |
| `python/` | Baseline and run-verification scripts |
| `MAKE_RETURN_PACK.m` | Output packaging helper |

## Input data

The model uses `multilayer_aligned_gcbrb.mat`, prepared from the EMOGI benchmark. The file contains 64-dimensional multi-omics features and six network layers.

Before running the model, extract `multilayer_aligned_gcbrb.zip` in the `model_input/cache/prepared/` folder. The extracted file should be located at:

```text
model_input/cache/prepared/multilayer_aligned_gcbrb.mat
```

Keep the `reference/` folder with the model code. It is used to check the identity and order of input records.

The prepared data file's SHA-256 is:

```text
8ba26edb29f8f70b3fcbdf7b07d0c3a05096f38b64df0ad1768232e89f5c72d3
```

## Running the model

Set the MATLAB current folder to the directory containing `gg_config.m` and the `STEP*.m` scripts. Run the stages in order:

```matlab
STEP1_CHECK_ONLY      % Check the 30 planned splits
STEP2_SMOKE_CPDB      % Run self-tests and a short CPDB test
STEP3_PILOT_CPDB      % Run one CPDB model with formal settings
STEP4_FORMAL_30RUNS   % Run six target networks, five times each
```

The smoke test uses reduced settings. The pilot and formal stages use the same `formal` output folder and resume saved runs when their fingerprints match.

## Output

Results are written to `results/GG_OOF_1_0/` under `preflight`, `smoke` or `formal`. Per-run files are stored by target network and run, for example `formal/CPDB/run_01/`. The workflow also saves `protocol.json`, split summaries, predictions and collected performance tables.

Existing outputs are checked against the input, code and configuration fingerprints. Use a separate output directory when these change.

## Data and code availability

The prepared input archive is provided in `model_input/cache/prepared/`. Archived run results are provided in `formal_results/formal/`. The model workflow has not yet been tested in a clean environment with this folder layout.

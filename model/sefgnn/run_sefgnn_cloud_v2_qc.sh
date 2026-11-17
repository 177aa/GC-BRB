#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
export SEFGNN_WORKDIR="$WORKDIR"
export GCBRB_ROOT="$WORKDIR/GC_BRB_3LAYER_FINAL"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sefgnn_qc

mkdir -p "$WORKDIR/logs"

echo "======================================================================"
echo "STEP A: adapter equivalence audit (no GC-BRB Test evaluation)"
echo "======================================================================"
python -u "$WORKDIR/audit_sefgnn_adapter_v2_equivalence.py" \
  2>&1 | tee "$WORKDIR/logs/adapter_v2_equivalence.log"

grep -q "SEFGNN_ADAPTER_V2_EQUIVALENCE_PASS" \
  "$WORKDIR/logs/adapter_v2_equivalence.log" || {
    echo "STOP: equivalence audit did not PASS."
    exit 3
  }

echo
echo "======================================================================"
echo "STEP B: CPDB run_01 only — formal QC gate"
echo "======================================================================"
python -u "$WORKDIR/run_sefgnn_formal_cloud_v2.py" \
  --targets CPDB --runs 1 --epochs 300 --gpu 0 \
  2>&1 | tee "$WORKDIR/logs/cpdb_run01_v2.log"

echo
echo "======================================================================"
echo "QC RUN COMPLETE"
echo "Upload/send back:"
echo "  logs/adapter_v2_equivalence.log"
echo "  logs/cpdb_run01_v2.log"
echo "  GC_BRB_3LAYER_FINAL/results/sefgnn_same_benchmark/"
echo "    08_formal_deterministic_v2/CPDB/run_01/FINAL_FORMAL_STATUS.json"
echo "======================================================================"

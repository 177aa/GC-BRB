#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "SEFGNN CLOUD QC ENVIRONMENT"
echo "Fresh conda env: Python 3.10"
echo "PyTorch 2.2.2 + cu121 | PyG 2.6.1"
echo "======================================================================"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found."
  echo "Choose a cloud image that includes Miniconda/Anaconda, then rerun."
  exit 2
fi

echo "[0/7] GPU/driver"
nvidia-smi

echo "[1/7] Creating isolated env sefgnn_qc (Python 3.10)"
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "sefgnn_qc"; then
  echo "Environment sefgnn_qc already exists; reusing it."
else
  conda create -y -n sefgnn_qc python=3.10
fi
conda activate sefgnn_qc

python -V
python -m pip install -U pip setuptools wheel

echo "[2/7] PyTorch 2.2.2 + CUDA 12.1 runtime"
python -m pip install --no-cache-dir \
  torch==2.2.2+cu121 \
  torchvision==0.17.2+cu121 \
  torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[3/7] PyG + scientific dependencies"
python -m pip install --no-cache-dir \
  torch-geometric==2.6.1 \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scipy==1.13.1 \
  scikit-learn==1.5.1 \
    h5py==3.11.0 \
  matplotlib

echo "[4/7] PyG compiled operators for induced NeighborLoader"
python -m pip install --no-cache-dir \
  pyg_lib torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

echo "[5/7] Version audit"
python - <<'PY'
import sys, torch, torch_geometric
import torch_geometric.typing as T
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("PyG:", torch_geometric.__version__)
print("WITH_PYG_LIB:", T.WITH_PYG_LIB)
print("WITH_TORCH_SPARSE:", T.WITH_TORCH_SPARSE)
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.startswith("2.2.2")
assert torch_geometric.__version__.startswith("2.6.1")
assert torch.cuda.is_available()
assert T.WITH_TORCH_SPARSE
PY

echo "[6/7] Exact induced NeighborLoader preflight"
python - <<'PY'
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
x=torch.randn(6,3)
e=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,0],
                [1,0,2,1,3,2,4,3,5,4,0,5]], dtype=torch.long)
d=Data(x=x, edge_index=e)
loader=NeighborLoader(
    d,
    num_neighbors=[2,2],
    batch_size=2,
    input_nodes=torch.tensor([0,1]),
    shuffle=False,
    num_workers=0,
    subgraph_type="induced",
)
b=next(iter(loader))
print("INDUCED NeighborLoader PASS | batch_size=", b.batch_size,
      "| sampled_nodes=", b.num_nodes)
PY

echo "[7/7] Environment location"
echo "CONDA_PREFIX=$CONDA_PREFIX"

echo "======================================================================"
echo "STATUS_CODE: SEFGNN_CLOUD_ENV_PASS"
echo "For every new terminal run:"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh"
echo "  conda activate sefgnn_qc"
echo "======================================================================"

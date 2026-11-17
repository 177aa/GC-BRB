# -*- coding: utf-8 -*-
r"""
SEFGNN adaptation for the GC-BRB benchmark.

The adapter uses six network views and the saved training, validation and test splits. Missing views are represented by a Dirichlet opinion with alpha=[1, 1]. The default run is a three-epoch CPDB smoke test.

Keep this script and the SEFGNN_official folder in the same directory. The SEFGNN_WORKDIR and GCBRB_ROOT environment variables can be used to set the input locations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

# Cloud/Linux path resolution.
# Recommended layout:
#   <WORKDIR>/
#       run_sefgnn_gcbrb_adapter_cloud.py
#       run_sefgnn_formal_cloud.py
#       SEFGNN_official/
#       GC_BRB_3LAYER_FINAL/
#
# You may override either location with environment variables:
#   export SEFGNN_WORKDIR=/workspace/sefgnn
#   export GCBRB_ROOT=/workspace/sefgnn/GC_BRB_3LAYER_FINAL
SCRIPT_ROOT = Path(
    os.environ.get("SEFGNN_WORKDIR", Path(__file__).resolve().parent)
).resolve()
PROJECT_ROOT = Path(
    os.environ.get("GCBRB_ROOT", SCRIPT_ROOT / "GC_BRB_3LAYER_FINAL")
).resolve()
OFFICIAL_REPO = SCRIPT_ROOT / "SEFGNN_official"

PREPARED_MAT = PROJECT_ROOT / "cache" / "prepared" / "multilayer_aligned_gcbrb.mat"
MANIFEST_ROOT = (
    PROJECT_ROOT / "results" / "sefgnn_same_benchmark" /
    "02_exact_30run_manifests" / "manifests"
)
BACKBONE_ROOT = PROJECT_ROOT / "results" / "backbone" / "official_exact"

CACHE_ROOT = (
    PROJECT_ROOT / "results" / "sefgnn_same_benchmark" / "03_adapter_cache"
)
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "sefgnn_same_benchmark" / "04_cpdb_smoke"
)

VIEWS = ["CPDB", "MULTINET", "PCNET", "STRINGdb", "IREF", "IREF_2015"]
H5_PATHS = {
    "CPDB": PROJECT_ROOT / "data" / "CPDB_multiomics.h5",
    "MULTINET": PROJECT_ROOT / "data" / "MULTINET_multiomics.h5",
    "PCNET": PROJECT_ROOT / "data" / "PCNET_multiomics.h5",
    "STRINGdb": PROJECT_ROOT / "data" / "STRINGdb_multiomics.h5",
    "IREF": PROJECT_ROOT / "data" / "IREF_multiomics.h5",
    "IREF_2015": PROJECT_ROOT / "data" / "IREF_2015_multiomics.h5",
}

EXPECTED_LOCAL_N = {
    "CPDB": 13627,
    "MULTINET": 14398,
    "PCNET": 19781,
    "STRINGdb": 13179,
    "IREF": 17013,
    "IREF_2015": 12129,
}

EXPECTED_SPLITS = {
    "CPDB":      dict(train=7104, val=790, test=746,  test_pos=199),
    "MULTINET":  dict(train=6763, val=752, test=1125, test_pos=198),
    "PCNET":     dict(train=6349, val=705, test=1586, test_pos=215),
    "STRINGdb":  dict(train=7056, val=784, test=800,  test_pos=196),
    "IREF":      dict(train=6675, val=742, test=1223, test_pos=209),
    "IREF_2015": dict(train=7155, val=795, test=690,  test_pos=196),
}

N_GLOBAL = 22659
N_LABELLED = 8640
N_POS = 887
N_NEG = 7753

# Official BIBM code/config behavior that is intentionally preserved.
OFFICIAL_NUM_NEIGHBORS = [40, 10]
OFFICIAL_BATCH_SIZE = 128
OFFICIAL_WEIGHT_DECAY = 0.005
OFFICIAL_DROP_NEG = 0.30
OFFICIAL_RESIDUAL = 1.0
OFFICIAL_HIDDEN_CHANNELS = 64  # hard-coded by official main.py for SEFMGNN
OFFICIAL_SES_DS_WEIGHT = 0.8
OFFICIAL_NUM_EPOCHS = 300
OFFICIAL_EARLY_PATIENCE = 50
OFFICIAL_EARLY_AFTER_FRACTION = 0.5
OFFICIAL_WARMUP_FRACTION = 0.2

# Graph scan block size: ~20k x 256 float64 ~ 40 MB.
NETWORK_BLOCK_COLS = 256


# =============================================================================
# OFFICIAL MODEL IMPORT
# =============================================================================

if not OFFICIAL_REPO.exists():
    raise FileNotFoundError(
        f"Official SEFGNN repository not found:\n{OFFICIAL_REPO}\n"
        "Run prepare_sefgnn_official_adaptation.py first."
    )

sys.path.insert(0, str(OFFICIAL_REPO))

# Official source: model.py
from model import SEFMGNN  # noqa: E402


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_linear_warmup_scheduler(optimizer, num_training_steps: int, warmup_fraction=0.2):
    """
    Behavior-equivalent to transformers.get_linear_schedule_with_warmup used by
    official main.py, but avoids an unnecessary Transformers dependency.
    """
    num_warmup_steps = int(round(warmup_fraction * num_training_steps))

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# MATLAB v7.3 READER
# =============================================================================

def _is_ref_dtype(ds):
    return h5py.check_dtype(ref=ds.dtype) is not None or ds.dtype.kind == "O"


def _resolve_dataset(h, ds):
    data = ds[()]
    if not _is_ref_dtype(ds):
        return np.asarray(data).squeeze()

    refs = np.asarray(data).reshape(-1)
    vals = []
    for ref in refs:
        if not ref:
            continue
        obj = h[ref]
        if isinstance(obj, h5py.Dataset):
            vals.append(_resolve_dataset(h, obj))
        elif isinstance(obj, h5py.Group):
            vals.append(obj)

    if len(vals) == 1:
        return vals[0]
    return vals


def _resolve_to_group(h, obj):
    if isinstance(obj, h5py.Group):
        return obj
    if isinstance(obj, h5py.Dataset) and _is_ref_dtype(obj):
        refs = [r for r in np.asarray(obj[()]).reshape(-1) if r]
        if len(refs) != 1:
            raise RuntimeError(
                f"Expected one MATLAB struct reference at {obj.name}; found {len(refs)}"
            )
        return _resolve_to_group(h, h[refs[0]])
    raise RuntimeError(f"Could not resolve {getattr(obj, 'name', '?')} to group")


def _get_struct_group(h, name: str):
    key = name if name in h else "/" + name.lstrip("/")
    if key not in h:
        raise KeyError(f"{name} not found")
    return _resolve_to_group(h, h[key])


def _get_group_field_numeric(h, group, field: str):
    key = field if field in group else next(
        (k for k in group.keys() if k.lower() == field.lower()), None
    )
    if key is None:
        raise KeyError(f"{field} not found in {group.name}; keys={list(group.keys())[:30]}")
    obj = group[key]
    if not isinstance(obj, h5py.Dataset):
        raise RuntimeError(f"{group.name}/{key} is not a dataset")
    val = _resolve_dataset(h, obj)
    if isinstance(val, list) or isinstance(val, h5py.Group):
        raise RuntimeError(f"{group.name}/{key} did not resolve to numeric array")
    return np.asarray(val).squeeze()


def load_prepared_alignment():
    """
    Returns exact GC-BRB:
      X:        [22659, 64]
      y:        [22659]
      presence: [22659, 6]
      local_to_global[view]: MATLAB 1-based local->global vector
    """
    if not PREPARED_MAT.exists():
        raise FileNotFoundError(PREPARED_MAT)

    with h5py.File(PREPARED_MAT, "r") as h:
        data = _get_struct_group(h, "data")

        X = _get_group_field_numeric(h, data, "X")
        y = _get_group_field_numeric(h, data, "y")
        presence = _get_group_field_numeric(h, data, "presence")

        # MATLAB 22659x64 appears as 64x22659 in HDF5 on many installations.
        X = np.asarray(X, dtype=np.float64)
        if X.shape == (64, N_GLOBAL):
            X = X.T
        elif X.shape != (N_GLOBAL, 64):
            raise RuntimeError(f"Unexpected data.X shape: {X.shape}")

        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != N_GLOBAL:
            raise RuntimeError(f"Unexpected data.y length: {y.size}")

        presence = np.asarray(presence)
        if presence.shape == (len(VIEWS), N_GLOBAL):
            presence = presence.T
        elif presence.shape != (N_GLOBAL, len(VIEWS)):
            raise RuntimeError(f"Unexpected data.presence shape: {presence.shape}")
        presence = presence.astype(bool)

        # Read data.layers cell array -> each layer struct -> local_to_global.
        if "layers" not in data:
            raise KeyError("data.layers missing from prepared MAT")
        layers_obj = data["layers"]
        if not isinstance(layers_obj, h5py.Dataset) or not _is_ref_dtype(layers_obj):
            raise RuntimeError("data.layers is not a MATLAB cell/reference dataset")

        refs = [r for r in np.asarray(layers_obj[()]).reshape(-1) if r]
        if len(refs) != len(VIEWS):
            raise RuntimeError(f"Expected 6 layer references, found {len(refs)}")

        local_to_global = {}
        for view, ref in zip(VIEWS, refs):
            layer_group = _resolve_to_group(h, h[ref])
            ltg = _get_group_field_numeric(h, layer_group, "local_to_global")
            ltg = np.asarray(ltg).reshape(-1)
            if not np.allclose(ltg, np.round(ltg)):
                raise RuntimeError(f"{view}: non-integer local_to_global")
            ltg = np.round(ltg).astype(np.int64)
            if len(ltg) != EXPECTED_LOCAL_N[view]:
                raise RuntimeError(
                    f"{view}: local_to_global length {len(ltg)} "
                    f"!= expected {EXPECTED_LOCAL_N[view]}"
                )
            local_to_global[view] = ltg

    # Hard benchmark audit.
    if X.shape != (N_GLOBAL, 64):
        raise RuntimeError("Global feature shape audit failed")
    if int(np.isfinite(y).sum()) != N_LABELLED:
        raise RuntimeError(
            f"Expected {N_LABELLED} labelled genes, got {int(np.isfinite(y).sum())}"
        )
    if int((y == 1).sum()) != N_POS or int((y == 0).sum()) != N_NEG:
        raise RuntimeError(
            f"Global label audit failed: +{int((y==1).sum())}/-{int((y==0).sum())}"
        )

    # Presence must agree exactly with local_to_global support.
    for j, view in enumerate(VIEWS):
        implied = np.zeros(N_GLOBAL, dtype=bool)
        implied[np.unique(local_to_global[view] - 1)] = True
        if not np.array_equal(implied, presence[:, j]):
            diff = int(np.sum(implied != presence[:, j]))
            raise RuntimeError(f"{view}: presence/local_to_global mismatch at {diff} genes")

    return X.astype(np.float32), y, presence, local_to_global


# =============================================================================
# GRAPH CACHE
# =============================================================================

def graph_cache_path(view: str):
    return CACHE_ROOT / "global_edge_index" / f"{view}_edge_index.npz"


def build_or_load_edge_index(view: str, local_to_global_1based: np.ndarray):
    """
    Build the exact undirected binary global graph used by GC-BRB:
      local H5 adjacency -> symmetrize -> remove self loops -> map to global ids.

    We encode each undirected pair (min,max) as min*N + max and unique it.
    """
    cache = graph_cache_path(view)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        z = np.load(cache)
        edge_index = z["edge_index"].astype(np.int64, copy=False)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise RuntimeError(f"{view}: invalid cached edge_index shape {edge_index.shape}")
        if np.any(edge_index < 0) or np.any(edge_index >= N_GLOBAL):
            raise RuntimeError(f"{view}: cached edge_index contains out-of-range node ids")
        if "n_global" in z and int(np.asarray(z["n_global"]).reshape(-1)[0]) != N_GLOBAL:
            raise RuntimeError(f"{view}: cached n_global mismatch")

        present_global = np.zeros(N_GLOBAL, dtype=bool)
        present_global[np.asarray(local_to_global_1based, dtype=np.int64) - 1] = True
        if not np.all(present_global[edge_index]):
            raise RuntimeError(
                f"{view}: cached graph has an edge incident to a gene absent from that view"
            )

        print(
            f"  {view:10s} edge cache: {edge_index.shape[1]:,} directed edges "
            f"({edge_index.shape[1]//2:,} undirected)"
        )
        return edge_index

    h5_path = H5_PATHS[view]
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    ltg = np.asarray(local_to_global_1based, dtype=np.int64) - 1
    n_local = len(ltg)
    encoded_chunks = []

    print(f"  {view:10s} scanning H5 network ({n_local:,} nodes)...")
    t0 = time.time()

    with h5py.File(h5_path, "r") as h:
        ds = h["/network"]
        if tuple(ds.shape) != (n_local, n_local):
            raise RuntimeError(
                f"{view}: /network shape {tuple(ds.shape)} != {(n_local,n_local)}"
            )

        for c0 in range(0, n_local, NETWORK_BLOCK_COLS):
            c1 = min(n_local, c0 + NETWORK_BLOCK_COLS)
            block = np.asarray(ds[:, c0:c1])
            rr, cc = np.nonzero(block)
            if rr.size:
                src = ltg[rr]
                dst = ltg[c0 + cc]
                lo = np.minimum(src, dst)
                hi = np.maximum(src, dst)
                keep = lo != hi
                if np.any(keep):
                    enc = lo[keep] * np.int64(N_GLOBAL) + hi[keep]
                    encoded_chunks.append(enc.astype(np.int64, copy=False))

            if (c0 // NETWORK_BLOCK_COLS) % 20 == 0 or c1 == n_local:
                print(f"    columns {c0+1:>6}-{c1:<6} / {n_local}")

    if not encoded_chunks:
        raise RuntimeError(f"{view}: no graph edges found")

    encoded = np.unique(np.concatenate(encoded_chunks))
    lo = encoded // np.int64(N_GLOBAL)
    hi = encoded % np.int64(N_GLOBAL)

    edge_index = np.vstack([
        np.concatenate([lo, hi]),
        np.concatenate([hi, lo]),
    ]).astype(np.int64, copy=False)

    np.savez_compressed(
        cache,
        edge_index=edge_index,
        n_global=np.array([N_GLOBAL], dtype=np.int64),
    )

    print(
        f"  {view:10s} built: {len(encoded):,} undirected / "
        f"{edge_index.shape[1]:,} directed edges in {(time.time()-t0):.1f}s"
    )
    return edge_index


# =============================================================================
# MANIFEST / LABELS
# =============================================================================

def read_manifest(target: str, run: int, split_name: str):
    path = (
        MANIFEST_ROOT / target / f"run_{run:02d}" /
        f"{split_name.lower()}_identity.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Exact split manifest missing:\n{path}\n"
            "Run freeze_gcbrb_exact_30run_manifests_v2.py first."
        )
    df = pd.read_csv(path)

    idx_col = (
        "GlobalIndex_1based"
        if "GlobalIndex_1based" in df.columns
        else "GlobalIndex"
    )
    if idx_col not in df.columns or "Label" not in df.columns:
        raise RuntimeError(f"{path}: required identity columns missing")

    idx1 = pd.to_numeric(df[idx_col], errors="raise").astype(np.int64).to_numpy()
    labels = pd.to_numeric(df["Label"], errors="raise").astype(np.int64).to_numpy()

    if np.any(idx1 < 1) or np.any(idx1 > N_GLOBAL):
        raise RuntimeError(f"{path}: GlobalIndex out of range")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise RuntimeError(f"{path}: non-binary label")

    return path, idx1 - 1, labels


def load_exact_target_split(target: str, run: int, global_y: np.ndarray):
    exp = EXPECTED_SPLITS[target]
    result = {}

    for split_name, expected_n in [
        ("train", exp["train"]),
        ("validation", exp["val"]),
        ("test", exp["test"]),
    ]:
        path, idx0, labels = read_manifest(target, run, split_name)
        if len(idx0) != expected_n:
            raise RuntimeError(
                f"{target} run_{run:02d} {split_name}: "
                f"{len(idx0)} != expected {expected_n}"
            )

        # Cross-check labels against authoritative prepared global y.
        gy = global_y[idx0]
        if np.any(~np.isfinite(gy)):
            raise RuntimeError(f"{split_name}: unlabeled gene present")
        if not np.array_equal(gy.astype(np.int64), labels):
            bad = np.where(gy.astype(np.int64) != labels)[0][:20]
            raise RuntimeError(
                f"{split_name}: manifest/global-y mismatch at rows {bad.tolist()}"
            )
        result[split_name] = {
            "path": path,
            "idx0": idx0,
            "labels": labels,
        }

    # Disjoint + union.
    tr = set(result["train"]["idx0"].tolist())
    va = set(result["validation"]["idx0"].tolist())
    te = set(result["test"]["idx0"].tolist())
    if tr & va or tr & te or va & te:
        raise RuntimeError("Exact split overlap detected")
    if len(tr | va | te) != N_LABELLED:
        raise RuntimeError(
            f"Formal split union {len(tr|va|te)} != {N_LABELLED} labelled genes"
        )
    if int(result["test"]["labels"].sum()) != exp["test_pos"]:
        raise RuntimeError("Formal Test positive-count audit failed")

    return result


# =============================================================================
# NEIGHBOR-LOADER PREFLIGHT
# =============================================================================

def preflight_neighbor_loader():
    """
    Fail before scanning multi-GB H5 networks if this PyG installation lacks a
    functioning neighbor-sampling backend.
    """
    x = torch.randn(4, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 2], [1, 0, 3, 2, 2, 1]], dtype=torch.long
    )
    data = Data(x=x, edge_index=edge_index)
    loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=2,
        input_nodes=torch.tensor([0, 1], dtype=torch.long),
        shuffle=False,
        num_workers=0,
        subgraph_type="induced",
    )
    try:
        batch = next(iter(loader))
    except Exception as e:
        raise RuntimeError(
            "PyG NeighborLoader preflight failed. Official SEFGNN relies on "
            "neighbor sampling. Install a neighbor-sampling backend compatible "
            "with the local PyTorch/PyG build (typically pyg-lib or torch-sparse), "
            "then re-run.\n\nOriginal error:\n" + repr(e)
        ) from e
    if batch.batch_size != 2:
        raise RuntimeError("NeighborLoader preflight returned unexpected batch size")


# =============================================================================
# MISSING-AWARE BATCH NORMALIZATION
# =============================================================================

class PresenceMaskedBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d that excludes nodes absent from the current PPI view from
    TRAINING batch statistics and running-stat updates.

    Why this is necessary:
    the published SEFGNN formulation assumes the same node set in every view.
    GC-BRB's six PPI lineages have unequal node sets. A gene absent from a view
    must therefore be vacuous for that view. Masking only the final Dirichlet
    alpha is not sufficient during training because the official GNNLR contains
    BatchNorm1d before the evidential head; absent self-only input nodes would
    otherwise alter the normalization seen by genuinely present nodes.

    In eval mode, BatchNorm uses frozen running statistics, so missing nodes do
    not affect other nodes and the standard BatchNorm path is retained.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_presence_mask = None

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm1d):
        new = cls(
            bn.num_features,
            eps=bn.eps,
            momentum=bn.momentum,
            affine=bn.affine,
            track_running_stats=bn.track_running_stats,
        )
        new.load_state_dict(bn.state_dict())
        new.train(bn.training)
        return new

    def forward(self, input):
        mask = self.active_presence_mask
        self.active_presence_mask = None

        # Official behavior in evaluation, or when no mask is supplied.
        if (not self.training) or mask is None:
            return super().forward(input)

        mask = mask.view(-1).bool()
        if mask.numel() != input.shape[0]:
            raise RuntimeError(
                f"Presence-masked BatchNorm size mismatch: "
                f"mask={mask.numel()} input={input.shape[0]}"
            )

        # Full coverage -> exact official BatchNorm behavior.
        if bool(mask.all().item()):
            return super().forward(input)

        out = torch.zeros_like(input)
        n_present = int(mask.sum().item())

        if n_present >= 2:
            out[mask] = super().forward(input[mask])
        elif n_present == 1:
            # A one-row training BatchNorm is undefined. Use the already-frozen
            # running statistics for that one present row and do not update them.
            out[mask] = F.batch_norm(
                input[mask],
                self.running_mean if self.track_running_stats else None,
                self.running_var if self.track_running_stats else None,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        # n_present == 0 -> all-zero output; the whole view is vacuous for
        # target nodes in this mini-batch and must not update BN statistics.
        return out


def _replace_batchnorm_with_presence_mask(module: nn.Module):
    """Recursively replace BatchNorm1d while preserving state_dict key layout."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d) and not isinstance(
            child, PresenceMaskedBatchNorm1d
        ):
            new = PresenceMaskedBatchNorm1d.from_batchnorm(child)
            setattr(module, name, new)
            replaced += 1
        else:
            replaced += _replace_batchnorm_with_presence_mask(child)
    return replaced


def _set_branch_presence_mask(module, args):
    if not args:
        return
    data = args[0]
    if not hasattr(data, "view_presence"):
        raise RuntimeError("view_presence missing from sampled PyG Data batch")
    mask = data.view_presence.view(-1).bool()
    found = 0
    for child in module.modules():
        if isinstance(child, PresenceMaskedBatchNorm1d):
            child.active_presence_mask = mask
            found += 1
    if found == 0:
        raise RuntimeError("No PresenceMaskedBatchNorm1d found in SEFGNN branch")


# =============================================================================
# MISSING-AWARE OFFICIAL SEFGNN ADAPTER
# =============================================================================

class SEFMGNNMissingAware(SEFMGNN):
    """
    The per-view GNN, evidential head, DS_Combin_two(), KL(), ce_loss(), l2(),
    initialization, and SES weight are inherited from the official model.

    Necessary unequal-node-set adaptation:
      - absent view -> alpha=[1,1] (vacuous binary opinion);
      - absent gene-view does not receive per-view evidential CE supervision;
      - embedding-length regularization is applied only where that view exists;
      - SES arithmetic mean is taken over available views only.

    The fused DS opinion still combines all six alphas. A vacuous alpha=[1,1]
    is neutral under the official DS combination.

    V2 correction:
      - absent view nodes are excluded from the GNN branch BatchNorm training
        statistics before the final vacuous-alpha masking.
      - when every node is present, this reduces exactly to official BatchNorm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        total_replaced = 0
        for branch in self.convs:
            n = _replace_batchnorm_with_presence_mask(branch)
            total_replaced += n
            branch.register_forward_pre_hook(_set_branch_presence_mask)

        if total_replaced != len(self.convs):
            raise RuntimeError(
                f"Expected exactly one BatchNorm1d per SEFGNN view branch; "
                f"replaced={total_replaced}, branches={len(self.convs)}"
            )

        self.missing_aware_bn_count = total_replaced

    def forward(self, data_tuple):
        batch_size = int(data_tuple[0].batch_size)

        # Hard paired-batch audit: target nodes must be identical across views.
        ref_targets = data_tuple[0].n_id[:batch_size]
        for i in range(1, len(data_tuple)):
            if not torch.equal(ref_targets, data_tuple[i].n_id[:batch_size]):
                raise RuntimeError(
                    f"Paired NeighborLoader target-order mismatch at view {i}"
                )

        x_list = [self.convs[i](data) for i, data in enumerate(data_tuple)]

        presence_masks = [
            data.view_presence[:batch_size].view(-1).bool()
            for data in data_tuple
        ]

        # Official embedding-length constraint, masked to genuine gene-view pairs.
        length_loss = torch.zeros((), device=x_list[0].device)
        for x, present in zip(x_list, presence_masks):
            if torch.any(present):
                length_loss = length_loss + x[present].norm(dim=1).sum()

        x_evident = [
            F.softplus(self.evidentlayer[i](x))
            for i, x in enumerate(x_list)
        ]
        alphas_raw = [e + 1 for e in x_evident]

        alphas = []
        for alpha, present in zip(alphas_raw, presence_masks):
            alpha_vacuous = torch.ones_like(alpha)
            alpha = torch.where(present.unsqueeze(1), alpha, alpha_vacuous)
            alphas.append(alpha)

        alpha_comb = alphas[0]
        for i in range(1, len(alphas)):
            alpha_comb = self.DS_Combin_two(alpha_comb, alphas[i], 2)

        true_lab = data_tuple[0].y[:batch_size, 1].long()

        # Official combined evidential loss.
        ce_loss = self.ce_loss(true_lab, alpha_comb, 2)

        # Official per-view evidential loss, but only for truly available views.
        for alpha, present in zip(alphas, presence_masks):
            if torch.any(present):
                ce_loss = ce_loss + self.ce_loss(
                    true_lab[present], alpha[present], 2
                )

        # Official DS output.
        ds_prob = torch.softmax(alpha_comb, dim=1)[:, 1].unsqueeze(1)

        # Official SES uses arithmetic mean over views. For unequal node coverage,
        # average only the views actually available for each gene.
        view_probs = [
            torch.softmax(alpha, dim=1)[:, 1].unsqueeze(1)
            for alpha in alphas
        ]
        prob_stack = torch.stack(view_probs, dim=0)             # [V,B,1]
        pres_stack = torch.stack(presence_masks, dim=0).unsqueeze(2).float()
        denom = pres_stack.sum(dim=0).clamp_min(1.0)
        avg_view_score = (prob_stack * pres_stack).sum(dim=0) / denom

        no_view = pres_stack.sum(dim=0) == 0
        if torch.any(no_view):
            avg_view_score = torch.where(
                no_view, torch.full_like(avg_view_score, 0.5), avg_view_score
            )

        out = (
            OFFICIAL_SES_DS_WEIGHT * ds_prob
            + (1.0 - OFFICIAL_SES_DS_WEIGHT) * avg_view_score
        )

        l2_loss = self.l2()
        constraint_loss = 0.1 * l2_loss + 0.1 * length_loss

        fused_uncertainty = 2.0 / torch.sum(alpha_comb, dim=1, keepdim=True)
        per_view_uncertainty = torch.stack(
            [2.0 / torch.sum(a, dim=1) for a in alphas],
            dim=1,
        )

        constraint_dict = {
            "constraint_loss": constraint_loss,
            "ce_loss": ce_loss,
            "avg_view_score": avg_view_score,
            "alpha": OFFICIAL_SES_DS_WEIGHT,
            "ds_prob": ds_prob,
            "fused_uncertainty": fused_uncertainty,
            "per_view_uncertainty": per_view_uncertainty,
            "presence_count": pres_stack.sum(dim=0).view(-1),
        }
        return out, x_list, self.ppi_weight, constraint_dict


# =============================================================================
# DATA OBJECTS / LOADERS
# =============================================================================

def make_onehot_y(global_y: np.ndarray):
    # Unlabeled rows are arbitrary class-0 placeholders; they are never used
    # as supervised input_nodes.
    y01 = np.zeros(N_GLOBAL, dtype=np.int64)
    labelled = np.isfinite(global_y)
    y01[labelled] = global_y[labelled].astype(np.int64)
    y = np.zeros((N_GLOBAL, 2), dtype=np.float32)
    y[:, 0] = 1.0
    y[y01 == 1, 0] = 0.0
    y[y01 == 1, 1] = 1.0
    return torch.from_numpy(y)


def make_data_views(X, global_y, presence, local_to_global):
    x = torch.from_numpy(X.astype(np.float32, copy=False))
    y = make_onehot_y(global_y)
    data_list = []
    graph_stats = []

    for j, view in enumerate(VIEWS):
        edge_np = build_or_load_edge_index(view, local_to_global[view])
        edge_index = torch.from_numpy(edge_np).long()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            view_presence=torch.from_numpy(presence[:, j].astype(np.bool_)),
        )
        data.num_nodes = N_GLOBAL
        data_list.append(data)

        graph_stats.append({
            "View": view,
            "PresentGenes": int(presence[:, j].sum()),
            "DirectedEdges": int(edge_index.shape[1]),
            "UndirectedEdges": int(edge_index.shape[1] // 2),
        })

    return data_list, pd.DataFrame(graph_stats)


def downsample_train_negatives(train_idx, global_y, drop_neg, seed):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    labels = global_y[train_idx].astype(np.int64)

    neg = train_idx[labels == 0].tolist()
    pos = train_idx[labels == 1].tolist()

    rng = random.Random(seed)
    n_drop = int(len(neg) * drop_neg)
    drop = set(rng.sample(neg, n_drop)) if n_drop > 0 else set()

    kept = np.array(
        sorted([i for i in train_idx.tolist() if i not in drop]),
        dtype=np.int64,
    )
    return kept, len(neg), len(pos), n_drop


def make_loaders(data_list, input_idx, batch_size, num_neighbors, shuffle=False):
    idx = torch.as_tensor(
        np.asarray(sorted(map(int, input_idx)), dtype=np.int64),
        dtype=torch.long,
    )
    loaders = []
    for data in data_list:
        loaders.append(
            NeighborLoader(
                data=data,
                num_neighbors=list(num_neighbors),
                batch_size=batch_size,
                input_nodes=idx,
                shuffle=shuffle,
                num_workers=0,
                subgraph_type="induced",
            )
        )
    return loaders


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def metric_dict(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=float)
    pred = (score > 0.5).astype(np.int64)

    return {
        "N": int(len(y_true)),
        "Positive": int(y_true.sum()),
        "AUPRC": float(average_precision_score(y_true, score)),
        "AUROC": float(roc_auc_score(y_true, score)),
        "F1_0p5": float(f1_score(y_true, pred, zero_division=0)),
        "Accuracy_0p5": float(accuracy_score(y_true, pred)),
        "TN": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 0]),
        "FP": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 1]),
        "FN": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 0]),
        "TP": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 1]),
    }


@torch.no_grad()
def evaluate(model, loaders, device, expected_target_idx=None, collect_details=False):
    model.eval()
    all_idx, all_y, all_score = [], [], []
    all_u, all_presence = [], []

    for data_tuple in zip(*loaders):
        data_tuple = [d.to(device) for d in data_tuple]
        b = int(data_tuple[0].batch_size)

        ref_idx = data_tuple[0].n_id[:b]
        for d in data_tuple[1:]:
            if not torch.equal(ref_idx, d.n_id[:b]):
                raise RuntimeError("Evaluation paired-batch target mismatch")

        out, _, _, aux = model(data_tuple)
        y = data_tuple[0].y[:b, 1].long()

        all_idx.append(ref_idx.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        all_score.append(out.view(-1).detach().cpu().numpy())
        all_u.append(aux["fused_uncertainty"].view(-1).detach().cpu().numpy())
        all_presence.append(aux["presence_count"].view(-1).detach().cpu().numpy())

    idx = np.concatenate(all_idx)
    y = np.concatenate(all_y)
    score = np.concatenate(all_score)
    u = np.concatenate(all_u)
    pc = np.concatenate(all_presence)

    if len(np.unique(idx)) != len(idx):
        raise RuntimeError("Evaluation returned duplicate target nodes")

    if expected_target_idx is not None:
        if set(idx.tolist()) != set(map(int, expected_target_idx)):
            missing = sorted(set(map(int, expected_target_idx)) - set(idx.tolist()))[:20]
            extra = sorted(set(idx.tolist()) - set(map(int, expected_target_idx)))[:20]
            raise RuntimeError(
                f"Evaluation target set mismatch. Missing={missing}, extra={extra}"
            )

    metrics = metric_dict(y, score)

    details = None
    if collect_details:
        details = pd.DataFrame({
            "GlobalIndex_0based": idx.astype(np.int64),
            "GlobalIndex_1based": idx.astype(np.int64) + 1,
            "Label": y.astype(np.int64),
            "Score": score,
            "FusedUncertainty": u,
            "PresentViews": pc.astype(np.int64),
        }).sort_values("GlobalIndex_1based").reset_index(drop=True)

    return metrics, details


def train_one_epoch(model, loaders, optimizer, device):
    model.train()
    total = 0.0
    steps = 0

    for data_tuple in zip(*loaders):
        data_tuple = [d.to(device) for d in data_tuple]
        b = int(data_tuple[0].batch_size)

        ref_idx = data_tuple[0].n_id[:b]
        for d in data_tuple[1:]:
            if not torch.equal(ref_idx, d.n_id[:b]):
                raise RuntimeError("Training paired-batch target mismatch")

        optimizer.zero_grad(set_to_none=True)

        out, _, _, aux = model(data_tuple)
        loss = aux["ce_loss"] + aux["constraint_loss"] * 1e-4

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu())
        steps += 1

    return total / max(steps, 1)


def train_adapter(
    data_list,
    split,
    device,
    model_seed,
    lr,
    epochs,
    batch_size,
    num_neighbors,
    drop_neg,
    out_dir,
):
    set_seed(model_seed)

    train_idx_full = split["train"]["idx0"]
    val_idx = split["validation"]["idx0"]
    test_idx = split["test"]["idx0"]

    train_idx, nneg, npos, ndrop = downsample_train_negatives(
        train_idx_full, split["_global_y"], drop_neg, model_seed
    )

    print(
        f"\nTraining input after official negative downsampling: "
        f"{len(train_idx):,} (pos={npos:,}, neg kept={nneg-ndrop:,}, neg dropped={ndrop:,})"
    )

    train_loaders = make_loaders(
        data_list, train_idx, batch_size, num_neighbors, shuffle=False
    )
    val_loaders = make_loaders(
        data_list, val_idx, batch_size, num_neighbors, shuffle=False
    )

    model_args = {"gpu": str(device)}
    model = SEFMGNNMissingAware(
        in_channels=64,
        hidden_channels=OFFICIAL_HIDDEN_CHANNELS,
        num_ppi=len(VIEWS),
        residual=OFFICIAL_RESIDUAL,
        learnable_weight=True,
        args=model_args,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=OFFICIAL_WEIGHT_DECAY,
    )
    scheduler = make_linear_warmup_scheduler(
        optimizer,
        num_training_steps=epochs,
        warmup_fraction=OFFICIAL_WARMUP_FRACTION,
    )

    history = []
    best_ap = -np.inf
    best_epoch = -1
    best_state = None
    patience = 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loaders, optimizer, device
        )
        scheduler.step()

        val_metrics, _ = evaluate(
            model,
            val_loaders,
            device,
            expected_target_idx=val_idx,
            collect_details=False,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "Epoch": epoch + 1,
            "TrainLoss": train_loss,
            "ValAUPRC": val_metrics["AUPRC"],
            "ValAUROC": val_metrics["AUROC"],
            "ValF1_0p5": val_metrics["F1_0p5"],
            "LR": current_lr,
            "Seconds": time.time() - t0,
        }
        history.append(row)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"loss={train_loss:.5f} | "
            f"Val AP={val_metrics['AUPRC']:.5f} "
            f"AUROC={val_metrics['AUROC']:.5f} "
            f"F1={val_metrics['F1_0p5']:.5f} | "
            f"{row['Seconds']:.1f}s"
        )

        if val_metrics["AUPRC"] >= best_ap:
            best_ap = val_metrics["AUPRC"]
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if (
            epochs >= 100
            and patience > OFFICIAL_EARLY_PATIENCE
            and (epoch + 1) > epochs * OFFICIAL_EARLY_AFTER_FRACTION
        ):
            print(
                f"Early stopping at epoch {epoch+1}; "
                f"best Val AUPRC={best_ap:.6f} at epoch {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")

    model.load_state_dict(best_state)
    model.to(device)

    # Validation checkpoint frozen. Test loader is created/evaluated only now.
    test_loaders = make_loaders(
        data_list, test_idx, batch_size, num_neighbors, shuffle=False
    )
    test_metrics, test_details = evaluate(
        model,
        test_loaders,
        device,
        expected_target_idx=test_idx,
        collect_details=True,
    )

    # Re-evaluate best validation checkpoint for final report.
    val_metrics, val_details = evaluate(
        model,
        val_loaders,
        device,
        expected_target_idx=val_idx,
        collect_details=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        out_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    val_details.to_csv(
        out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    test_details.to_csv(
        out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_auprc": best_ap,
            "model_seed": model_seed,
            "lr": lr,
        },
        out_dir / "best_validation_auprc_checkpoint.pt",
    )

    return {
        "BestEpoch": best_epoch,
        "BestValAUPRC": float(best_ap),
        "Validation": val_metrics,
        "Test": test_metrics,
        "TrainInputNAfterNegativeDrop": int(len(train_idx)),
        "DroppedNegativeTrainN": int(ndrop),
    }


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SEFGNN official-code adaptation to exact GC-BRB benchmark"
    )
    p.add_argument("--target", default="CPDB", choices=VIEWS)
    p.add_argument("--run", default=1, type=int, choices=list(RUNS := range(1, 6)))
    p.add_argument("--mode", default="smoke", choices=["smoke", "formal"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--batch-size", type=int, default=OFFICIAL_BATCH_SIZE)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--rebuild-edge-cache",
        action="store_true",
        help="Delete cached global edge_index files and rebuild all six views.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    epochs = (
        args.epochs
        if args.epochs is not None
        else (3 if args.mode == "smoke" else OFFICIAL_NUM_EPOCHS)
    )

    if args.mode == "formal" and epochs < 50:
        print(
            "[WARNING] formal mode with <50 epochs is unusual; "
            "this will not reproduce the intended 300-epoch training budget."
        )

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("SEFGNN OFFICIAL-CODE × EXACT GC-BRB BENCHMARK ADAPTER")
    print(f"Target={args.target} | run_{args.run:02d} | mode={args.mode.upper()}")
    print("SMOKE results are NOT manuscript results." if args.mode == "smoke" else "FORMAL training mode")
    print("=" * 100)
    print(f"Python          : {sys.executable}")
    print(f"PyTorch         : {torch.__version__}")
    import torch_geometric
    print(f"PyG             : {torch_geometric.__version__}")
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(args.gpu)}")
    print(f"Official source : {OFFICIAL_REPO}")
    print(f"LR              : {args.lr}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Neighbors       : {OFFICIAL_NUM_NEIGHBORS}")
    print(f"Negative drop   : {OFFICIAL_DROP_NEG}")
    print(f"Views           : {VIEWS}")

    print("\n[1/7] PyG NeighborLoader preflight...")
    preflight_neighbor_loader()
    print("  PASS")

    if args.rebuild_edge_cache:
        cache_dir = CACHE_ROOT / "global_edge_index"
        if cache_dir.exists():
            for p in cache_dir.glob("*.npz"):
                p.unlink()
        print("  Existing edge caches removed.")

    print("\n[2/7] Loading authoritative GC-BRB aligned X/y/presence/local_to_global...")
    X, global_y, presence, local_to_global = load_prepared_alignment()
    print(f"  X         : {X.shape}")
    print(f"  presence  : {presence.shape}")
    print(
        f"  labels    : finite={np.isfinite(global_y).sum():,}, "
        f"+{int((global_y==1).sum()):,}/-{int((global_y==0).sum()):,}"
    )

    print("\n[3/7] Loading exact frozen target/run manifest...")
    split = load_exact_target_split(args.target, args.run, global_y)
    split["_global_y"] = global_y
    for s in ["train", "validation", "test"]:
        print(
            f"  {s:10s}: N={len(split[s]['idx0']):,}, "
            f"positive={int(split[s]['labels'].sum()):,}"
        )

    print("\n[4/7] Building/loading six exact global PPI edge caches...")
    data_list, graph_stats = make_data_views(
        X, global_y, presence, local_to_global
    )
    print(graph_stats.to_string(index=False))

    # Split-level view availability audit.
    avail_rows = []
    for s in ["train", "validation", "test"]:
        idx = split[s]["idx0"]
        counts = presence[idx].sum(axis=1)
        avail_rows.append({
            "Split": s,
            "N": len(idx),
            "MeanPresentViews": float(counts.mean()),
            "MinPresentViews": int(counts.min()),
            "MaxPresentViews": int(counts.max()),
            "AllSixViewsPct": float(np.mean(counts == 6)),
            "AtLeastOneViewPct": float(np.mean(counts >= 1)),
        })
    avail_df = pd.DataFrame(avail_rows)
    print("\nView availability:")
    print(avail_df.to_string(index=False))

    run_out = (
        RESULT_ROOT / args.target / f"run_{args.run:02d}" /
        f"{args.mode}_lr_{args.lr:g}_seed_{args.seed}"
    )
    run_out.mkdir(parents=True, exist_ok=True)
    graph_stats.to_csv(
        run_out / "graph_stats.csv", index=False, encoding="utf-8-sig"
    )
    avail_df.to_csv(
        run_out / "split_view_availability.csv",
        index=False, encoding="utf-8-sig"
    )

    config = {
        "target": args.target,
        "run": args.run,
        "mode": args.mode,
        "epochs": epochs,
        "lr": args.lr,
        "model_seed": args.seed,
        "device": str(device),
        "views": VIEWS,
        "num_neighbors": OFFICIAL_NUM_NEIGHBORS,
        "batch_size": args.batch_size,
        "weight_decay": OFFICIAL_WEIGHT_DECAY,
        "drop_neg": OFFICIAL_DROP_NEG,
        "residual": OFFICIAL_RESIDUAL,
        "hidden_channels": OFFICIAL_HIDDEN_CHANNELS,
        "ses_ds_weight": OFFICIAL_SES_DS_WEIGHT,
        "checkpoint_selection": "Validation AUPRC",
        "test_policy": "Evaluate once after checkpoint freeze",
        "missing_view_policy": "Vacuous Dirichlet alpha=[1,1]",
        "ses_average_policy": "Mean over available views only",
        "official_model_py": str(OFFICIAL_REPO / "model.py"),
        "prepared_mat": str(PREPARED_MAT),
    }
    (run_out / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[5/7] Training official-architecture SEFGNN adapter...")
    t0 = time.time()
    result = train_adapter(
        data_list=data_list,
        split=split,
        device=device,
        model_seed=args.seed,
        lr=args.lr,
        epochs=epochs,
        batch_size=args.batch_size,
        num_neighbors=OFFICIAL_NUM_NEIGHBORS,
        drop_neg=OFFICIAL_DROP_NEG,
        out_dir=run_out,
    )

    print("\n[6/7] Frozen-checkpoint Test result (evaluated once)...")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    (run_out / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status = {
        "Status": "PASS",
        "Purpose": "SMOKE_ONLY_NOT_FOR_MANUSCRIPT"
            if args.mode == "smoke" else "FORMAL_SINGLE_RUN",
        "Target": args.target,
        "Run": args.run,
        "Mode": args.mode,
        "ExactSplitUsed": True,
        "SixPPIUsed": True,
        "MissingViewVacuousOpinion": True,
        "TestEvaluatedAfterCheckpointFreeze": True,
        "SecondsTotal": time.time() - t0,
        "BestEpoch": result["BestEpoch"],
        "ValidationAUPRC": result["Validation"]["AUPRC"],
        "TestAUPRC": result["Test"]["AUPRC"],
        "TestAUROC": result["Test"]["AUROC"],
    }
    pd.DataFrame([status]).to_csv(
        run_out / "FINAL_ADAPTER_STATUS.csv",
        index=False, encoding="utf-8-sig"
    )

    report = [
        "SEFGNN × GC-BRB CPDB ADAPTER STATUS",
        "=" * 80,
        f"Target / run: {args.target} / run_{args.run:02d}",
        f"Mode: {args.mode.upper()}",
        f"Exact frozen GC-BRB split: PASS",
        f"Six PPI views: PASS",
        f"Unequal coverage handling: vacuous alpha=[1,1]",
        f"Validation-only checkpoint selection: PASS",
        f"Test evaluated only after checkpoint freeze: PASS",
        "",
        f"Best epoch: {result['BestEpoch']}",
        f"Validation AUPRC: {result['Validation']['AUPRC']:.6f}",
        f"Test AUPRC: {result['Test']['AUPRC']:.6f}",
        f"Test AUROC: {result['Test']['AUROC']:.6f}",
        "",
        (
            "STATUS_CODE: CPDB_SMOKE_PASS_NOT_FOR_MANUSCRIPT"
            if args.mode == "smoke"
            else "STATUS_CODE: FORMAL_SINGLE_RUN_COMPLETE"
        ),
    ]
    (run_out / "FINAL_ADAPTER_REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n[7/7] COMPLETE")
    print("\n".join(report))
    print(f"\nOutput directory:\n{run_out}")
    print("=" * 100)


if __name__ == "__main__":
    main()

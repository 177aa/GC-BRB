# -*- coding: utf-8 -*-
"""
SEFGNN adapter V2 pre-cloud equivalence audit
=============================================

FAST / SYNTHETIC / NO TRAINING DATA / NO TEST DATA.

Checks:
1) Official public-repository SEFMGNN and our V2 adapter have identical
   parameter/state_dict structure.
2) When all views are present, V2 collapses to the official model exactly
   (eval-mode and train-mode forward/loss within tight tolerance).
3) Vacuous alpha=[1,1] is neutral under the official DS combination.
4) Every branch has exactly one presence-masked BatchNorm in V2.
5) Missing-view BatchNorm excludes absent nodes from training statistics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SEFGNN_official"))

from model import SEFMGNN as OfficialSEFMGNN  # noqa: E402
import run_sefgnn_gcbrb_adapter_cloud_v2 as V2  # noqa: E402


def seed_all(s=2025):
    V2.set_seed(s)


def make_tuple(num_views=6, n=24, batch_size=12, all_present=True):
    g = torch.Generator().manual_seed(77)
    x = torch.randn(n, 64, generator=g)

    # Deterministic small undirected ring + skip edges.
    edges = []
    for i in range(n):
        for j in [(i + 1) % n, (i + 3) % n]:
            edges += [(i, j), (j, i)]
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

    y01 = torch.zeros(n, 2)
    y01[:, 0] = 1
    y01[::4, 0] = 0
    y01[::4, 1] = 1

    out = []
    for v in range(num_views):
        presence = torch.ones(n, dtype=torch.bool)
        if not all_present:
            # Only target/input nodes are made missing here.
            # They are isolated from graph edges in a real GC-BRB missing view,
            # but this synthetic test focuses on the BN mask behavior.
            presence[1:batch_size:3] = False

        d = Data(
            x=x.clone(),
            edge_index=edge_index.clone(),
            y=y01.clone(),
            view_presence=presence,
            n_id=torch.arange(n),
        )
        d.batch_size = batch_size
        out.append(d)
    return out


def clone_state(src, dst):
    a = src.state_dict()
    b = dst.state_dict()
    if list(a.keys()) != list(b.keys()):
        missing = [k for k in a if k not in b]
        extra = [k for k in b if k not in a]
        raise RuntimeError(
            f"State dict key mismatch. missing={missing[:20]} extra={extra[:20]}"
        )
    dst.load_state_dict(a, strict=True)


def compare_tensor(name, a, b, atol=2e-6, rtol=2e-6):
    diff = float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())))
    ok = torch.allclose(a.detach().cpu(), b.detach().cpu(), atol=atol, rtol=rtol)
    print(f"{name:32s} max_abs_diff={diff:.3e} PASS={ok}")
    return bool(ok), diff


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 92)
    print("SEFGNN ADAPTER V2 PRE-CLOUD EQUIVALENCE AUDIT")
    print("=" * 92)
    print("Torch:", torch.__version__)
    import torch_geometric
    print("PyG:", torch_geometric.__version__)
    print("Device:", device)

    args = {"gpu": str(device)}

    seed_all(2025)
    official = OfficialSEFMGNN(
        in_channels=64, hidden_channels=64, num_ppi=6,
        residual=1.0, learnable_weight=True, args=args
    ).to(device)

    seed_all(9999)
    adapted = V2.SEFMGNNMissingAware(
        in_channels=64, hidden_channels=64, num_ppi=6,
        residual=1.0, learnable_weight=True, args=args
    ).to(device)

    clone_state(official, adapted)
    print("\n[1] State dict structure/load: PASS")

    masked_bn_count = sum(
        isinstance(m, V2.PresenceMaskedBatchNorm1d)
        for m in adapted.modules()
    )
    print(f"[2] Presence-masked BN count: {masked_bn_count}")
    if masked_bn_count != 6:
        raise RuntimeError("Expected exactly 6 presence-masked BN layers")

    data = [d.to(device) for d in make_tuple(all_present=True)]

    # Eval-mode equivalence.
    official.eval()
    adapted.eval()
    with torch.no_grad():
        seed_all(123)
        o_out, _, _, o_aux = official(data)
        seed_all(123)
        a_out, _, _, a_aux = adapted(data)

    checks = []
    checks.append(compare_tensor("eval final output", o_out, a_out))
    checks.append(compare_tensor("eval ce_loss", o_aux["ce_loss"], a_aux["ce_loss"]))
    checks.append(compare_tensor(
        "eval constraint_loss", o_aux["constraint_loss"], a_aux["constraint_loss"]
    ))
    checks.append(compare_tensor(
        "eval avg_view_score", o_aux["avg_view_score"], a_aux["avg_view_score"]
    ))

    # Train-mode equivalence on fresh identical models/states.
    seed_all(2025)
    official_t = OfficialSEFMGNN(
        in_channels=64, hidden_channels=64, num_ppi=6,
        residual=1.0, learnable_weight=True, args=args
    ).to(device)
    seed_all(9999)
    adapted_t = V2.SEFMGNNMissingAware(
        in_channels=64, hidden_channels=64, num_ppi=6,
        residual=1.0, learnable_weight=True, args=args
    ).to(device)
    clone_state(official_t, adapted_t)

    official_t.train()
    adapted_t.train()

    seed_all(456)
    o_out_t, _, _, o_aux_t = official_t(data)
    seed_all(456)
    a_out_t, _, _, a_aux_t = adapted_t(data)

    checks.append(compare_tensor("train final output", o_out_t, a_out_t, 5e-6, 5e-6))
    checks.append(compare_tensor("train ce_loss", o_aux_t["ce_loss"], a_aux_t["ce_loss"], 5e-6, 5e-6))
    checks.append(compare_tensor(
        "train constraint_loss", o_aux_t["constraint_loss"],
        a_aux_t["constraint_loss"], 5e-6, 5e-6
    ))

    # Vacuous DS identity.
    seed_all(88)
    alpha = 1.0 + torch.rand(32, 2, device=device) * 4.0
    vac = torch.ones_like(alpha)
    fused = official.DS_Combin_two(alpha, vac, 2)
    vac_ok, vac_diff = compare_tensor("DS(alpha, vacuous)==alpha", fused, alpha, 2e-6, 2e-6)
    checks.append((vac_ok, vac_diff))

    # Direct BN behavior: absent rows must not change training running stats.
    bn = next(m for m in adapted_t.modules()
              if isinstance(m, V2.PresenceMaskedBatchNorm1d))
    bn.train()
    feat = torch.randn(20, bn.num_features, device=device)
    mask = torch.zeros(20, dtype=torch.bool, device=device)
    mask[:12] = True

    # Compare masked BN to a plain BN fed only present rows, same initial state.
    plain = torch.nn.BatchNorm1d(
        bn.num_features, eps=bn.eps, momentum=bn.momentum,
        affine=bn.affine, track_running_stats=bn.track_running_stats
    ).to(device)
    plain.load_state_dict(bn.state_dict())
    bn_ref_state = {k: v.detach().clone() for k, v in bn.state_dict().items()}

    bn.active_presence_mask = mask
    y_masked = bn(feat)

    plain.load_state_dict(bn_ref_state)
    y_plain_present = plain(feat[mask])

    bn_value_ok = torch.allclose(
        y_masked[mask], y_plain_present, atol=2e-6, rtol=2e-6
    )
    bn_absent_zero = bool(torch.all(y_masked[~mask] == 0).item())
    print(f"masked BN present-row equivalence  PASS={bn_value_ok}")
    print(f"masked BN absent rows zero         PASS={bn_absent_zero}")
    checks.append((bool(bn_value_ok), 0.0))
    checks.append((bn_absent_zero, 0.0))

    overall = all(x[0] for x in checks)
    print("\n" + "=" * 92)
    print("STATUS_CODE:",
          "SEFGNN_ADAPTER_V2_EQUIVALENCE_PASS" if overall
          else "SEFGNN_ADAPTER_V2_EQUIVALENCE_FAIL")
    print("=" * 92)

    report = {
        "torch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "device": str(device),
        "masked_bn_count": masked_bn_count,
        "all_checks_pass": overall,
        "status": (
            "SEFGNN_ADAPTER_V2_EQUIVALENCE_PASS" if overall
            else "SEFGNN_ADAPTER_V2_EQUIVALENCE_FAIL"
        ),
    }
    (ROOT / "SEFGNN_ADAPTER_V2_EQUIVALENCE_AUDIT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

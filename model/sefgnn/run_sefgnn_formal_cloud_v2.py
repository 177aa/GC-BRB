# -*- coding: utf-8 -*-
r"""
SEFGNN training and evaluation for the GC-BRB benchmark.

The workflow uses six network views and five runs per network. The checkpoint is selected by validation AUPRC and evaluated on the test set. Validation and test sampling use fixed evaluation seeds. Completed runs marked PASS are skipped.

Keep this script, run_sefgnn_gcbrb_adapter_cloud_v2.py and the SEFGNN_official folder in the same directory.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Cloud/Linux path resolution. Keep this script beside the adapter and
# SEFGNN_official folder. Override with SEFGNN_WORKDIR/GCBRB_ROOT if needed.
SCRIPT_ROOT = Path(
    os.environ.get("SEFGNN_WORKDIR", Path(__file__).resolve().parent)
).resolve()
PROJECT_ROOT = Path(
    os.environ.get("GCBRB_ROOT", SCRIPT_ROOT / "GC_BRB_3LAYER_FINAL")
).resolve()
sys.path.insert(0, str(SCRIPT_ROOT))

import run_sefgnn_gcbrb_adapter_cloud_v2 as A  # noqa: E402


# ---------------------------------------------------------------------
# Frozen formal protocol
# ---------------------------------------------------------------------

FROZEN_LR = 1e-4
FORMAL_EPOCH_BUDGET = 300
BASE_MODEL_SEED = 2025

OUT_ROOT = (
    PROJECT_ROOT / "results" / "sefgnn_same_benchmark" /
    "08_formal_deterministic_v2"
)

TARGET_ORDER = list(A.VIEWS)


def target_index(target: str) -> int:
    return TARGET_ORDER.index(target) + 1


def model_seed_for_run(run: int) -> int:
    # Paired seed sequence across targets.
    return BASE_MODEL_SEED + (run - 1)


def eval_seed(target: str, run: int, split: str) -> int:
    """
    Stable split-specific evaluation seeds.
    Distinct from model-training seeds.
    """
    split_code = {"validation": 1, "test": 2}[split]
    return 2026091400 + 1000 * target_index(target) + 10 * run + split_code


# ---------------------------------------------------------------------
# RNG isolation
# ---------------------------------------------------------------------

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextlib.contextmanager
def isolated_eval_rng(seed: int):
    """
    Deterministic evaluation without changing the training RNG trajectory.
    """
    state = capture_rng_state()
    try:
        A.set_seed(seed)
        yield
    finally:
        restore_rng_state(state)


# ---------------------------------------------------------------------
# Model / loaders
# ---------------------------------------------------------------------

def make_model(device, model_seed):
    A.set_seed(model_seed)
    model_args = {"gpu": str(device)}
    model = A.SEFMGNNMissingAware(
        in_channels=64,
        hidden_channels=A.OFFICIAL_HIDDEN_CHANNELS,
        num_ppi=len(A.VIEWS),
        residual=A.OFFICIAL_RESIDUAL,
        learnable_weight=True,
        args=model_args,
    ).to(device)
    return model


def make_split_loaders(data_list, idx, batch_size):
    return A.make_loaders(
        data_list,
        idx,
        batch_size,
        A.OFFICIAL_NUM_NEIGHBORS,
        shuffle=False,
    )


# ---------------------------------------------------------------------
# Deterministic evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_detailed(
    model,
    loaders,
    device,
    expected_idx,
    seed,
    collect_details,
):
    """
    Re-seeds evaluation sampling inside an isolated RNG context.
    Validation/Test sampling therefore repeats while training RNG remains untouched.
    """
    with isolated_eval_rng(seed):
        model.eval()

        all_idx = []
        all_y = []
        all_score = []
        all_ds = []
        all_avg = []
        all_fu = []
        all_pvu = []
        all_presence = []

        for data_tuple in zip(*loaders):
            data_tuple = [d.to(device) for d in data_tuple]
            b = int(data_tuple[0].batch_size)

            ref_idx = data_tuple[0].n_id[:b]
            for d in data_tuple[1:]:
                if not torch.equal(ref_idx, d.n_id[:b]):
                    raise RuntimeError("Paired NeighborLoader target-order mismatch")

            out, _, _, aux = model(data_tuple)
            y = data_tuple[0].y[:b, 1].long()

            all_idx.append(ref_idx.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_score.append(out.view(-1).detach().cpu().numpy())
            all_ds.append(aux["ds_prob"].view(-1).detach().cpu().numpy())
            all_avg.append(aux["avg_view_score"].view(-1).detach().cpu().numpy())
            all_fu.append(aux["fused_uncertainty"].view(-1).detach().cpu().numpy())
            all_pvu.append(aux["per_view_uncertainty"].detach().cpu().numpy())
            all_presence.append(aux["presence_count"].view(-1).detach().cpu().numpy())

        idx = np.concatenate(all_idx)
        y = np.concatenate(all_y)
        score = np.concatenate(all_score)
        ds = np.concatenate(all_ds)
        avg = np.concatenate(all_avg)
        fu = np.concatenate(all_fu)
        pvu = np.concatenate(all_pvu, axis=0)
        pc = np.concatenate(all_presence)

        if len(np.unique(idx)) != len(idx):
            raise RuntimeError("Evaluation returned duplicate target nodes")
        if set(idx.tolist()) != set(map(int, expected_idx)):
            raise RuntimeError("Evaluation target identity mismatch")

        metrics = A.metric_dict(y, score)

        details = None
        if collect_details:
            d = {
                "GlobalIndex_0based": idx.astype(np.int64),
                "GlobalIndex_1based": idx.astype(np.int64) + 1,
                "Label": y.astype(np.int64),
                "SES_FinalScore": score,
                "DS_Score": ds,
                "AvailableViewMeanScore": avg,
                "FusedUncertainty": fu,
                "PresentViews": pc.astype(np.int64),
            }
            for j, view in enumerate(A.VIEWS):
                d[f"Uncertainty_{view}"] = pvu[:, j]

            details = (
                pd.DataFrame(d)
                .sort_values("GlobalIndex_1based")
                .reset_index(drop=True)
            )

        return metrics, details


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def run_one_formal(
    target,
    run,
    data_list,
    global_y,
    device,
    epochs,
    batch_size,
):
    out_dir = OUT_ROOT / target / f"run_{run:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    status_path = out_dir / "FINAL_FORMAL_STATUS.json"
    if status_path.exists():
        try:
            old = json.loads(status_path.read_text(encoding="utf-8"))
            if old.get("Status") == "PASS":
                print(
                    f"\n[SKIP] {target} run_{run:02d} already PASS "
                    f"(Test AP={old.get('TestAUPRC')})"
                )
                return old
        except Exception:
            pass

    model_seed = model_seed_for_run(run)
    val_seed = eval_seed(target, run, "validation")
    test_seed = eval_seed(target, run, "test")

    print("\n" + "=" * 100)
    print(f"FORMAL | {target} | run_{run:02d}")
    print("=" * 100)
    print(f"Model seed       : {model_seed}")
    print(f"Validation seed  : {val_seed}")
    print(f"Test seed        : {test_seed}")
    print(f"Frozen LR        : {FROZEN_LR:g}")
    print(f"Epoch budget     : {epochs}")

    split = A.load_exact_target_split(target, run, global_y)
    split["_global_y"] = global_y

    train_idx_full = split["train"]["idx0"]
    val_idx = split["validation"]["idx0"]
    test_idx = split["test"]["idx0"]

    # Official code behavior retained from adapter: 30% negative downsampling.
    train_idx, nneg, npos, ndrop = A.downsample_train_negatives(
        train_idx_full,
        global_y,
        A.OFFICIAL_DROP_NEG,
        model_seed,
    )

    train_loaders = make_split_loaders(data_list, train_idx, batch_size)
    val_loaders = make_split_loaders(data_list, val_idx, batch_size)

    # IMPORTANT: no Test loader is created at this point.

    A.set_seed(model_seed)
    model = make_model(device, model_seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_LR,
        weight_decay=A.OFFICIAL_WEIGHT_DECAY,
    )
    scheduler = A.make_linear_warmup_scheduler(
        optimizer,
        num_training_steps=epochs,
        warmup_fraction=A.OFFICIAL_WARMUP_FRACTION,
    )

    best_ap = -np.inf
    best_epoch = -1
    best_state = None
    best_val_metrics = None
    patience = 0
    history = []

    t_run = time.time()

    for epoch in range(epochs):
        t0 = time.time()

        # Training RNG proceeds naturally. The validation call below restores it.
        train_loss = A.train_one_epoch(
            model,
            train_loaders,
            optimizer,
            device,
        )
        scheduler.step()

        val_metrics, _ = evaluate_detailed(
            model=model,
            loaders=val_loaders,
            device=device,
            expected_idx=val_idx,
            seed=val_seed,
            collect_details=False,
        )

        row = {
            "Epoch": epoch + 1,
            "TrainLoss": train_loss,
            "ValidationAUPRC": val_metrics["AUPRC"],
            "ValidationAUROC": val_metrics["AUROC"],
            "ValidationF1_0p5": val_metrics["F1_0p5"],
            "LR": float(optimizer.param_groups[0]["lr"]),
            "Seconds": time.time() - t0,
        }
        history.append(row)

        print(
            f"{target:10s} run_{run:02d} | "
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"loss={train_loss:.5f} | "
            f"Val AP={val_metrics['AUPRC']:.5f} "
            f"AUROC={val_metrics['AUROC']:.5f} | "
            f"{row['Seconds']:.1f}s"
        )

        if val_metrics["AUPRC"] >= best_ap:
            best_ap = float(val_metrics["AUPRC"])
            best_epoch = epoch + 1
            best_val_metrics = dict(val_metrics)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if (
            epochs >= 100
            and patience > A.OFFICIAL_EARLY_PATIENCE
            and (epoch + 1) > epochs * A.OFFICIAL_EARLY_AFTER_FRACTION
        ):
            print(
                f"EARLY STOP | best Val AP={best_ap:.6f} "
                f"at epoch {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(f"{target} run_{run:02d}: no best checkpoint")

    # Save history/checkpoint BEFORE touching Test.
    pd.DataFrame(history).to_csv(
        out_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ckpt_path = out_dir / "best_validation_auprc_checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "target": target,
            "run": run,
            "model_seed": model_seed,
            "validation_eval_seed": val_seed,
            "test_eval_seed": test_seed,
            "learning_rate": FROZEN_LR,
            "best_epoch": best_epoch,
            "best_validation_auprc": best_ap,
            "best_validation_metrics": best_val_metrics,
        },
        ckpt_path,
    )

    # Freeze and reload the selected checkpoint.
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Deterministic re-evaluation of frozen Validation checkpoint.
    final_val_metrics, val_details = evaluate_detailed(
        model=model,
        loaders=val_loaders,
        device=device,
        expected_idx=val_idx,
        seed=val_seed,
        collect_details=True,
    )

    # Test loader is created ONLY NOW, after checkpoint is frozen.
    test_loaders = make_split_loaders(data_list, test_idx, batch_size)
    test_metrics, test_details = evaluate_detailed(
        model=model,
        loaders=test_loaders,
        device=device,
        expected_idx=test_idx,
        seed=test_seed,
        collect_details=True,
    )

    val_details.to_csv(
        out_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_details.to_csv(
        out_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "Target": target,
        "Run": run,
        "Architecture": "Official SEFGNN mathematics + six-view/missing-view benchmark adaptation",
        "LearningRate": FROZEN_LR,
        "EpochBudget": epochs,
        "BatchSize": batch_size,
        "NumNeighbors": A.OFFICIAL_NUM_NEIGHBORS,
        "NegativeDrop": A.OFFICIAL_DROP_NEG,
        "WeightDecay": A.OFFICIAL_WEIGHT_DECAY,
        "HiddenChannelsInternal": A.OFFICIAL_HIDDEN_CHANNELS,
        "SES_DS_Weight": A.OFFICIAL_SES_DS_WEIGHT,
        "ModelSeed": model_seed,
        "ValidationEvalSeed": val_seed,
        "TestEvalSeed": test_seed,
        "CheckpointSelection": "Validation AUPRC",
        "MissingViewPolicy": "Vacuous Dirichlet alpha=[1,1]",
        "MissingViewBatchNormPolicy": "Absent-view nodes excluded from training BN statistics",
        "MissingViewPerViewLossNormalization": "Mean over present targets within each source",
        "TrainingRNGIsolatedFromEvaluationRNG": True,
        "TestLoaderCreation": "After checkpoint freeze only",
    }
    (out_dir / "formal_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status = {
        "Status": "PASS",
        "Target": target,
        "Run": run,
        "ModelSeed": model_seed,
        "ValidationEvalSeed": val_seed,
        "TestEvalSeed": test_seed,
        "FrozenLearningRate": FROZEN_LR,
        "BestEpoch": best_epoch,
        "ValidationAUPRC": float(final_val_metrics["AUPRC"]),
        "ValidationAUROC": float(final_val_metrics["AUROC"]),
        "TestN": int(test_metrics["N"]),
        "TestPositive": int(test_metrics["Positive"]),
        "TestAUPRC": float(test_metrics["AUPRC"]),
        "TestAUROC": float(test_metrics["AUROC"]),
        "TestF1_0p5": float(test_metrics["F1_0p5"]),
        "TrainInputNAfterNegativeDrop": int(len(train_idx)),
        "DroppedTrainNegatives": int(ndrop),
        "RuntimeSeconds": float(time.time() - t_run),
        "TestEvaluatedAfterCheckpointFreeze": True,
    }
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nPASS | {target} run_{run:02d} | "
        f"Val AP={status['ValidationAUPRC']:.6f} | "
        f"Test AP={status['TestAUPRC']:.6f} | "
        f"Test AUROC={status['TestAUROC']:.6f}"
    )

    return status


# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------

def write_aggregate(statuses):
    df = pd.DataFrame(statuses)
    if len(df) == 0:
        return

    df = df.sort_values(["Target", "Run"]).reset_index(drop=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        OUT_ROOT / "formal_run_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for target, g in df.groupby("Target"):
        rows.append({
            "Target": target,
            "RunsCompleted": len(g),
            "MeanTestAUPRC": g["TestAUPRC"].mean(),
            "SDTestAUPRC": g["TestAUPRC"].std(ddof=1) if len(g) > 1 else np.nan,
            "MeanTestAUROC": g["TestAUROC"].mean(),
            "SDTestAUROC": g["TestAUROC"].std(ddof=1) if len(g) > 1 else np.nan,
        })

    agg = pd.DataFrame(rows)
    agg.to_csv(
        OUT_ROOT / "formal_target_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCurrent aggregate:")
    print(agg.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_ORDER,
        default=TARGET_ORDER,
    )
    p.add_argument(
        "--runs",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
    )
    p.add_argument("--epochs", type=int, default=FORMAL_EPOCH_BUDGET)
    p.add_argument("--batch-size", type=int, default=A.OFFICIAL_BATCH_SIZE)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("P0-8 | SEFGNN FORMAL DETERMINISTIC SAME-BENCHMARK RUNNER V2")
    print("=" * 100)
    print("Targets           :", args.targets)
    print("Runs              :", args.runs)
    print("Frozen LR         :", FROZEN_LR)
    print("Epoch budget      :", args.epochs)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(args.gpu))
    print("Resume completed  : YES")
    print("Test during train : NO")

    print("\n[1/3] NeighborLoader preflight...")
    A.preflight_neighbor_loader()
    print("  PASS")

    print("\n[2/3] Loading authoritative global alignment + cached six-view graphs...")
    X, global_y, presence, local_to_global = A.load_prepared_alignment()
    data_list, graph_stats = A.make_data_views(
        X, global_y, presence, local_to_global
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    graph_stats.to_csv(
        OUT_ROOT / "graph_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[3/3] Formal runs...")
    statuses = []

    # Include already-completed PASS statuses so aggregate stays complete.
    for target in args.targets:
        for run in args.runs:
            status = run_one_formal(
                target=target,
                run=run,
                data_list=data_list,
                global_y=global_y,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            statuses.append(status)
            write_aggregate(statuses)

    print("\n" + "=" * 100)
    print("REQUESTED FORMAL RUN SET COMPLETE")
    print("Summary:")
    print(OUT_ROOT / "formal_run_summary.csv")
    print(OUT_ROOT / "formal_target_aggregate.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

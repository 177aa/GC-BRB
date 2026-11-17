#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible reconstruction of the GSE126964 paired-expression analysis used in GC-BRB.

Inputs expected in --input-dir (default: script directory):
  - GSE126964_Expr_RSEM.txt.gz
Default reference outputs for self-audit:
  - results/GSE126964_PAIR_VALUES_independent.csv
  - results/GSE126964_PAIRED_SUMMARY_independent.csv

Outputs written to --output-dir (default: recomputed/ beside this script):
  - GSE126964_FROZEN6_PAIR_VALUES_REPRODUCED.csv
  - GSE126964_FROZEN6_PAIRED_SUMMARY_REPRODUCED.csv
  - GSE126964_REPRODUCIBILITY_AUDIT.txt

Reconstructed analysis logic:
  1) Parse gene symbol as text before '|'.
  2) Identify normal samples CKC* N with a matching tumor sample obtained by removing terminal 'N'.
  3) For the frozen six genes, calculate log2(FPKM + 1) in each paired tumor/normal sample.
  4) PairedLog2FC = tumor log2(FPKM+1) - normal log2(FPKM+1).
  5) Summarize medians/mean and pairwise direction counts.
  6) Two-sided exact Wilcoxon signed-rank test per gene.
  7) Benjamini-Hochberg FDR correction across the six Wilcoxon P values.

"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FROZEN6 = ["RPS6KA1", "IL2RB", "NR0B2", "CBFA2T2", "HSPG2", "PRKDC"]
RAW_NAME = "GSE126964_Expr_RSEM.txt.gz"
REF_PAIR_NAME = "GSE126964_FROZEN6_PAIR_VALUES.csv"
REF_SUMMARY_NAME = "GSE126964_FROZEN6_PAIRED_SUMMARY.csv"
OUT_PAIR_NAME = "GSE126964_FROZEN6_PAIR_VALUES_REPRODUCED.csv"
OUT_SUMMARY_NAME = "GSE126964_FROZEN6_PAIRED_SUMMARY_REPRODUCED.csv"
AUDIT_NAME = "GSE126964_REPRODUCIBILITY_AUDIT.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment, preserving original order."""
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank0 in range(n - 1, -1, -1):
        idx = order[rank0]
        rank = rank0 + 1
        running = min(running, p[idx] * n / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip")
    if "Gene" not in df.columns:
        raise ValueError("Raw matrix does not contain a 'Gene' column.")
    df["GeneSymbol"] = df["Gene"].astype(str).str.split("|", n=1).str[0]
    return df


def identify_pairs(columns: Iterable[str]) -> list[tuple[str, str]]:
    cols = list(columns)
    colset = set(cols)
    pairs: list[tuple[str, str]] = []
    for normal in cols:
        if normal.startswith("CKC") and normal.endswith("N"):
            tumor = normal[:-1]
            if tumor in colset:
                pairs.append((tumor, normal))
    if not pairs:
        raise ValueError("No CKC tumor/normal pairs were detected.")
    return pairs


def build_pair_values(raw: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for gene in FROZEN6:
        hit = raw.loc[raw["GeneSymbol"] == gene]
        if len(hit) != 1:
            raise ValueError(f"Expected exactly one raw row for {gene}; found {len(hit)}")
        r = hit.iloc[0]
        for tumor_sample, normal_sample in pairs:
            tumor_fpkm = float(r[tumor_sample])
            normal_fpkm = float(r[normal_sample])
            tumor_log = float(np.log2(tumor_fpkm + 1.0))
            normal_log = float(np.log2(normal_fpkm + 1.0))
            rows.append({
                "Gene": gene,
                "TumorSample": tumor_sample,
                "NormalSample": normal_sample,
                "Tumor_FPKM": tumor_fpkm,
                "Normal_FPKM": normal_fpkm,
                "Tumor_log2FPKM1": tumor_log,
                "Normal_log2FPKM1": normal_log,
                "PairedLog2FC": tumor_log - normal_log,
            })
    return pd.DataFrame(rows)


def build_summary(pair_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_values = []
    for gene in FROZEN6:
        sub = pair_values.loc[pair_values["Gene"] == gene]
        d = sub["PairedLog2FC"].to_numpy(dtype=float)
        test = wilcoxon(
            d,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )
        p_values.append(float(test.pvalue))
        median_fc = float(np.median(d))
        rows.append({
            "Gene": gene,
            "N_pairs": int(len(sub)),
            "MedianTumor_FPKM": float(np.median(sub["Tumor_FPKM"])),
            "MedianNormal_FPKM": float(np.median(sub["Normal_FPKM"])),
            "MedianPairedLog2FC": median_fc,
            "MeanPairedLog2FC": float(np.mean(d)),
            "TumorHigherPairs": int(np.sum(d > 0)),
            "TumorLowerPairs": int(np.sum(d < 0)),
            "Wilcoxon_W": float(test.statistic),
            "Wilcoxon_P": float(test.pvalue),
            "Direction": "Tumor_up" if median_fc > 0 else "Tumor_down",
        })

    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("Direction"), "BH_FDR", bh_adjust(p_values))
    return out


def compare_frames(reference: pd.DataFrame, reproduced: pd.DataFrame, atol: float = 1e-12) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if list(reference.columns) != list(reproduced.columns):
        notes.append(f"Column mismatch: reference={list(reference.columns)}, reproduced={list(reproduced.columns)}")
        return False, notes
    if reference.shape != reproduced.shape:
        notes.append(f"Shape mismatch: reference={reference.shape}, reproduced={reproduced.shape}")
        return False, notes

    ok = True
    for col in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[col]) and pd.api.types.is_numeric_dtype(reproduced[col]):
            a = reference[col].to_numpy(dtype=float)
            b = reproduced[col].to_numpy(dtype=float)
            diff = np.abs(a - b)
            max_diff = float(np.nanmax(diff)) if diff.size else 0.0
            col_ok = bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True))
            notes.append(f"{col}: numeric max_abs_diff={max_diff:.3e}; {'PASS' if col_ok else 'FAIL'}")
        else:
            a = reference[col].astype(str).to_numpy()
            b = reproduced[col].astype(str).to_numpy()
            col_ok = bool(np.array_equal(a, b))
            notes.append(f"{col}: text/order {'PASS' if col_ok else 'FAIL'}")
        ok &= col_ok
    return ok, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "recomputed")
    parser.add_argument("--reference-pairs", type=Path, default=script_dir / "results" / "GSE126964_PAIR_VALUES_independent.csv")
    parser.add_argument("--reference-summary", type=Path, default=script_dir / "results" / "GSE126964_PAIRED_SUMMARY_independent.csv")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = input_dir / RAW_NAME
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw = load_raw(raw_path)
    pairs = identify_pairs(raw.columns)
    pair_values = build_pair_values(raw, pairs)
    summary = build_summary(pair_values)

    out_pair = output_dir / OUT_PAIR_NAME
    out_summary = output_dir / OUT_SUMMARY_NAME
    pair_values.to_csv(out_pair, index=False)
    summary.to_csv(out_summary, index=False)

    audit_lines = [
        "GSE126964 REPRODUCIBILITY AUDIT",
        "=" * 80,
        f"Raw input: {raw_path}",
        f"Raw SHA256: {sha256_file(raw_path)}",
        f"Raw matrix shape: {raw.shape[0]} rows x {raw.shape[1]-1} original columns (+GeneSymbol helper)",
        f"Detected paired samples ({len(pairs)}): " + ", ".join(f"{t}<->{n}" for t, n in pairs),
        "Frozen genes: " + ", ".join(FROZEN6),
        "Transform: log2(FPKM + 1)",
        "Paired effect: tumor log2(FPKM+1) - normal log2(FPKM+1)",
        "Statistics: two-sided exact Wilcoxon signed-rank; BH-FDR across 6 genes",
        "",
    ]

    pair_ref = args.reference_pairs.resolve()
    summary_ref = args.reference_summary.resolve()
    overall_ok = True
    comparisons = 0

    if pair_ref.exists():
        comparisons += 1
        ref = pd.read_csv(pair_ref)
        ok, notes = compare_frames(ref, pair_values)
        overall_ok &= ok
        audit_lines += [f"PAIR_VALUES reference: {pair_ref}", f"PAIR_VALUES SHA256: {sha256_file(pair_ref)}", f"PAIR_VALUES comparison: {'PASS' if ok else 'FAIL'}"] + ["  " + x for x in notes] + [""]
    else:
        audit_lines += ["PAIR_VALUES reference not present; comparison SKIPPED", ""]

    if summary_ref.exists():
        comparisons += 1
        ref = pd.read_csv(summary_ref)
        ok, notes = compare_frames(ref, summary)
        overall_ok &= ok
        audit_lines += [f"PAIRED_SUMMARY reference: {summary_ref}", f"PAIRED_SUMMARY SHA256: {sha256_file(summary_ref)}", f"PAIRED_SUMMARY comparison: {'PASS' if ok else 'FAIL'}"] + ["  " + x for x in notes] + [""]
    else:
        audit_lines += ["PAIRED_SUMMARY reference not present; comparison SKIPPED", ""]

    status = ("PASS" if overall_ok else "FAIL") if comparisons == 2 else ("INCOMPLETE_REFERENCE_CHECK" if overall_ok else "FAIL")
    audit_lines += [
        "OVERALL REPRODUCIBILITY STATUS: " + status,
        "",
        "Note: this is a reconstructed implementation validated against the retained outputs;",
        "it is not represented as the missing historical source file.",
    ]
    audit_path = output_dir / AUDIT_NAME
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_pair}")
    print(f"Wrote: {out_summary}")
    print(f"Wrote: {audit_path}")
    print("OVERALL:", status)
    if not overall_ok:
        raise SystemExit("Reference comparison failed.")


if __name__ == "__main__":
    main()

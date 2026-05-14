#!/usr/bin/env python3
"""Common structural variant overlap analysis for gene-desert anomalies."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.desert_utils import (
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_gnomad_sv,
    load_gnocchi,
    window_interval_overlap,
)

SV_OVERLAP_FLAG_THRESHOLD = 0.10
SV_TYPES = ["INV", "DEL", "DUP", "CPX", "CNV"]


def _plot_fleet(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.scatter(summary["sv_overlap_fraction"], summary["mean_delta_z"], s=16, alpha=0.65, color="tab:blue")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(SV_OVERLAP_FLAG_THRESHOLD, color="tab:red", lw=1, ls="--")
    for desert_id in DESERT_ORDER:
        row = summary[summary["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id, (row["sv_overlap_fraction"].iloc[0], row["mean_delta_z"].iloc[0]), fontsize=8)
    ax.set_xlabel("SV overlap fraction")
    ax.set_ylabel("mean delta_z")
    ax.set_title("Common SV overlap vs adjustment effect")

    ax = axes[0, 1]
    chrom_means = (
        summary.groupby("chrom", observed=True)["sv_overlap_fraction"]
        .mean()
        .sort_values(ascending=False)
    )
    ax.bar(chrom_means.index, chrom_means.values, color="tab:gray", alpha=0.85)
    ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax.set_ylabel("mean SV overlap fraction")
    ax.set_title("Chromosome-level SV overlap burden")

    ax = axes[1, 0]
    vals = summary["sv_overlap_fraction"].dropna()
    ax.hist(vals, bins=40, color="tab:purple", edgecolor="white")
    ax.axvline(SV_OVERLAP_FLAG_THRESHOLD, color="tab:red", lw=1, ls="--")
    ax.set_xlabel("SV overlap fraction")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Distribution of desert SV overlap")

    ax = axes[1, 1]
    type_cols = [c for c in summary.columns if c.startswith("sv_overlap_fraction_")]
    type_means = summary[type_cols].mean().sort_values(ascending=False)
    labels = [c.replace("sv_overlap_fraction_", "") for c in type_means.index]
    ax.bar(labels, type_means.values, color="tab:green", alpha=0.85)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("mean overlap fraction")
    ax.set_title("Average overlap by SV type")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(usecols=["element_id", "chrom", "start", "end", "z_adj", "z_unadj", "delta_z"])
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Labeling all deserts ...")
    deserts_df = load_all_deserts()
    labeled = label_deserts_fleet(gn, deserts_df)
    in_deserts = labeled[labeled["desert"].notna()].copy().rename(columns={"desert": "desert_id"})

    print("Loading and filtering gnomAD SV calls ...")
    sv = load_gnomad_sv(af_min=0.01, size_min=10_000, svtypes=SV_TYPES)
    sv_intervals = sv[["chrom", "start", "end", "svtype"]].copy()

    print("Computing overlap with any qualifying common SV ...")
    in_deserts["sv_overlap_any"] = window_interval_overlap(
        in_deserts[["chrom", "start", "end"]],
        sv_intervals[["chrom", "start", "end"]],
        return_fraction=False,
    ).astype(bool)

    print("Computing overlap by SV type ...")
    for svtype in SV_TYPES:
        sub = sv_intervals[sv_intervals["svtype"] == svtype]
        in_deserts[f"sv_overlap_{svtype}"] = window_interval_overlap(
            in_deserts[["chrom", "start", "end"]],
            sub[["chrom", "start", "end"]],
            return_fraction=False,
        ).astype(bool)

    print("Summarizing per-desert overlap and score shifts ...")
    rows = []
    for desert_id, sub in in_deserts.groupby("desert_id", sort=False):
        ov = sub[sub["sv_overlap_any"]]
        non = sub[~sub["sv_overlap_any"]]
        row = {
            "desert_id": desert_id,
            "n_windows": int(len(sub)),
            "sv_overlap_fraction": float(sub["sv_overlap_any"].mean()),
            "mean_z_adj": float(sub["z_adj"].mean()),
            "mean_z_unadj": float(sub["z_unadj"].mean()),
            "mean_delta_z": float(sub["delta_z"].mean()),
            "mean_z_adj_sv_overlap": float(ov["z_adj"].mean()) if len(ov) else np.nan,
            "mean_z_adj_sv_nonoverlap": float(non["z_adj"].mean()) if len(non) else np.nan,
            "mean_delta_z_sv_overlap": float(ov["delta_z"].mean()) if len(ov) else np.nan,
            "mean_delta_z_sv_nonoverlap": float(non["delta_z"].mean()) if len(non) else np.nan,
        }
        for svtype in SV_TYPES:
            row[f"sv_overlap_fraction_{svtype}"] = float(sub[f"sv_overlap_{svtype}"].mean())
        row["is_sv_flagged"] = bool(row["sv_overlap_fraction"] > SV_OVERLAP_FLAG_THRESHOLD)
        rows.append(row)
    summary = pd.DataFrame(rows).merge(
        deserts_df[["desert_id", "chrom", "start", "end", "panel_label"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "sv_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    flagged = summary[summary["is_sv_flagged"]].sort_values("sv_overlap_fraction", ascending=False).copy()
    out_flagged = os.path.join(RESULTS_DIR, "sv_flagged_deserts.tsv")
    flagged.to_csv(out_flagged, sep="\t", index=False)
    print(f"  wrote {out_flagged}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "sv_fleet_overview.png")
    _plot_fleet(summary, fleet_png)
    print(f"  wrote {fleet_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()

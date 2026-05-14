#!/usr/bin/env python3
"""Lamina-associated domain (LAD) overlap diagnostics across gene deserts."""

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
from scipy.stats import mannwhitneyu

from utils.desert_utils import (
    DESERT_ORDER,
    LADS_DIR,
    RESULTS_DIR,
    compute_constitutive_lads,
    label_deserts_fleet,
    load_all_deserts,
    load_bed_intervals,
    load_gnocchi,
    window_interval_overlap,
)

MIN_CELL_TYPES_CONST_LAD = 10


def _wilcoxon(lad_vals: pd.Series, non_lad_vals: pd.Series) -> float:
    lad_vals = pd.to_numeric(lad_vals, errors="coerce").dropna()
    non_lad_vals = pd.to_numeric(non_lad_vals, errors="coerce").dropna()
    if len(lad_vals) < 5 or len(non_lad_vals) < 5:
        return np.nan
    return float(mannwhitneyu(lad_vals, non_lad_vals, alternative="two-sided").pvalue)


def _plot_exemplar(desert_id: str, sub: pd.DataFrame, out_path: str) -> None:
    if sub.empty:
        return
    work = sub.sort_values("start").copy()
    pos_mb = (work["start"] + work["end"]) / 2 / 1e6

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.plot(pos_mb, work["z_adj"], color="tab:blue", lw=0.8, label="z_adj")
    ax.plot(pos_mb, work["z_unadj"], color="tab:orange", lw=0.7, alpha=0.8, label="z_unadj")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    for _, row in work[work["is_constitutive_lad"]].iterrows():
        ax.axvspan(row["start"] / 1e6, row["end"] / 1e6, color="tab:green", alpha=0.12, lw=0)
    ax.set_ylabel("z score")
    ax.legend(fontsize=8)
    ax.set_title(f"{desert_id}: z profile with constitutive LAD shading")

    ax = axes[1]
    ax.plot(pos_mb, work["lad_occupancy_score"], color="tab:green", lw=0.9, label="LAD occupancy score")
    ax.axhline(MIN_CELL_TYPES_CONST_LAD / 12.0, color="tab:red", ls="--", lw=1, label="constitutive threshold")
    ax.set_ylabel("fraction of LAD tracks")
    ax.set_xlabel("position (Mb)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Cell-type occupancy profile")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet(summary: pd.DataFrame, track_frac: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.scatter(summary["constitutive_lad_fraction"], summary["mean_z_adj"], s=16, alpha=0.65, color="tab:blue")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("constitutive LAD fraction")
    ax.set_ylabel("mean z_adj")
    ax.set_title("LAD fraction vs adjusted score")

    ax = axes[0, 1]
    ax.scatter(summary["constitutive_lad_fraction"], summary["mean_delta_z"], s=16, alpha=0.65, color="tab:purple")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("constitutive LAD fraction")
    ax.set_ylabel("mean delta_z")
    ax.set_title("LAD fraction vs adjustment effect")

    ax = axes[1, 0]
    bins = pd.cut(
        summary["constitutive_lad_fraction"],
        bins=[-1e-9, 0.25, 0.75, 1.0],
        labels=["low (<=0.25)", "mid (0.25-0.75)", "high (>0.75)"],
    )
    grouped = [summary.loc[bins == label, "mean_z_adj"].dropna() for label in bins.cat.categories]
    ax.boxplot(grouped, tick_labels=list(bins.cat.categories), showfliers=False)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("mean z_adj")
    ax.set_title("z_adj stratified by LAD fraction")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    heat = track_frac.set_index("desert_id").reindex(DESERT_ORDER).dropna(how="all", axis=1)
    if not heat.empty:
        im = ax.imshow(heat.to_numpy(), aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([c.replace("_LADs.bed.gz", "") for c in heat.columns], rotation=75, fontsize=7)
        ax.set_title("Exemplar LAD occupancy by cell type")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="overlap fraction")
    else:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Exemplar LAD occupancy by cell type")

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

    print("Computing constitutive LAD intervals ...")
    const_lad, lad_paths = compute_constitutive_lads(
        lad_dir=LADS_DIR,
        min_cell_types=MIN_CELL_TYPES_CONST_LAD,
    )

    print("Computing LAD occupancy per window across all tracks ...")
    occupancy_cols = []
    track_rows = []
    for lad_path in lad_paths:
        track_name = Path(lad_path).name
        intervals = load_bed_intervals(lad_path)
        overlap = window_interval_overlap(
            in_deserts[["chrom", "start", "end"]],
            intervals,
            return_fraction=False,
        ).astype(float)
        col = f"lad_track_{track_name}"
        in_deserts[col] = overlap.values
        occupancy_cols.append(col)

        # Per-desert overlap fraction for this track (for heatmap output).
        track_frac = (
            in_deserts.groupby("desert_id", observed=True)[col]
            .mean()
            .rename("overlap_fraction")
            .reset_index()
        )
        track_frac["track"] = track_name
        track_rows.append(track_frac)

    in_deserts["lad_occupancy_score"] = in_deserts[occupancy_cols].mean(axis=1)
    in_deserts["is_constitutive_lad"] = in_deserts["lad_occupancy_score"] >= (MIN_CELL_TYPES_CONST_LAD / len(lad_paths))

    print("Computing per-desert LAD summary metrics ...")
    rows = []
    for desert_id, sub in in_deserts.groupby("desert_id", sort=False):
        lad_vals = sub.loc[sub["is_constitutive_lad"], "z_adj"]
        non_lad_vals = sub.loc[~sub["is_constitutive_lad"], "z_adj"]
        rows.append(
            {
                "desert_id": desert_id,
                "n_windows": int(len(sub)),
                "lad_occupancy_mean": float(sub["lad_occupancy_score"].mean()),
                "constitutive_lad_fraction": float(sub["is_constitutive_lad"].mean()),
                "mean_z_adj": float(sub["z_adj"].mean()),
                "mean_z_unadj": float(sub["z_unadj"].mean()),
                "mean_delta_z": float(sub["delta_z"].mean()),
                "mean_z_adj_constitutive_lad": float(lad_vals.mean()) if len(lad_vals) else np.nan,
                "mean_z_adj_non_lad": float(non_lad_vals.mean()) if len(non_lad_vals) else np.nan,
                "wilcoxon_p_zadj_lad_vs_nonlad": _wilcoxon(lad_vals, non_lad_vals),
            }
        )
    summary = pd.DataFrame(rows).merge(
        deserts_df[["desert_id", "chrom", "start", "end", "panel_label"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "lad_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    track_frac = pd.concat(track_rows, ignore_index=True)
    track_wide = track_frac.pivot(index="desert_id", columns="track", values="overlap_fraction").reset_index()
    out_track_tsv = os.path.join(RESULTS_DIR, "lad_track_overlap_by_desert.tsv")
    track_wide.to_csv(out_track_tsv, sep="\t", index=False)
    print(f"  wrote {out_track_tsv}")

    print("Generating exemplar plots ...")
    for desert_id in DESERT_ORDER:
        sub = in_deserts[in_deserts["desert_id"] == desert_id].copy()
        if sub.empty:
            continue
        out_png = os.path.join(RESULTS_DIR, f"lad_exemplar_{desert_id}.png")
        _plot_exemplar(desert_id, sub, out_png)
        print(f"  wrote {out_png}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "lad_fleet_overview.png")
    _plot_fleet(summary, track_wide, fleet_png)
    print(f"  wrote {fleet_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()

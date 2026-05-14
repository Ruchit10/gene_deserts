#!/usr/bin/env python3
"""Conservation overlay: phyloP support for z-score spikes in deserts."""

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
    aggregate_block_weighted_mean,
    label_deserts_fleet,
    load_all_deserts,
    load_gnocchi,
    load_phyloP_summary,
)

HIGH_CONSERVED_THRESH = 2.0
MOD_CONSERVED_THRESH = 1.0
SPIKE_THRESH = 2.0


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _plot_exemplar(desert_id: str, sub: pd.DataFrame, out_path: str) -> None:
    if sub.empty:
        return
    work = sub.sort_values("start").copy()
    pos_mb = (work["start"] + work["end"]) / 2 / 1e6
    spike_mask = work["z_adj"] > SPIKE_THRESH
    cons_mask = work["mean_phyloP"] > HIGH_CONSERVED_THRESH

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.plot(pos_mb, work["z_adj"], lw=0.8, color="tab:blue", label="z_adj")
    ax.plot(pos_mb, work["z_unadj"], lw=0.7, color="tab:orange", alpha=0.8, label="z_unadj")
    ax.scatter(pos_mb[spike_mask], work.loc[spike_mask, "z_adj"], s=18, color="tab:red", label="spikes (z>2)")
    ax.scatter(
        pos_mb[spike_mask & cons_mask],
        work.loc[spike_mask & cons_mask, "z_adj"],
        s=20,
        color="tab:green",
        label="spikes + high phyloP",
    )
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("z score")
    ax.legend(fontsize=8)
    ax.set_title(f"{desert_id}: z profile with spike/conservation overlap")

    ax = axes[1]
    ax.plot(pos_mb, work["mean_phyloP"], lw=0.85, color="tab:purple", label="mean phyloP")
    ax.axhline(MOD_CONSERVED_THRESH, color="gray", lw=0.8, ls="--", label="moderate threshold")
    ax.axhline(HIGH_CONSERVED_THRESH, color="tab:red", lw=0.8, ls="--", label="high threshold")
    ax.set_ylabel("phyloP")
    ax.set_xlabel("position (Mb)")
    ax.legend(fontsize=8)
    ax.set_title("Conservation profile")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    plot_df = summary.dropna(subset=["mean_phyloP", "mean_z_adj"])
    ax.scatter(plot_df["mean_phyloP"], plot_df["mean_z_adj"], s=16, alpha=0.65, color="tab:blue")
    for desert_id in DESERT_ORDER:
        row = plot_df[plot_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id, (row["mean_phyloP"].iloc[0], row["mean_z_adj"].iloc[0]), fontsize=8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("mean phyloP")
    ax.set_ylabel("mean z_adj")
    ax.set_title("Conservation vs adjusted score")

    ax = axes[0, 1]
    vals = summary["spike_explained_fraction"].dropna()
    ax.hist(vals, bins=30, color="tab:green", edgecolor="white")
    ax.set_xlabel("fraction of spikes with high phyloP")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Spike attribution by conservation")

    ax = axes[1, 0]
    plot_df = summary.dropna(subset=["corr_phyloP_zadj", "mean_phyloP"])
    ax.scatter(plot_df["mean_phyloP"], plot_df["corr_phyloP_zadj"], s=16, alpha=0.65, color="tab:purple")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("mean phyloP")
    ax.set_ylabel("corr(phyloP, z_adj)")
    ax.set_title("Within-desert conservation-score coupling")

    ax = axes[1, 1]
    top = summary.nlargest(12, "spike_explained_fraction")
    ax.barh(top["desert_id"], top["spike_explained_fraction"], color="tab:orange", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("spike explained fraction")
    ax.set_title("Top deserts with conservation-backed spikes")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(usecols=["element_id", "chrom", "start", "end", "z_adj", "z_unadj", "delta_z"])
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Loading phyloP summary blocks ...")
    phy = load_phyloP_summary()

    print("Aggregating phyloP to 1kb windows ...")
    gn["mean_phyloP"] = aggregate_block_weighted_mean(
        windows_df=gn[["chrom", "start", "end"]],
        blocks_df=phy[["chrom", "start", "end", "mean_phyloP"]],
        value_col="mean_phyloP",
    )

    print("Labeling all deserts ...")
    deserts_df = load_all_deserts()
    labeled = label_deserts_fleet(gn, deserts_df)
    in_deserts = labeled[labeled["desert"].notna()].copy().rename(columns={"desert": "desert_id"})
    in_deserts["is_spike"] = in_deserts["z_adj"] > SPIKE_THRESH
    in_deserts["is_highly_conserved"] = in_deserts["mean_phyloP"] > HIGH_CONSERVED_THRESH
    in_deserts["is_moderately_conserved"] = in_deserts["mean_phyloP"] > MOD_CONSERVED_THRESH

    print("Computing per-desert conservation metrics ...")
    rows = []
    for desert_id, sub in in_deserts.groupby("desert_id", sort=False):
        n_spikes = int(sub["is_spike"].sum())
        n_spikes_cons = int((sub["is_spike"] & sub["is_highly_conserved"]).sum())
        rows.append(
            {
                "desert_id": desert_id,
                "n_windows": int(len(sub)),
                "mean_phyloP": float(sub["mean_phyloP"].mean()),
                "frac_highly_conserved": float(sub["is_highly_conserved"].mean()),
                "frac_moderately_conserved": float(sub["is_moderately_conserved"].mean()),
                "mean_z_adj": float(sub["z_adj"].mean()),
                "mean_z_unadj": float(sub["z_unadj"].mean()),
                "mean_delta_z": float(sub["delta_z"].mean()),
                "n_spike_windows": n_spikes,
                "n_spike_highly_conserved": n_spikes_cons,
                "spike_explained_fraction": (n_spikes_cons / n_spikes) if n_spikes else np.nan,
                "corr_phyloP_zadj": _safe_corr(sub["mean_phyloP"], sub["z_adj"]),
            }
        )
    summary = pd.DataFrame(rows).merge(
        deserts_df[["desert_id", "chrom", "start", "end", "panel_label"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "conserved_elements_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    print("Generating exemplar plots ...")
    for desert_id in DESERT_ORDER:
        sub = in_deserts[in_deserts["desert_id"] == desert_id].copy()
        if sub.empty:
            continue
        out_png = os.path.join(RESULTS_DIR, f"conserved_elements_exemplar_{desert_id}.png")
        _plot_exemplar(desert_id, sub, out_png)
        print(f"  wrote {out_png}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "conserved_elements_fleet_overview.png")
    _plot_fleet(summary, fleet_png)
    print(f"  wrote {fleet_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()

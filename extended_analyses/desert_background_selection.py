#!/usr/bin/env python3
"""Background-selection proxy analysis via desert edge-distance gradients."""

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
    load_features,
    load_gnocchi,
)

EDGE_THRESHOLD = 0.2
CENTER_THRESHOLD = 0.8


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10:
        return np.nan
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def _interaction_coef(sub: pd.DataFrame) -> float:
    req = sub[["dist_to_edge_kb", "recomb_male_1M", "z_adj"]].dropna()
    if len(req) < 20:
        return np.nan
    x1 = req["dist_to_edge_kb"].to_numpy(dtype=float)
    x2 = req["recomb_male_1M"].to_numpy(dtype=float)
    y = req["z_adj"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x1), x1, x2, x1 * x2])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[3])


def _classify_gradient(r: float) -> str:
    if not np.isfinite(r):
        return "unclassified"
    if r > 0.2:
        return "strong_bgs_like_gradient"
    if abs(r) < 0.1:
        return "flat"
    if r < -0.2:
        return "inverted"
    return "weak_gradient"


def _plot_exemplar(desert_id: str, sub: pd.DataFrame, out_path: str) -> None:
    if sub.empty:
        return
    work = sub.sort_values("start").copy()
    pos_mb = (work["start"] + work["end"]) / 2 / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.scatter(work["relative_position"], work["z_adj"], s=8, alpha=0.3, color="tab:blue", label="z_adj")
    ax.scatter(work["relative_position"], work["z_unadj"], s=8, alpha=0.25, color="tab:orange", label="z_unadj")
    ax.axvline(EDGE_THRESHOLD, color="gray", lw=1, ls="--")
    ax.axvline(CENTER_THRESHOLD, color="gray", lw=1, ls="--")
    ax.set_xlabel("relative position (0=edge, 1=center)")
    ax.set_ylabel("z score")
    ax.set_title("Edge-center gradient")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sc = ax.scatter(pos_mb, work["z_adj"], c=work["relative_position"], s=9, cmap="viridis", alpha=0.8)
    ax.set_xlabel("position (Mb)")
    ax.set_ylabel("z_adj")
    ax.set_title("Spatial z_adj colored by edge-distance")
    plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, label="relative position")

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 11)
    work["rel_bin"] = pd.cut(work["relative_position"], bins=bins, include_lowest=True)
    curve = work.groupby("rel_bin", observed=True).agg(z_adj=("z_adj", "mean"), z_unadj=("z_unadj", "mean"))
    mids = np.array([iv.mid for iv in curve.index])
    ax.plot(mids, curve["z_adj"], "o-", color="tab:blue", label="z_adj")
    ax.plot(mids, curve["z_unadj"], "o-", color="tab:orange", label="z_unadj")
    ax.set_xlabel("relative position bin midpoint")
    ax.set_ylabel("mean z")
    ax.set_title("Binned edge-center profiles")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(work["recomb_male_1M"], work["z_adj"], s=8, alpha=0.3, color="tab:green")
    ax.set_xlabel("recomb_male_1M")
    ax.set_ylabel("z_adj")
    ax.set_title("Recombination interaction view")

    fig.suptitle(f"{desert_id} BGS proxy diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    plot_df = summary.dropna(subset=["desert_length_kb", "corr_zadj_dist_to_edge"])
    ax.scatter(plot_df["desert_length_kb"], plot_df["corr_zadj_dist_to_edge"], s=16, alpha=0.65, color="tab:blue")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("desert length (kb)")
    ax.set_ylabel("corr(z_adj, dist_to_edge)")
    ax.set_title("Gradient strength vs desert size")

    ax = axes[0, 1]
    plot_df = summary.dropna(subset=["mean_recomb_male_1M", "corr_zadj_dist_to_edge"])
    ax.scatter(plot_df["mean_recomb_male_1M"], plot_df["corr_zadj_dist_to_edge"], s=16, alpha=0.65, color="tab:green")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("mean recomb_male_1M")
    ax.set_ylabel("corr(z_adj, dist_to_edge)")
    ax.set_title("Gradient strength vs recombination")

    ax = axes[1, 0]
    ax.hist(summary["edge_center_delta_zadj"].dropna(), bins=35, color="tab:purple", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("edge-center mean z_adj difference")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Edge-center z_adj differences")

    ax = axes[1, 1]
    counts = summary["gradient_class"].value_counts()
    ax.bar(counts.index, counts.values, color="tab:gray", alpha=0.9)
    ax.set_ylabel("Number of deserts")
    ax.set_title("Gradient-class distribution")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(usecols=["element_id", "chrom", "start", "end", "z_adj", "z_unadj"])
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Loading recombination feature(s) ...")
    feats = load_features(element_ids=gn["element_id"].tolist())[["element_id", "recomb_male_1M"]]
    df = gn.merge(feats, on="element_id", how="left")

    print("Labeling all deserts ...")
    deserts_df = load_all_deserts().rename(columns={"start": "desert_start", "end": "desert_end"})
    labeled = label_deserts_fleet(df, deserts_df.rename(columns={"desert_start": "start", "desert_end": "end"}))
    in_deserts = labeled[labeled["desert"].notna()].copy().rename(columns={"desert": "desert_id"})
    in_deserts = in_deserts.merge(
        deserts_df[["desert_id", "chrom", "desert_start", "desert_end"]],
        left_on=["desert_id", "chrom"],
        right_on=["desert_id", "chrom"],
        how="left",
    )

    center = (in_deserts["start"] + in_deserts["end"]) / 2.0
    left_dist = center - in_deserts["desert_start"]
    right_dist = in_deserts["desert_end"] - center
    in_deserts["dist_to_edge_kb"] = np.minimum(left_dist, right_dist) / 1000.0
    half_len = (in_deserts["desert_end"] - in_deserts["desert_start"]).clip(lower=1) / 2.0
    in_deserts["relative_position"] = (np.minimum(left_dist, right_dist) / half_len).clip(0, 1)

    print("Computing per-desert gradient metrics ...")
    rows = []
    for desert_id, sub in in_deserts.groupby("desert_id", sort=False):
        edge = sub[sub["relative_position"] <= EDGE_THRESHOLD]["z_adj"]
        center_vals = sub[sub["relative_position"] >= CENTER_THRESHOLD]["z_adj"]
        row = {
            "desert_id": desert_id,
            "n_windows": int(len(sub)),
            "desert_length_kb": float((sub["desert_end"].iloc[0] - sub["desert_start"].iloc[0]) / 1000.0),
            "mean_z_adj": float(sub["z_adj"].mean()),
            "mean_z_unadj": float(sub["z_unadj"].mean()),
            "mean_recomb_male_1M": float(sub["recomb_male_1M"].mean()),
            "corr_zadj_dist_to_edge": _safe_corr(sub["z_adj"], sub["dist_to_edge_kb"]),
            "corr_zunadj_dist_to_edge": _safe_corr(sub["z_unadj"], sub["dist_to_edge_kb"]),
            "slope_zadj_dist_to_edge": _linear_slope(
                sub["dist_to_edge_kb"].to_numpy(dtype=float),
                sub["z_adj"].to_numpy(dtype=float),
            ),
            "edge_mean_zadj": float(edge.mean()) if len(edge) else np.nan,
            "center_mean_zadj": float(center_vals.mean()) if len(center_vals) else np.nan,
            "edge_center_delta_zadj": float(edge.mean() - center_vals.mean())
            if len(edge) and len(center_vals)
            else np.nan,
            "interaction_coef_dist_x_recomb": _interaction_coef(sub),
        }
        row["gradient_class"] = _classify_gradient(row["corr_zadj_dist_to_edge"])
        rows.append(row)
    summary = pd.DataFrame(rows).merge(
        deserts_df[["desert_id", "chrom", "desert_start", "desert_end"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "bgs_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    print("Generating exemplar panels ...")
    for desert_id in DESERT_ORDER:
        sub = in_deserts[in_deserts["desert_id"] == desert_id].copy()
        if sub.empty:
            continue
        out_png = os.path.join(RESULTS_DIR, f"bgs_exemplar_{desert_id}.png")
        _plot_exemplar(desert_id, sub, out_png)
        print(f"  wrote {out_png}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "bgs_fleet_overview.png")
    _plot_fleet(summary, fleet_png)
    print(f"  wrote {fleet_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()

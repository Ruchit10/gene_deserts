#!/usr/bin/env python3
"""Replication timing overlay for gene-desert Gnocchi anomalies."""

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
    bin_values_to_windows,
    label_deserts_fleet,
    load_all_deserts,
    load_features,
    load_gnocchi,
    load_replication_timing,
)

GC_COL = "GC_content_1k"


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _partial_corr_rt_zadj_gc(sub: pd.DataFrame) -> float:
    req = sub[["rt_value", "z_adj", GC_COL]].dropna()
    if len(req) < 20:
        return np.nan
    gc = req[GC_COL].to_numpy(dtype=float)
    rt = req["rt_value"].to_numpy(dtype=float)
    z = req["z_adj"].to_numpy(dtype=float)

    design = np.column_stack([np.ones_like(gc), gc])
    beta_rt, *_ = np.linalg.lstsq(design, rt, rcond=None)
    beta_z, *_ = np.linalg.lstsq(design, z, rcond=None)
    rt_resid = rt - design @ beta_rt
    z_resid = z - design @ beta_z
    if np.allclose(np.std(rt_resid), 0.0) or np.allclose(np.std(z_resid), 0.0):
        return np.nan
    return float(np.corrcoef(rt_resid, z_resid)[0, 1])


def _r2_increment(sub: pd.DataFrame) -> tuple[float, float, float]:
    req = sub[["z_adj", "rt_value", GC_COL]].dropna()
    if len(req) < 20:
        return np.nan, np.nan, np.nan
    y = req["z_adj"].to_numpy(dtype=float)
    gc = req[GC_COL].to_numpy(dtype=float)
    rt = req["rt_value"].to_numpy(dtype=float)

    d1 = np.column_stack([np.ones_like(gc), gc])
    d2 = np.column_stack([np.ones_like(gc), gc, rt])
    b1, *_ = np.linalg.lstsq(d1, y, rcond=None)
    b2, *_ = np.linalg.lstsq(d2, y, rcond=None)
    r2_gc = _r2(y, d1 @ b1)
    r2_gc_rt = _r2(y, d2 @ b2)
    return r2_gc, r2_gc_rt, r2_gc_rt - r2_gc


def _plot_exemplar(desert_id: str, sub: pd.DataFrame, out_path: str) -> None:
    if sub.empty:
        return
    work = sub.sort_values("start").copy()
    pos_mb = (work["start"] + work["end"]) / 2 / 1e6

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    ax = axes[0]
    ax.plot(pos_mb, work["z_adj"], lw=0.8, color="tab:blue", label="z_adj")
    ax.plot(pos_mb, work["z_unadj"], lw=0.7, color="tab:orange", label="z_unadj", alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.6)
    ax.set_ylabel("z-score")
    ax.legend(fontsize=8, loc="upper left")

    ax_rt = ax.twinx()
    ax_rt.plot(pos_mb, work["rt_value"], lw=0.8, color="tab:green", alpha=0.8, label="replication_timing")
    ax_rt.set_ylabel("Replication timing", color="tab:green")
    ax_rt.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title(f"{desert_id}: spatial z and replication timing")

    ax = axes[1]
    plot_df = work[["rt_value", "z_adj", GC_COL]].dropna()
    ax.scatter(plot_df["rt_value"], plot_df["z_adj"], s=10, alpha=0.35, color="tab:blue")
    if len(plot_df) > 10:
        coef = np.polyfit(plot_df["rt_value"], plot_df["z_adj"], 1)
        xs = np.linspace(plot_df["rt_value"].min(), plot_df["rt_value"].max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color="tab:red", lw=1.2, label="linear fit")
    ax.set_xlabel("Replication timing")
    ax.set_ylabel("z_adj")
    ax.set_title("RT vs z_adj")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    plot_df = summary.dropna(subset=["mean_rt", "mean_delta_z"])
    ax.scatter(plot_df["mean_rt"], plot_df["mean_delta_z"], s=16, alpha=0.65, color="tab:gray")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for desert_id in DESERT_ORDER:
        row = plot_df[plot_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id, (row["mean_rt"].iloc[0], row["mean_delta_z"].iloc[0]), fontsize=8)
    ax.set_xlabel("mean replication timing")
    ax.set_ylabel("mean delta_z")
    ax.set_title("RT vs adjustment effect")

    ax = axes[0, 1]
    plot_df = summary.dropna(subset=["mean_rt", "mean_z_adj"])
    ax.scatter(plot_df["mean_rt"], plot_df["mean_z_adj"], s=16, alpha=0.65, color="tab:blue")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("mean replication timing")
    ax.set_ylabel("mean z_adj")
    ax.set_title("RT vs adjusted scores")

    ax = axes[1, 0]
    vals = summary["partial_corr_rt_zadj_given_gc"].dropna()
    ax.hist(vals, bins=35, color="tab:green", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("partial corr(RT, z_adj | GC)")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Partial-correlation distribution")

    ax = axes[1, 1]
    top = summary.dropna(subset=["r2_increment_rt_over_gc"]).nlargest(12, "r2_increment_rt_over_gc")
    ax.barh(top["desert_id"], top["r2_increment_rt_over_gc"], color="tab:purple", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("R2 increment from RT")
    ax.set_title("Top RT-informative deserts")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(usecols=["element_id", "chrom", "start", "end", "z_adj", "z_unadj", "delta_z"])
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Loading replication timing points ...")
    rt_points = load_replication_timing()
    gn["rt_value"] = bin_values_to_windows(
        points_df=rt_points,
        windows_df=gn[["chrom", "start", "end"]],
        value_col="rt_value",
    )

    print("Loading GC feature ...")
    feats = load_features(element_ids=gn["element_id"].tolist())[["element_id", GC_COL]]
    gn = gn.merge(feats, on="element_id", how="left")

    print("Labeling all deserts ...")
    deserts_df = load_all_deserts()
    labeled = label_deserts_fleet(gn, deserts_df)
    in_deserts = labeled[labeled["desert"].notna()].copy().rename(columns={"desert": "desert_id"})

    print("Computing per-desert replication timing metrics ...")
    rows = []
    for desert_id, sub in in_deserts.groupby("desert_id", sort=False):
        r2_gc, r2_gc_rt, r2_inc = _r2_increment(sub)
        rows.append(
            {
                "desert_id": desert_id,
                "n_windows": int(len(sub)),
                "mean_rt": float(sub["rt_value"].mean()),
                "mean_z_adj": float(sub["z_adj"].mean()),
                "mean_z_unadj": float(sub["z_unadj"].mean()),
                "mean_delta_z": float(sub["delta_z"].mean()),
                "corr_rt_z_adj": _safe_corr(sub["rt_value"], sub["z_adj"]),
                "corr_rt_z_unadj": _safe_corr(sub["rt_value"], sub["z_unadj"]),
                "corr_rt_delta_z": _safe_corr(sub["rt_value"], sub["delta_z"]),
                "partial_corr_rt_zadj_given_gc": _partial_corr_rt_zadj_gc(sub),
                "r2_gc_only": r2_gc,
                "r2_gc_plus_rt": r2_gc_rt,
                "r2_increment_rt_over_gc": r2_inc,
            }
        )
    summary = pd.DataFrame(rows).merge(
        deserts_df[["desert_id", "chrom", "start", "end", "panel_label"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "replication_timing_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    print("Generating exemplar panels ...")
    for desert_id in DESERT_ORDER:
        sub = in_deserts[in_deserts["desert_id"] == desert_id].copy()
        if sub.empty:
            continue
        out_png = os.path.join(RESULTS_DIR, f"replication_timing_exemplar_{desert_id}.png")
        _plot_exemplar(desert_id, sub, out_png)
        print(f"  wrote {out_png}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "replication_timing_fleet_overview.png")
    _plot_fleet(summary, fleet_png)
    print(f"  wrote {fleet_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()

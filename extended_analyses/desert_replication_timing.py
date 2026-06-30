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
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from utils.desert_utils import (
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_features,
    load_gnocchi,
    load_replication_timing_feature,
)

GC_COL = "GC_content_1k"
TOP_RT_N = 10  # number of top RT-informative deserts to plot spatially


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


def _plot_top_rt_spatial(
    summary: pd.DataFrame,
    in_deserts: pd.DataFrame,
    n_top: int,
    out_path: str,
) -> None:
    """Compound spatial figure for the top N deserts ranked by RT R² increment.

    Each desert occupies one row of 3 stacked panels sharing the x-axis:
      col 0 – z_adj (blue) and z_unadj (orange)
      col 1 – RT_BG02 replication timing
      col 2 – delta_z (z_unadj − z_adj)
    Per-desert stats (ΔR², r(RT,z_adj), partial corr) are annotated on col 2.
    Exemplar deserts are labelled with a star in the row ylabel.
    """
    top = (
        summary.dropna(subset=["r2_increment_rt_over_gc"])
        .nlargest(n_top, "r2_increment_rt_over_gc")
        .reset_index(drop=True)
    )
    if top.empty:
        return

    n = len(top)
    _mb = ticker.FuncFormatter(lambda x, _: f"{x:.1f}")

    fig, axes = plt.subplots(n, 3, figsize=(16, 3.2 * n), squeeze=False)

    for row_idx, row_data in top.iterrows():
        desert_id   = str(row_data["desert_id"])
        r2_inc      = float(row_data["r2_increment_rt_over_gc"])
        corr_rt_z   = row_data.get("corr_rt_z_adj", np.nan)
        partial_r   = row_data.get("partial_corr_rt_zadj_given_gc", np.nan)
        is_exemplar = desert_id in set(DESERT_ORDER)

        sub = in_deserts[in_deserts["desert_id"] == desert_id].sort_values("start")
        pos_mb = (sub["start"] + sub["end"]) / 2 / 1e6

        row_label = f"{'★ ' if is_exemplar else ''}{desert_id}"

        # ── col 0: z_adj / z_unadj ───────────────────────────────────────
        ax = axes[row_idx, 0]
        if not sub.empty:
            ax.plot(pos_mb, sub["z_adj"],   lw=0.85, color="tab:blue",   label="z_adj")
            ax.plot(pos_mb, sub["z_unadj"], lw=0.75, color="tab:orange", alpha=0.75, label="z_unadj")
            ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel(row_label, fontsize=9, labelpad=4)
        if row_idx == 0:
            ax.set_title("z_adj / z_unadj", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
        ax.xaxis.set_major_formatter(_mb)
        ax.tick_params(axis="x", labelsize=7)
        if row_idx < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

        # ── col 1: RT profile ─────────────────────────────────────────────
        ax = axes[row_idx, 1]
        if not sub.empty and sub["rt_value"].notna().any():
            ax.plot(pos_mb, sub["rt_value"], lw=0.85, color="tab:green")
        ax.set_ylabel("RT_BG02", fontsize=8, color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")
        if row_idx == 0:
            ax.set_title("Replication timing (RT_BG02)", fontsize=10)
        ax.xaxis.set_major_formatter(_mb)
        ax.tick_params(axis="x", labelsize=7)
        if row_idx < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

        # ── col 2: delta_z + stats annotation ────────────────────────────
        ax = axes[row_idx, 2]
        if not sub.empty and "delta_z" in sub.columns:
            ax.plot(pos_mb, sub["delta_z"], lw=0.85, color="tab:purple")
            ax.axhline(0, color="grey", lw=0.5, ls="--")
        if row_idx == 0:
            ax.set_title("delta_z  (z_unadj − z_adj)", fontsize=10)
        stats_txt = (
            f"ΔR² = {r2_inc:.3f}\n"
            f"r(RT, z_adj) = {corr_rt_z:+.2f}\n"
            f"partial r | GC = {partial_r:+.2f}"
        )
        ax.text(
            0.02, 0.97, stats_txt,
            transform=ax.transAxes, fontsize=7,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )
        ax.xaxis.set_major_formatter(_mb)
        ax.tick_params(axis="x", labelsize=7)
        if row_idx < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    # Shared x-label on the bottom row only
    for col in range(3):
        axes[n - 1, col].set_xlabel("Position (Mb)", fontsize=9)

    fig.suptitle(
        f"Top {n} RT-informative deserts — spatial profiles  "
        f"(ranked by R² increment from adding RT to GC model)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(usecols=["element_id", "chrom", "start", "end", "z_adj", "z_unadj", "delta_z"])
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Loading per-window replication timing (RT_BG02) ...")
    rt_feature = load_replication_timing_feature(element_ids=gn["element_id"].tolist())
    gn = gn.merge(rt_feature, on="element_id", how="left")
    print(f"  RT joined; missing fraction: {gn['rt_value'].isna().mean():.3f}")

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

    print(f"Generating spatial profiles for top {TOP_RT_N} RT-informative deserts ...")
    top_spatial_png = os.path.join(RESULTS_DIR, "replication_timing_top_rt_spatial.png")
    _plot_top_rt_spatial(summary, in_deserts, TOP_RT_N, top_spatial_png)
    print(f"  wrote {top_spatial_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()

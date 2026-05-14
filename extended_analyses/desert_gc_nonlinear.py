#!/usr/bin/env python3
"""Test non-linear GC effects on adjusted Gnocchi scores in gene deserts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from utils.desert_utils import (
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_features,
    load_gnocchi,
)

MIN_POINTS = 30
NONLINEARITY_DELTA_R2 = 0.05
GC_COL = "GC_content_1k"


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or np.allclose(np.nanvar(y), 0.0):
        return np.nan
    ss_res = np.nansum((y - pred) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    return pred, _r2(y, pred)


def _fit_piecewise(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    design = np.column_stack(
        [
            np.ones_like(x),
            x,
            np.clip(x - q1, 0, None),
            np.clip(x - q2, 0, None),
            np.clip(x - q3, 0, None),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    return pred, _r2(y, pred)


def _fit_spline(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    uniq = np.unique(xs)
    if len(uniq) < 6:
        pred = np.full_like(y, np.nan, dtype=float)
        return pred, np.nan

    # Use progressively stronger smoothing; if fitpack warns, retry with larger s.
    base_var = float(np.nanvar(ys))
    base_s = max(len(xs) * base_var * 0.2, 1e-6)
    pred_sorted = None
    for mult in (1.0, 2.0, 5.0, 10.0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                spline = UnivariateSpline(xs, ys, s=base_s * mult)
                pred_sorted = spline(xs)
            break
        except Warning:
            pred_sorted = None
            continue
    if pred_sorted is None:
        # Conservative fallback: return linear fit prediction without raising warnings.
        pred_fallback, _ = _fit_linear(x, y)
        return pred_fallback, _r2(y, pred_fallback)

    pred = np.empty_like(pred_sorted)
    pred[order] = pred_sorted
    return pred, _r2(y, pred)


def _safe_nanstd(arr: np.ndarray) -> float:
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return np.nan
    return float(np.nanstd(arr, ddof=1))


def _desert_metrics(sub: pd.DataFrame) -> dict[str, Any]:
    valid = sub[[GC_COL, "z_adj"]].dropna()
    n = len(valid)
    row: dict[str, Any] = {"n_windows": n}
    if n < MIN_POINTS:
        row.update(
            r2_linear=np.nan,
            r2_spline=np.nan,
            r2_piecewise=np.nan,
            delta_r2_spline_linear=np.nan,
            resid_sd_linear=np.nan,
            resid_sd_spline=np.nan,
            resid_sd_piecewise=np.nan,
            is_nonlinear=False,
        )
        return row

    x = valid[GC_COL].to_numpy(dtype=float)
    y = valid["z_adj"].to_numpy(dtype=float)
    pred_lin, r2_lin = _fit_linear(x, y)
    pred_spl, r2_spl = _fit_spline(x, y)
    pred_pw, r2_pw = _fit_piecewise(x, y)

    row.update(
        r2_linear=r2_lin,
        r2_spline=r2_spl,
        r2_piecewise=r2_pw,
        delta_r2_spline_linear=r2_spl - r2_lin if np.isfinite(r2_spl) else np.nan,
        resid_sd_linear=_safe_nanstd(y - pred_lin),
        resid_sd_spline=_safe_nanstd(y - pred_spl)
        if np.isfinite(r2_spl)
        else np.nan,
        resid_sd_piecewise=_safe_nanstd(y - pred_pw),
        is_nonlinear=bool(np.isfinite(r2_spl) and (r2_spl - r2_lin > NONLINEARITY_DELTA_R2)),
    )
    return row


def _gc_decile_curve(df: pd.DataFrame) -> pd.DataFrame:
    work = df[[GC_COL, "z_adj"]].dropna().copy()
    work["gc_decile"] = pd.qcut(work[GC_COL], 10, labels=False, duplicates="drop")
    out = work.groupby("gc_decile", observed=True).agg(gc_mid=(GC_COL, "mean"), z_mean=("z_adj", "mean"))
    return out.reset_index(drop=True)


def _plot_exemplar(name: str, sub: pd.DataFrame, genome_curve: pd.DataFrame, out_path: str) -> None:
    valid = sub[[GC_COL, "z_adj", "start", "end"]].dropna().copy()
    if len(valid) < MIN_POINTS:
        return
    x = valid[GC_COL].to_numpy(dtype=float)
    y = valid["z_adj"].to_numpy(dtype=float)
    pred_lin, _ = _fit_linear(x, y)
    pred_spl, _ = _fit_spline(x, y)
    pred_pw, _ = _fit_piecewise(x, y)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.scatter(x, y, s=8, alpha=0.25, color="tab:gray", label="windows")
    order = np.argsort(x)
    ax.plot(x[order], pred_lin[order], lw=1.2, color="tab:orange", label="linear")
    ax.plot(x[order], pred_spl[order], lw=1.2, color="tab:blue", label="spline")
    ax.plot(x[order], pred_pw[order], lw=1.0, color="tab:green", label="piecewise")
    ax.set_xlabel("GC_content_1k")
    ax.set_ylabel("z_adj")
    ax.set_title(f"{name}: GC vs z_adj")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(x, y - pred_lin, s=8, alpha=0.25, color="tab:orange", label="linear residual")
    ax.scatter(x, y - pred_spl, s=8, alpha=0.25, color="tab:blue", label="spline residual")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("GC_content_1k")
    ax.set_ylabel("Residual")
    ax.set_title(f"{name}: residual comparison")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    desert_curve = _gc_decile_curve(valid.rename(columns={"z_adj": "z_adj"}))
    ax.plot(genome_curve["gc_mid"], genome_curve["z_mean"], "o-", lw=1.3, label="genome", color="tab:gray")
    ax.plot(desert_curve["gc_mid"], desert_curve["z_mean"], "o-", lw=1.3, label=name, color="tab:blue")
    ax.set_xlabel("GC_content_1k (decile mean)")
    ax.set_ylabel("mean z_adj")
    ax.set_title("GC-decile trend: genome vs desert")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(x, bins=30, alpha=0.7, color="tab:blue", density=True, label=f"{name} GC")
    ax.hist(sub[GC_COL].dropna(), bins=30, alpha=0.35, color="tab:gray", density=True, label=f"{name} windows")
    ax.set_xlabel("GC_content_1k")
    ax.set_ylabel("density")
    ax.set_title("GC distribution (desert)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet(summary: pd.DataFrame, genome_curve: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    vals = summary["delta_r2_spline_linear"].dropna()
    ax.hist(vals, bins=40, color="tab:blue", alpha=0.8, edgecolor="white")
    ax.axvline(NONLINEARITY_DELTA_R2, color="tab:red", ls="--", lw=1.1, label="flag threshold")
    ax.set_xlabel("delta R2 (spline - linear)")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Non-linearity gain across deserts")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    plot_df = summary.dropna(subset=["r2_linear", "r2_spline"])
    ax.scatter(plot_df["r2_linear"], plot_df["r2_spline"], s=16, alpha=0.6, color="tab:gray")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(-0.05, min(1.0, plot_df["r2_linear"].max() + 0.05))
    ax.set_ylim(-0.05, min(1.0, plot_df["r2_spline"].max() + 0.05))
    ax.set_xlabel("R2 linear")
    ax.set_ylabel("R2 spline")
    ax.set_title("Linear vs spline fit quality")

    ax = axes[1, 0]
    ax.plot(genome_curve["gc_mid"], genome_curve["z_mean"], "o-", color="tab:gray", lw=1.5)
    ax.set_xlabel("GC_content_1k (decile mean)")
    ax.set_ylabel("Genome mean z_adj")
    ax.set_title("Genome-wide GC-z trend")

    ax = axes[1, 1]
    top = summary.dropna(subset=["delta_r2_spline_linear"]).nlargest(10, "delta_r2_spline_linear")
    ax.barh(top["desert_id"], top["delta_r2_spline_linear"], color="tab:purple", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("delta R2 (spline - linear)")
    ax.set_title("Top 10 non-linear deserts")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(usecols=["element_id", "chrom", "start", "end", "z_adj"])
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Loading GC feature values ...")
    feats = load_features(element_ids=gn["element_id"].tolist())[["element_id", GC_COL]]
    df = gn.merge(feats, on="element_id", how="left")

    print("Labeling all deserts ...")
    deserts_df = load_all_deserts()
    labeled = label_deserts_fleet(df, deserts_df)
    in_deserts = labeled[labeled["desert"].notna()].copy()
    in_deserts = in_deserts.rename(columns={"desert": "desert_id"})

    print("Computing per-desert non-linearity metrics ...")
    rows = []
    for desert_id, sub in in_deserts.groupby("desert_id", sort=False):
        row = {"desert_id": desert_id}
        row.update(_desert_metrics(sub))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.merge(
        deserts_df[["desert_id", "chrom", "start", "end", "panel_label"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "gc_nonlinear_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    genome_curve = _gc_decile_curve(labeled)

    print("Generating exemplar figures ...")
    for name in DESERT_ORDER:
        sub = in_deserts[in_deserts["desert_id"] == name].copy()
        if sub.empty:
            continue
        out_path = os.path.join(RESULTS_DIR, f"gc_nonlinear_exemplar_{name}.png")
        _plot_exemplar(name, sub, genome_curve, out_path)
        print(f"  wrote {out_path}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "gc_nonlinear_fleet_overview.png")
    _plot_fleet(summary, genome_curve, fleet_png)
    print(f"  wrote {fleet_png}")

    n_flagged = int(summary["is_nonlinear"].sum())
    print(f"\nDone. Non-linearity flagged deserts: {n_flagged}/{len(summary)}")


if __name__ == "__main__":
    main()

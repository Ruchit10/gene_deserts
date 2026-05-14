#!/usr/bin/env python3
"""GC-content analysis at Gnocchi score extrema in gene deserts.

This script expands GC diagnostics beyond point-scatter correlations by adding:
  1) spatial GC profiles (line plots, not point clouds),
  2) GC deviation from each desert baseline,
  3) local-vs-regional GC contrast (GC_1k - GC_100k),
  4) GC-detrended residual z profiles,
  5) lagged cross-correlation of z vs GC, and
  6) fleet-level volcano and pooled-density summaries.

Primary question:
  Do local dips/spikes in z_adj align with unusual GC patterns in a way that
  suggests the GC-adjustment term is driving the anomaly?

Outputs (all in results/):
  gc_extrema_desert_flags.tsv
  gc_extrema_comparison.tsv
  gc_extrema_exemplar_{name}.png
  gc_extrema_fleet_overview.png
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

from utils.desert_utils import (
    DESERTS,
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_features,
    load_gnocchi,
)

# ── Tunable thresholds ────────────────────────────────────────────────────────
ABS_DIP_THRESH = -2.0  # absolute z_adj floor for "dip"
ABS_SPIKE_THRESH = 2.0  # absolute z_adj ceiling for "spike"
REL_N_SD = 1.5  # within-desert SD multiplier for relative classification
MIN_EXTREME_WIN = 5  # min windows in a class to consider a desert "notable"
MIN_DESERT_WINDOWS = 10  # deserts with fewer windows are skipped
GC_SCALES = ["1k", "10k", "100k", "1M"]
PRIMARY_GC_COL = "GC_content_1k"
CONTEXT_GC_COL = "GC_content_100k"
GC_BIN_COUNT = 10
MAX_CCF_LAG_KB = 50


# ── Window classification ─────────────────────────────────────────────────────


def _classify(sub: pd.DataFrame) -> pd.Series:
    """Assign each row a class label using absolute + relative thresholds.

    Priority: absolute thresholds first; relative thresholds extend the
    classification for within-desert outliers that don't cross the genome-wide
    absolute cutoff (e.g. a desert that is uniformly at z = -1 may still have
    relative dips worth investigating).
    """
    z = sub["z_adj"].values
    n = len(z)
    classes = np.full(n, "normal", dtype=object)

    abs_dip = z < ABS_DIP_THRESH
    abs_spike = z > ABS_SPIKE_THRESH
    classes[abs_dip] = "dip"
    classes[abs_spike] = "spike"

    # Relative: only applied where not already classified
    normal_mask = classes == "normal"
    if normal_mask.sum() >= MIN_DESERT_WINDOWS:
        mu = np.nanmean(z[normal_mask])
        sd = np.nanstd(z[normal_mask], ddof=1)
        if sd > 0.1:
            rel_dip = normal_mask & (z < mu - REL_N_SD * sd)
            rel_spike = normal_mask & (z > mu + REL_N_SD * sd)
            classes[rel_dip] = "rel_dip"
            classes[rel_spike] = "rel_spike"

    return pd.Series(classes, index=sub.index, name="window_class")


def _add_gc_derived_columns(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    if PRIMARY_GC_COL in out.columns:
        out["gc_dev"] = out[PRIMARY_GC_COL] - out[PRIMARY_GC_COL].mean()
    else:
        out["gc_dev"] = np.nan

    if PRIMARY_GC_COL in out.columns and CONTEXT_GC_COL in out.columns:
        out["gc_local_contrast"] = out[PRIMARY_GC_COL] - out[CONTEXT_GC_COL]
    else:
        out["gc_local_contrast"] = np.nan
    return out


def _cross_correlation_peak(
    z: np.ndarray, gc: np.ndarray, max_lag: int
) -> tuple[float, float]:
    """Return (lag_kb, correlation) at max absolute cross-correlation."""
    best_lag = np.nan
    best_corr = np.nan
    best_abs = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = z[-lag:]
            b = gc[:lag]
        elif lag > 0:
            a = z[:-lag]
            b = gc[lag:]
        else:
            a = z
            b = gc
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 8:
            continue
        r = np.corrcoef(a[mask], b[mask])[0, 1]
        if np.isfinite(r) and abs(r) > best_abs:
            best_abs = abs(r)
            best_lag = float(lag)
            best_corr = float(r)
    return best_lag, best_corr


def _detrend_z_by_gc(sub: pd.DataFrame) -> pd.Series:
    if PRIMARY_GC_COL not in sub.columns:
        return pd.Series(np.nan, index=sub.index)
    combined = sub[[PRIMARY_GC_COL, "z_adj"]].dropna()
    if len(combined) < 8:
        return pd.Series(np.nan, index=sub.index)
    coef = np.polyfit(combined[PRIMARY_GC_COL], combined["z_adj"], 1)
    pred = np.polyval(coef, sub[PRIMARY_GC_COL].values)
    return pd.Series(sub["z_adj"].values - pred, index=sub.index)


# ── Per-desert statistics ─────────────────────────────────────────────────────


def _desert_stats(sub: pd.DataFrame) -> dict[str, Any]:
    """Compute per-desert flagging and GC stats for a classified window set."""
    z = sub["z_adj"].dropna()
    gc = (
        sub[PRIMARY_GC_COL].dropna()
        if PRIMARY_GC_COL in sub.columns
        else pd.Series([], dtype=float)
    )
    n = len(sub)

    n_dip = int((sub["window_class"] == "dip").sum())
    n_spike = int((sub["window_class"] == "spike").sum())
    n_rel_dip = int((sub["window_class"] == "rel_dip").sum())
    n_rel_spike = int((sub["window_class"] == "rel_spike").sum())
    within_sd = float(z.std(ddof=1)) if len(z) > 1 else np.nan
    z_range = float(z.max() - z.min()) if len(z) > 1 else np.nan

    row: dict[str, Any] = {
        "n_windows": n,
        "n_dip": n_dip,
        "n_spike": n_spike,
        "n_rel_dip": n_rel_dip,
        "n_rel_spike": n_rel_spike,
        "dip_frac": n_dip / max(n, 1),
        "spike_frac": n_spike / max(n, 1),
        "within_sd": within_sd,
        "z_range": z_range,
        "has_dips": (n_dip + n_rel_dip) >= MIN_EXTREME_WIN,
        "has_spikes": (n_spike + n_rel_spike) >= MIN_EXTREME_WIN,
        "heterogeneous": (not np.isnan(within_sd))
        and (within_sd > 1.0)
        and ((n_dip + n_rel_dip + n_spike + n_rel_spike) >= MIN_EXTREME_WIN),
    }

    # GC correlation with z_adj
    if len(gc) >= 20 and PRIMARY_GC_COL in sub.columns:
        combined = sub[["z_adj", PRIMARY_GC_COL]].dropna()
        if len(combined) >= 20:
            r_p, p_p = pearsonr(combined["z_adj"], combined[PRIMARY_GC_COL])
            r_s, p_s = spearmanr(combined["z_adj"], combined[PRIMARY_GC_COL])
            row["gc_r_pearson"] = float(r_p)
            row["gc_p_pearson"] = float(p_p)
            row["gc_r_spearman"] = float(r_s)
            row["gc_p_spearman"] = float(p_s)
        else:
            row.update(
                gc_r_pearson=np.nan,
                gc_p_pearson=np.nan,
                gc_r_spearman=np.nan,
                gc_p_spearman=np.nan,
            )
    else:
        row.update(
            gc_r_pearson=np.nan,
            gc_p_pearson=np.nan,
            gc_r_spearman=np.nan,
            gc_p_spearman=np.nan,
        )

    # Cross-correlation peak (lag-sensitive) and detrended residual SD
    lag, corr = np.nan, np.nan
    resid_sd = np.nan
    if PRIMARY_GC_COL in sub.columns:
        sorted_sub = sub.sort_values("start")
        lag, corr = _cross_correlation_peak(
            sorted_sub["z_adj"].to_numpy(dtype=float),
            sorted_sub[PRIMARY_GC_COL].to_numpy(dtype=float),
            MAX_CCF_LAG_KB,
        )
        resid = _detrend_z_by_gc(sorted_sub)
        if resid.notna().sum() > 1:
            resid_sd = float(resid.std(ddof=1))
    row["gc_ccf_peak_lag_kb"] = lag
    row["gc_ccf_peak_r"] = corr
    row["z_resid_sd_after_gc"] = resid_sd

    # GC comparison between classes
    if PRIMARY_GC_COL in sub.columns:
        for cls in ["dip", "rel_dip", "spike", "rel_spike", "normal"]:
            vals = sub.loc[sub["window_class"] == cls, PRIMARY_GC_COL].dropna()
            row[f"mean_gc_{cls}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"n_gc_{cls}"] = len(vals)

        # Effect size: (mean_gc_dip - mean_gc_normal) / sd_gc_normal
        all_dip = sub.loc[
            sub["window_class"].isin(["dip", "rel_dip"]), PRIMARY_GC_COL
        ].dropna()
        all_spike = sub.loc[
            sub["window_class"].isin(["spike", "rel_spike"]), PRIMARY_GC_COL
        ].dropna()
        normal_gc = sub.loc[sub["window_class"] == "normal", PRIMARY_GC_COL].dropna()
        normal_sd = float(normal_gc.std(ddof=1)) if len(normal_gc) > 1 else np.nan

        if len(all_dip) >= 3 and len(normal_gc) >= 3:
            stat, p = ttest_ind(all_dip, normal_gc, equal_var=False)
            row["gc_effect_dip"] = float(
                (all_dip.mean() - normal_gc.mean()) / max(normal_sd or 1e-9, 1e-9)
            )
            row["gc_ttest_p_dip"] = float(p)
        else:
            row.update(gc_effect_dip=np.nan, gc_ttest_p_dip=np.nan)

        if len(all_spike) >= 3 and len(normal_gc) >= 3:
            stat, p = ttest_ind(all_spike, normal_gc, equal_var=False)
            row["gc_effect_spike"] = float(
                (all_spike.mean() - normal_gc.mean()) / max(normal_sd or 1e-9, 1e-9)
            )
            row["gc_ttest_p_spike"] = float(p)
        else:
            row.update(gc_effect_spike=np.nan, gc_ttest_p_spike=np.nan)

    return row


# ── Plots ─────────────────────────────────────────────────────────────────────

_CLASS_COLOR = {
    "dip": "tab:red",
    "rel_dip": "tab:orange",
    "spike": "tab:blue",
    "rel_spike": "tab:cyan",
    "normal": "tab:gray",
}
_CLASS_ALPHA = {"dip": 0.18, "rel_dip": 0.10, "spike": 0.18, "rel_spike": 0.10}


def _mb(x: float, _: Any) -> str:
    return f"{x/1e6:.2f}"


def _save_exemplar_profile(name: str, sub: pd.DataFrame, out_path: str) -> None:
    """2x2 exemplar figure with spatial and binned GC diagnostics."""
    chrom, ds, de, note = DESERTS[name]
    sub = _add_gc_derived_columns(sub.sort_values("start").copy())
    sub["z_resid_gc"] = _detrend_z_by_gc(sub)
    pos = (sub["start"] + sub["end"]) / 2
    has_gc = PRIMARY_GC_COL in sub.columns

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax_overlay = axes[0, 0]
    ax_scales = axes[1, 0]
    ax_resid = axes[0, 1]
    ax_binned = axes[1, 1]

    # Panel 1: z + GC deviation overlay
    ax_overlay.plot(
        pos,
        sub["z_adj"],
        lw=0.85,
        color="tab:blue",
        alpha=0.95,
        label="z_adj",
        zorder=3,
    )
    ax_overlay.plot(
        pos,
        sub["z_unadj"],
        lw=0.75,
        color="tab:orange",
        alpha=0.75,
        label="z_unadj",
        zorder=3,
    )
    ax_overlay.axhline(0, color="grey", lw=0.5, ls="--", zorder=1)
    ax_overlay.axhline(ABS_DIP_THRESH, color="tab:red", lw=0.6, ls=":", alpha=0.7)
    ax_overlay.axhline(ABS_SPIKE_THRESH, color="tab:blue", lw=0.6, ls=":", alpha=0.7)
    for cls in ["dip", "rel_dip", "spike", "rel_spike"]:
        for _, w in sub[sub["window_class"] == cls].iterrows():
            ax_overlay.axvspan(
                w["start"],
                w["end"],
                alpha=_CLASS_ALPHA.get(cls, 0.1),
                color=_CLASS_COLOR[cls],
                lw=0,
                zorder=2,
            )
    ax_overlay.set_ylabel("z score")
    ax_overlay.set_title(f"{name}  {chrom}:{ds:,}-{de:,}  ({note})")
    ax_overlay.legend(fontsize=8, loc="upper left")
    ax_overlay.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))

    ax_gc = ax_overlay.twinx()
    if has_gc:
        ax_gc.plot(
            pos,
            sub["gc_dev"],
            color="tab:green",
            lw=1.0,
            alpha=0.8,
            label="GC_1k - desert mean",
        )
        ax_gc.fill_between(pos, 0, sub["gc_dev"], color="tab:green", alpha=0.18)
        ax_gc.set_ylabel("GC deviation (1k)", color="tab:green")
        ax_gc.tick_params(axis="y", labelcolor="tab:green")

    # Panel 2: multi-scale GC and local contrast
    if has_gc:
        for scale, ls in zip(GC_SCALES, ["-", "--", ":", "-."]):
            col = f"GC_content_{scale}"
            if col in sub.columns:
                ax_scales.plot(pos, sub[col], lw=0.9, ls=ls, label=col)
        if "gc_local_contrast" in sub.columns:
            ax_scales2 = ax_scales.twinx()
            ax_scales2.plot(
                pos,
                sub["gc_local_contrast"],
                color="black",
                lw=0.8,
                alpha=0.7,
                label="GC_1k-GC_100k",
            )
            ax_scales2.axhline(0, color="black", lw=0.4, ls="--", alpha=0.6)
            ax_scales2.set_ylabel("local contrast", color="black")
            ax_scales2.tick_params(axis="y", labelcolor="black")
        ax_scales.set_ylabel("GC content")
        ax_scales.legend(fontsize=7, loc="upper left")
    ax_scales.set_xlabel(f"{chrom} position (Mb)")
    ax_scales.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))
    ax_scales.set_title("Spatial GC profiles (multi-scale)")

    # Panel 3: delta_z and GC-detrended residual
    dz = sub["delta_z"] if "delta_z" in sub.columns else (sub["z_unadj"] - sub["z_adj"])
    ax_resid.plot(pos, dz, lw=0.9, color="tab:purple", alpha=0.85, label="delta_z")
    ax_resid.plot(
        pos,
        sub["z_resid_gc"],
        lw=0.8,
        color="tab:red",
        alpha=0.75,
        label="z_adj residual after GC",
    )
    ax_resid.axhline(0, color="grey", lw=0.5, ls="--")
    ax_resid.set_ylabel("delta / residual")
    ax_resid.set_xlabel(f"{chrom} position (Mb)")
    ax_resid.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))
    ax_resid.legend(fontsize=8)

    lag, corr = _cross_correlation_peak(
        sub["z_adj"].to_numpy(dtype=float),
        sub[PRIMARY_GC_COL].to_numpy(dtype=float) if has_gc else np.array([]),
        MAX_CCF_LAG_KB,
    )
    if np.isfinite(corr):
        ax_resid.set_title(
            f"delta_z + GC-detrended residual  (CCF peak lag={lag:+.0f} kb, r={corr:+.2f})"
        )
    else:
        ax_resid.set_title("delta_z + GC-detrended residual")

    # Panel 4: GC-quantile binned z
    if has_gc:
        combined = sub[[PRIMARY_GC_COL, "z_adj"]].dropna()
        if len(combined) >= GC_BIN_COUNT:
            combined = combined.copy()
            combined["gc_bin"] = pd.qcut(
                combined[PRIMARY_GC_COL], q=GC_BIN_COUNT, duplicates="drop"
            )
            binned = (
                combined.groupby("gc_bin", observed=True)["z_adj"]
                .agg(["mean", "sem", "count"])
                .reset_index()
            )
            mids = [float(iv.mid) for iv in binned["gc_bin"]]
            ax_binned.errorbar(
                mids,
                binned["mean"],
                yerr=binned["sem"],
                fmt="o-",
                lw=1.0,
                ms=4,
                color="tab:blue",
            )
            ax_binned.set_xlabel("GC_content_1k (quantile-bin midpoint)")
            ax_binned.set_ylabel("mean z_adj +/- SEM")
            ax_binned.axhline(0, color="grey", lw=0.5, ls="--")
            ax_binned.set_title("Binned GC-vs-z relationship")
        else:
            ax_binned.text(
                0.5,
                0.5,
                "Not enough points for quantile bins",
                ha="center",
                va="center",
                transform=ax_binned.transAxes,
            )
    else:
        ax_binned.text(
            0.5,
            0.5,
            "GC features not available",
            ha="center",
            va="center",
            transform=ax_binned.transAxes,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_fleet_overview(flags: pd.DataFrame, out_path: str) -> None:
    """4-panel fleet overview: dip/spike counts, variability, volcano, pooled density."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    exemplars = set(DESERT_ORDER)

    # ── Panel A: scatter n_dip vs n_spike (size = n_windows) ─────────────
    ax = axes[0, 0]
    het = flags["heterogeneous"]
    size = 20 + 80 * (flags["n_windows"] - flags["n_windows"].min()) / max(
        1, flags["n_windows"].max() - flags["n_windows"].min()
    )
    ax.scatter(
        flags.loc[~het, "n_dip"],
        flags.loc[~het, "n_spike"],
        s=size[~het],
        alpha=0.5,
        color="tab:gray",
        label="not heterogeneous",
    )
    ax.scatter(
        flags.loc[het, "n_dip"],
        flags.loc[het, "n_spike"],
        s=size[het],
        alpha=0.7,
        color="tab:purple",
        label="heterogeneous",
    )
    for _, row in flags[flags["desert_id"].isin(exemplars)].iterrows():
        ax.annotate(
            row["desert_id"],
            (row["n_dip"], row["n_spike"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("n_dip windows (z < −2 or relative)")
    ax.set_ylabel("n_spike windows (z > +2 or relative)")
    ax.set_title("Dip vs spike window counts per desert")
    ax.legend(fontsize=8)

    # ── Panel B: within-desert SD distribution ────────────────────────────
    ax = axes[0, 1]
    ax.hist(
        flags["within_sd"].dropna(),
        bins=40,
        color="tab:gray",
        edgecolor="white",
        alpha=0.8,
    )
    ax.axvline(
        1.0, color="tab:purple", lw=1, ls="--", label=f"heterogeneity cut (SD=1.0)"
    )
    for _, row in flags[flags["desert_id"].isin(exemplars)].iterrows():
        if pd.notna(row["within_sd"]):
            ax.axvline(row["within_sd"], color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("Within-desert SD of z_adj")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Internal z_adj variability across all deserts")
    ax.legend(fontsize=8)

    # ── Panel C: GC-effect volcano plot ───────────────────────────────────
    ax = axes[1, 0]
    volc = flags[flags["gc_effect_dip"].notna()].copy()
    pvals = volc["gc_ttest_p_dip"].fillna(1.0).clip(lower=1e-300, upper=1.0)
    volc_y = -np.log10(pvals)
    ax.scatter(volc["gc_effect_dip"], volc_y, s=18, alpha=0.55, color="tab:gray")
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(-np.log10(0.05), color="black", lw=0.8, ls="--", alpha=0.6)
    for _, row in volc[volc["desert_id"].isin(exemplars)].iterrows():
        yy = (
            -np.log10(max(row["gc_ttest_p_dip"], 1e-300))
            if pd.notna(row["gc_ttest_p_dip"])
            else 0
        )
        ax.annotate(
            row["desert_id"],
            (row["gc_effect_dip"], yy),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("GC effect at dips (Cohen d)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Dip GC effect: magnitude vs confidence")

    # ── Panel D: pooled GC densities by window class ──────────────────────
    ax = axes[1, 1]
    pooled_rows = []
    for cls_name, src_cols in {
        "dip_like": ["mean_gc_dip", "mean_gc_rel_dip"],
        "spike_like": ["mean_gc_spike", "mean_gc_rel_spike"],
        "normal": ["mean_gc_normal"],
    }.items():
        vals = flags[src_cols].to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            pooled_rows.append((cls_name, vals))
    if pooled_rows:
        all_vals = np.concatenate([vals for _, vals in pooled_rows])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)
        for cls_name, vals in pooled_rows:
            hist, edges = np.histogram(vals, bins=bins, density=True)
            mids = (edges[:-1] + edges[1:]) / 2
            color = {
                "dip_like": "tab:red",
                "spike_like": "tab:blue",
                "normal": "tab:gray",
            }[cls_name]
            ax.plot(
                mids, hist, lw=1.5, color=color, label=f"{cls_name} (n={len(vals)})"
            )
        ax.set_xlabel("GC_content_1k")
        ax.set_ylabel("density")
        ax.set_title("Pooled per-desert GC means by class")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No pooled GC data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.suptitle("Fleet-wide GC extrema analysis (all deserts)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading Gnocchi z-scores ...")
    gn = load_gnocchi(
        usecols=["chrom", "start", "end", "element_id", "z_adj", "z_unadj"]
    )
    gn["delta_z"] = gn["z_unadj"] - gn["z_adj"]
    print(f"  {len(gn):,} windows")

    print("Loading all desert coordinates ...")
    deserts_df = load_all_deserts()

    # ── Label all windows with fleet desert IDs ───────────────────────────
    print("Labeling windows (fleet) ...")
    labeled_fleet = label_deserts_fleet(gn, deserts_df)
    desert_windows = labeled_fleet[labeled_fleet["desert"].notna()].copy()
    print(
        f"  {len(desert_windows):,} windows in deserts "
        f"({desert_windows['desert'].nunique()} unique deserts)"
    )

    # ── Load GC features for desert windows only ──────────────────────────
    print("Loading GC features for desert windows ...")
    gc_cols = ["element_id"] + [f"GC_content_{s}" for s in GC_SCALES]
    feats = load_features(element_ids=desert_windows["element_id"].tolist())
    feats = feats[[c for c in gc_cols if c in feats.columns]]
    desert_windows = desert_windows.merge(feats, on="element_id", how="left")
    print(
        f"  GC joined; missing fraction: "
        f"{desert_windows[PRIMARY_GC_COL].isna().mean():.3f}"
    )

    # ── Classify windows and build per-desert flag table ──────────────────
    print("Classifying windows and computing per-desert stats ...")
    class_parts: list[pd.DataFrame] = []
    flag_rows: list[dict[str, Any]] = []
    for desert_id, sub in desert_windows.groupby("desert", sort=False):
        if len(sub) < MIN_DESERT_WINDOWS:
            continue
        sub = _add_gc_derived_columns(sub.copy())
        cls = _classify(sub)
        sub["window_class"] = cls
        sub["z_resid_gc"] = _detrend_z_by_gc(sub)
        class_parts.append(
            sub[["window_class", "gc_dev", "gc_local_contrast", "z_resid_gc"]]
        )
        row = {"desert_id": desert_id}
        row.update(_desert_stats(sub))
        flag_rows.append(row)

    if class_parts:
        desert_windows = desert_windows.join(pd.concat(class_parts))
    else:
        desert_windows["window_class"] = "normal"

    flags = pd.DataFrame(flag_rows)
    flags = flags.merge(
        deserts_df[["desert_id", "chrom", "start", "end"]].rename(
            columns={"start": "desert_start", "end": "desert_end"}
        ),
        on="desert_id",
        how="left",
    )

    out_flags = os.path.join(RESULTS_DIR, "gc_extrema_desert_flags.tsv")
    flags.to_csv(out_flags, sep="\t", index=False)
    print(f"  wrote {out_flags}")

    # ── GC comparison table (notable deserts only) ────────────────────────
    notable = flags[flags["has_dips"] | flags["has_spikes"]].copy()
    comp_cols = [
        c
        for c in flags.columns
        if c.startswith(
            (
                "mean_gc_",
                "n_gc_",
                "gc_effect",
                "gc_ttest",
                "gc_r_",
                "gc_p_",
                "gc_ccf_",
                "z_resid_",
            )
        )
    ]
    out_comp = os.path.join(RESULTS_DIR, "gc_extrema_comparison.tsv")
    notable[["desert_id"] + comp_cols].to_csv(out_comp, sep="\t", index=False)
    print(f"  wrote {out_comp} ({len(notable)} notable deserts)")

    # ── Exemplar enhanced profiles ────────────────────────────────────────
    print("\nGenerating exemplar profiles ...")
    for name in DESERT_ORDER:
        sub = desert_windows[desert_windows["desert"] == name].copy()
        if sub.empty:
            print(f"  {name}: no windows found, skipping")
            continue
        out_path = os.path.join(RESULTS_DIR, f"gc_extrema_exemplar_{name}.png")
        _save_exemplar_profile(name, sub, out_path)
        stats = _desert_stats(sub)
        n_ext = (
            stats["n_dip"]
            + stats["n_spike"]
            + stats["n_rel_dip"]
            + stats["n_rel_spike"]
        )
        r_gc = stats.get("gc_r_pearson", np.nan)
        eff = stats.get("gc_effect_dip", np.nan)
        lag = stats.get("gc_ccf_peak_lag_kb", np.nan)
        print(
            f"  {name}: {n_ext} extreme windows, "
            f"r(z~GC_1k)={r_gc:+.3f}, GC effect at dips={eff:+.3f} SD, "
            f"CCF lag={lag:+.0f} kb"
        )
        print(f"    wrote {out_path}")

    # ── Fleet overview figure ─────────────────────────────────────────────
    print("\nGenerating fleet overview figure ...")
    _save_fleet_overview(
        flags, os.path.join(RESULTS_DIR, "gc_extrema_fleet_overview.png")
    )
    print("  wrote gc_extrema_fleet_overview.png")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n=== Fleet summary ({len(flags)} deserts) ===")
    print(f"  Has notable dips  : {flags['has_dips'].sum()}")
    print(f"  Has notable spikes: {flags['has_spikes'].sum()}")
    print(f"  Heterogeneous     : {flags['heterogeneous'].sum()}")
    print(f"  Top |r(z~GC_1k)| deserts:")
    top_r = (
        flags.dropna(subset=["gc_r_pearson"])
        .assign(abs_r=lambda x: x["gc_r_pearson"].abs())
        .nlargest(10, "abs_r")
    )
    for _, row in top_r.iterrows():
        marker = " <-- exemplar" if row["desert_id"] in DESERT_ORDER else ""
        print(
            f"    {row['desert_id']:8s}  r={row['gc_r_pearson']:+.3f}  "
            f"effect_dip={row.get('gc_effect_dip', np.nan):+.3f} SD  "
            f"ccf_lag={row.get('gc_ccf_peak_lag_kb', np.nan):+.0f} kb{marker}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

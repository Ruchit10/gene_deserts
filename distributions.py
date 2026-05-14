#!/usr/bin/env python3
"""Analysis B: Distribution comparison of z_adj vs z_unadj per desert.

Produces per-desert summary statistics (mean/median/SD/skew plus tail
fractions) for both adjusted and unadjusted z, side-by-side Obs-vs-Exp
scatter plots and O/E box plots, and an attribution table recording
whether r "fixes", "breaks" or is "neutral" on each desert's anomaly.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

from utils.desert_utils import (
    DESERTS,
    DESERT_ORDER,
    RESULTS_DIR,
    desert_palette,
    label_deserts,
    load_gnocchi,
)


def z_stats(z: pd.Series) -> dict:
    z = z.dropna().values
    if len(z) == 0:
        return {k: np.nan for k in ("n", "mean", "median", "sd", "skew",
                                     "frac_abs_gt2", "frac_gt4")}
    return {
        "n": len(z),
        "mean": float(np.mean(z)),
        "median": float(np.median(z)),
        "sd": float(np.std(z, ddof=1)),
        "skew": float(skew(z)),
        "frac_abs_gt2": float(np.mean(np.abs(z) > 2)),
        "frac_gt4": float(np.mean(z > 4)),
    }


def attribute_r(adj: dict, unadj: dict, anomaly_key: str) -> str:
    """Classify the role of r for a desert's headline anomaly.

    anomaly_key is one of:
      'mean_high'  - anomaly was high mean/median z
      'mean_low'   - anomaly was low mean/median z
      'gc_corr'    - handled in Analysis D (returned 'see_D')
      'acf'        - handled in Analysis C (returned 'see_C')
    """
    if anomaly_key in ("gc_corr",):
        return "see_D"
    if anomaly_key in ("acf",):
        return "see_C"
    adj_mean = adj["mean"]
    unadj_mean = unadj["mean"]
    if anomaly_key == "mean_high":
        if abs(unadj_mean) < 0.5 * abs(adj_mean):
            return "fixes"
        if abs(unadj_mean) > 1.2 * abs(adj_mean):
            return "breaks"
        if np.sign(unadj_mean) != np.sign(adj_mean):
            return "flips"
        return "neutral"
    if anomaly_key == "mean_low":
        if abs(unadj_mean) < 0.5 * abs(adj_mean):
            return "fixes"
        if abs(unadj_mean) > 1.2 * abs(adj_mean):
            return "breaks"
        if np.sign(unadj_mean) != np.sign(adj_mean):
            return "flips"
        return "neutral"
    return "unknown"


ANOMALY_KEYS = {
    "GD513": "mean_high",
    "GD198": "mean_low",
    "GD588": "gc_corr",
    "GD158": "acf",
    "GD167": "acf",
}


def main() -> None:
    print("Loading merged Gnocchi table ...")
    df = load_gnocchi()
    df = label_deserts(df)
    df["oe_adj"] = df["observed"] / df["expected"]
    print(f"  loaded {len(df):,} windows")

    # ── Per-desert summary stats ───────────────────────────────────────────
    rows = []
    genome_adj = z_stats(df["z_adj"])
    genome_unadj = z_stats(df["z_unadj"])
    rows.append({"scope": "genome", "desert": None, "note": "",
                 **{f"adj_{k}": v for k, v in genome_adj.items()},
                 **{f"unadj_{k}": v for k, v in genome_unadj.items()}})
    for name in DESERT_ORDER:
        note = DESERTS[name][3]
        sub = df[df["desert"] == name]
        adj = z_stats(sub["z_adj"])
        unadj = z_stats(sub["z_unadj"])
        rows.append({"scope": "desert", "desert": name, "note": note,
                     **{f"adj_{k}": v for k, v in adj.items()},
                     **{f"unadj_{k}": v for k, v in unadj.items()}})

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "analysis_b_z_distribution_stats.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"  wrote {summary_path}")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:,.3f}".format):
        print(summary.to_string(index=False))

    # ── Attribution table ─────────────────────────────────────────────────
    attr_rows = []
    for name in DESERT_ORDER:
        sub = df[df["desert"] == name]
        adj = z_stats(sub["z_adj"])
        unadj = z_stats(sub["z_unadj"])
        key = ANOMALY_KEYS[name]
        verdict = attribute_r(adj, unadj, key)
        attr_rows.append({
            "desert": name,
            "anomaly": DESERTS[name][3],
            "anomaly_class": key,
            "mean_z_adj": adj["mean"],
            "mean_z_unadj": unadj["mean"],
            "median_z_adj": adj["median"],
            "median_z_unadj": unadj["median"],
            "delta_mean_z": unadj["mean"] - adj["mean"],
            "verdict": verdict,
        })
    attr = pd.DataFrame(attr_rows)
    attr_path = os.path.join(RESULTS_DIR, "desert_anomaly_attribution.tsv")
    attr.to_csv(attr_path, sep="\t", index=False)
    print(f"  wrote {attr_path}")
    print(attr.to_string(index=False))

    palette = desert_palette()

    # ── Obs vs Exp scatter (adjusted and unadjusted) ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    bg = df.sample(min(50_000, len(df)), random_state=0)
    for ax, xcol, title in [
        (axes[0], "expected",       "Obs vs Exp (adjusted)"),
        (axes[1], "expected_unadj", "Obs vs Exp (unadjusted)"),
    ]:
        ax.scatter(bg[xcol], bg["observed"], s=2, alpha=0.1, color="lightgray",
                   label="genome (50k sample)")
        for name in DESERT_ORDER:
            sub = df[df["desert"] == name]
            ax.scatter(sub[xcol], sub["observed"], s=6, alpha=0.6,
                       color=palette[name], label=name)
        lo = 0
        hi = float(np.nanpercentile(np.concatenate(
            [bg[xcol].values, bg["observed"].values]), 99.9))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xcol)
        ax.set_ylabel("observed")
        ax.set_title(title)
        ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_b_obs_vs_exp_scatter.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # ── O/E box plots (adjusted vs unadjusted) per desert vs genome ──────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, col, title in [
        (axes[0], "oe_adj",   "O/E (adjusted)"),
        (axes[1], "oe_unadj", "O/E (unadjusted)"),
    ]:
        data = [df[col].dropna().values] + [
            df.loc[df["desert"] == name, col].dropna().values
            for name in DESERT_ORDER
        ]
        labels = ["genome"] + DESERT_ORDER
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"],
                                 ["lightgray"] + [palette[n] for n in DESERT_ORDER]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(1.0, color="k", ls="--", lw=0.7)
        ax.set_title(title)
        ax.set_ylabel("O/E ratio")
        ax.set_ylim(0, 2.5)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_b_oe_boxplots.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # ── z-score kde / hist overlays per desert with genome background ─────
    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    bins = np.linspace(-8, 8, 81)
    for ax, name in zip(axes, DESERT_ORDER):
        sub = df[df["desert"] == name]
        ax.hist(df["z_adj"].dropna(), bins=bins, density=True, histtype="step",
                color="black", alpha=0.4, label="genome z_adj")
        ax.hist(sub["z_adj"], bins=bins, density=True, alpha=0.55,
                color=palette[name], label=f"{name} z_adj")
        ax.hist(sub["z_unadj"], bins=bins, density=True, alpha=0.35,
                color=palette[name], hatch="//", edgecolor="black",
                label=f"{name} z_unadj")
        ax.axvline(0, color="grey", lw=0.5, ls="--")
        ax.set_title(f"{name} ({DESERTS[name][3]})", fontsize=10)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Gnocchi z-score")
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_b_z_hist_overlay.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    print("Analysis B done.")


if __name__ == "__main__":
    main()

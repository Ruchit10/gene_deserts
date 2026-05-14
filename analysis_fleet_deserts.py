#!/usr/bin/env python3
"""Fleet-scale analysis of adjusted vs unadjusted Gnocchi for all deserts.

Scales the exemplar-only diagnostics to all ~633 deserts listed in
`data/hg38_desert_nczscores.txt` by:
  1) labeling 1kb windows by desert ID,
  2) computing per-desert summary statistics for z_adj/z_unadj/delta_z,
  3) categorizing deserts by adjustment behavior, and
  4) generating fleet-level visual summaries.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, ttest_1samp

from utils.desert_utils import (
    DESERTS,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_gnocchi,
)


def _chrom_sort_key(chrom: str) -> tuple[int, Any]:
    value = str(chrom).replace("chr", "")
    if value.isdigit():
        return (0, int(value))
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if value in special:
        return (0, special[value])
    return (1, value)


def _category(row: pd.Series) -> str:
    sign_flip = np.sign(row["mean_z_adj"]) != np.sign(row["mean_z_unadj"])
    if sign_flip:
        return "Sign-flip"
    if row["mean_delta_z"] < -1.0:
        return "Inflated by adjustment"
    if row["mean_delta_z"] > 1.0:
        return "Deflated by adjustment"
    if abs(row["mean_delta_z"]) < 0.5:
        return "Neutral"
    return "Intermediate"


def _per_desert_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for desert_id, sub in df.groupby("desert", sort=False):
        z_adj = sub["z_adj"].values
        z_unadj = sub["z_unadj"].values
        delta = sub["delta_z"].values
        n = len(sub)

        finite_mask = np.isfinite(z_adj) & np.isfinite(z_unadj)
        if finite_mask.any():
            sign_flip_windows = np.mean(np.sign(z_adj[finite_mask]) != np.sign(z_unadj[finite_mask]))
        else:
            sign_flip_windows = np.nan
        if n > 1 and np.nanstd(delta) > 0:
            pval = float(ttest_1samp(delta, popmean=0, nan_policy="omit").pvalue)
        else:
            pval = np.nan

        rows.append(
            {
                "desert_id": desert_id,
                "n_windows": n,
                "mean_z_adj": float(np.nanmean(z_adj)),
                "median_z_adj": float(np.nanmedian(z_adj)),
                "sd_z_adj": float(np.nanstd(z_adj, ddof=1)) if n > 1 else np.nan,
                "mean_z_unadj": float(np.nanmean(z_unadj)),
                "median_z_unadj": float(np.nanmedian(z_unadj)),
                "sd_z_unadj": float(np.nanstd(z_unadj, ddof=1)) if n > 1 else np.nan,
                "mean_delta_z": float(np.nanmean(delta)),
                "median_delta_z": float(np.nanmedian(delta)),
                "sd_delta_z": float(np.nanstd(delta, ddof=1)) if n > 1 else np.nan,
                "skew_delta_z": float(skew(delta, bias=False, nan_policy="omit")) if n > 2 else np.nan,
                "frac_sign_flip_windows": float(sign_flip_windows),
                "ttest_p_delta_z": pval,
            }
        )
    return pd.DataFrame(rows)


def _save_scatter(summary: pd.DataFrame, out_path: str) -> None:
    colors = {
        "Inflated by adjustment": "tab:red",
        "Deflated by adjustment": "tab:blue",
        "Sign-flip": "tab:purple",
        "Neutral": "tab:green",
        "Intermediate": "tab:gray",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    min_n = summary["n_windows"].min()
    max_n = summary["n_windows"].max()
    size = 25 + 175 * (summary["n_windows"] - min_n) / max(1, (max_n - min_n))
    for category, sub in summary.groupby("category"):
        ax.scatter(
            sub["mean_z_adj"],
            sub["mean_z_unadj"],
            s=size.loc[sub.index],
            alpha=0.65,
            color=colors.get(category, "tab:gray"),
            label=f"{category} (n={len(sub)})",
            edgecolor="none",
        )

    lim_lo = min(summary["mean_z_adj"].min(), summary["mean_z_unadj"].min()) - 0.5
    lim_hi = max(summary["mean_z_adj"].max(), summary["mean_z_unadj"].max()) + 0.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.6)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Mean z_adj")
    ax.set_ylabel("Mean z_unadj")
    ax.set_title("All deserts: mean adjusted vs unadjusted z")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")

    exemplars = set(DESERTS.keys())
    for _, row in summary[summary["desert_id"].isin(exemplars)].iterrows():
        ax.annotate(
            row["desert_id"],
            (row["mean_z_adj"], row["mean_z_unadj"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_waterfall(summary: pd.DataFrame, out_path: str) -> None:
    ordered = summary.sort_values("mean_delta_z").reset_index(drop=True)
    colors = ordered["category"].map(
        {
            "Inflated by adjustment": "tab:red",
            "Deflated by adjustment": "tab:blue",
            "Sign-flip": "tab:purple",
            "Neutral": "tab:green",
            "Intermediate": "tab:gray",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(ordered)), ordered["mean_delta_z"], color=colors, width=1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(-1.0, color="tab:red", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(1.0, color="tab:blue", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(-2, len(ordered) + 2)
    ax.set_ylabel("Mean delta_z (z_unadj - z_adj)")
    ax.set_xlabel("Deserts ranked by mean delta_z")
    ax.set_title("Ranked adjustment effect across all deserts")
    ax.set_xticks([])

    exemplars = set(DESERTS.keys())
    ex = ordered[ordered["desert_id"].isin(exemplars)]
    for _, row in ex.iterrows():
        idx = ordered.index[ordered["desert_id"] == row["desert_id"]][0]
        ax.text(idx, row["mean_delta_z"], row["desert_id"], rotation=90, fontsize=7, va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_volcano(summary: pd.DataFrame, out_path: str) -> None:
    p = summary["ttest_p_delta_z"].fillna(1.0).clip(lower=1e-300, upper=1.0)
    y = -np.log10(p)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mean_delta_z"], y, s=30, alpha=0.7, color="tab:gray")
    ax.axvline(-1.0, color="tab:red", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(1.0, color="tab:blue", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-np.log10(0.05), color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Mean delta_z (z_unadj - z_adj)")
    ax.set_ylabel("-log10(p) from one-sample t-test on delta_z")
    ax.set_title("Magnitude vs consistency of adjustment effect")

    exemplars = set(DESERTS.keys())
    for _, row in summary[summary["desert_id"].isin(exemplars)].iterrows():
        yy = -np.log10(max(row["ttest_p_delta_z"], 1e-300)) if pd.notna(row["ttest_p_delta_z"]) else 0
        ax.annotate(row["desert_id"], (row["mean_delta_z"], yy), xytext=(4, 4), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_chrom_plot(summary: pd.DataFrame, out_path: str) -> None:
    ordered_chroms = sorted(summary["chrom"].dropna().unique(), key=_chrom_sort_key)
    data = [summary.loc[summary["chrom"] == c, "mean_delta_z"].values for c in ordered_chroms]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(data, positions=np.arange(1, len(ordered_chroms) + 1), widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e3f0")
        patch.set_alpha(0.8)

    rng = np.random.default_rng(0)
    for i, chrom in enumerate(ordered_chroms, start=1):
        vals = summary.loc[summary["chrom"] == chrom, "mean_delta_z"].values
        x = i + rng.normal(0, 0.04, size=len(vals))
        ax.scatter(x, vals, s=10, alpha=0.5, color="tab:gray")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(1, len(ordered_chroms) + 1))
    ax.set_xticklabels(ordered_chroms, rotation=90, fontsize=8)
    ax.set_ylabel("Per-desert mean delta_z")
    ax.set_xlabel("Chromosome")
    ax.set_title("Chromosome-level distribution of adjustment effects")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_size_plot(summary: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["n_windows"], summary["mean_delta_z"], s=20, alpha=0.6, color="tab:gray")
    coef = np.polyfit(summary["n_windows"], summary["mean_delta_z"], 1)
    xs = np.linspace(summary["n_windows"].min(), summary["n_windows"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="tab:orange", lw=1.5, label=f"slope={coef[0]:+.4f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Desert size (n_windows)")
    ax.set_ylabel("Mean delta_z")
    ax.set_title("Desert size vs adjustment effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_distributions(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("mean_z_adj", "Mean z_adj"),
        ("mean_z_unadj", "Mean z_unadj"),
        ("mean_delta_z", "Mean delta_z"),
    ]
    exemplars = set(DESERTS.keys())

    for ax, (col, label) in zip(axes, specs):
        vals = summary[col].dropna()
        ax.hist(vals, bins=40, alpha=0.7, color="tab:gray", edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Number of deserts")
        ax.set_title(f"Distribution of {label}")

        sub = summary[summary["desert_id"].isin(exemplars)]
        for _, row in sub.iterrows():
            ax.axvline(row[col], lw=1, ls="--", alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi table ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id", "z_adj", "z_unadj"])
    gn["delta_z"] = gn["z_unadj"] - gn["z_adj"]
    print(f"  {len(gn):,} windows loaded")

    print("Loading all desert coordinates ...")
    deserts = load_all_deserts()
    print(f"  {len(deserts):,} deserts loaded")

    print("Labeling windows with desert IDs ...")
    labeled = label_deserts_fleet(gn, deserts)
    labeled = labeled[labeled["desert"].notna()].copy()
    print(f"  {len(labeled):,} windows assigned to deserts")

    print("Computing per-desert summary statistics ...")
    summary = _per_desert_stats(labeled)
    desert_meta = (
        deserts.rename(columns={"start": "desert_start", "end": "desert_end"})[
            ["desert_id", "chrom", "desert_start", "desert_end", "panel_label"]
        ]
    )
    summary = summary.merge(
        desert_meta,
        left_on="desert_id",
        right_on="desert_id",
        how="left",
        validate="one_to_one",
    )
    summary["category"] = summary.apply(_category, axis=1)
    summary = summary.sort_values(["chrom", "desert_start"], key=lambda s: s.map(_chrom_sort_key) if s.name == "chrom" else s)

    out_summary = os.path.join(RESULTS_DIR, "fleet_summary.tsv")
    summary.to_csv(out_summary, sep="\t", index=False)
    print(f"  wrote {out_summary}")

    print("Generating fleet-level visualizations ...")
    _save_scatter(summary, os.path.join(RESULTS_DIR, "fleet_scatter_adj_vs_unadj.png"))
    _save_waterfall(summary, os.path.join(RESULTS_DIR, "fleet_waterfall_delta_z.png"))
    _save_volcano(summary, os.path.join(RESULTS_DIR, "fleet_volcano_delta_z.png"))
    _save_chrom_plot(summary, os.path.join(RESULTS_DIR, "fleet_chrom_delta_z.png"))
    _save_size_plot(summary, os.path.join(RESULTS_DIR, "fleet_size_vs_delta_z.png"))
    _save_distributions(summary, os.path.join(RESULTS_DIR, "fleet_distributions.png"))

    print("\n=== Category counts ===")
    counts = summary["category"].value_counts().sort_values(ascending=False)
    for name, n in counts.items():
        print(f"  {name:24s} {n:4d}")

    print("\n=== Exemplar positions within 633-desert fleet ===")
    rank = summary["mean_delta_z"].rank(method="min", ascending=True)
    n_deserts = len(summary)
    exemplars = sorted(set(DESERTS.keys()) & set(summary["desert_id"]))
    for ex in exemplars:
        row = summary[summary["desert_id"] == ex].iloc[0]
        r = int(rank.loc[row.name])
        pct = 100.0 * r / n_deserts
        print(
            f"  {ex}: mean_z_adj={row['mean_z_adj']:+.3f}, "
            f"mean_z_unadj={row['mean_z_unadj']:+.3f}, "
            f"mean_delta_z={row['mean_delta_z']:+.3f}, "
            f"category={row['category']}, "
            f"delta_rank={r}/{n_deserts} ({pct:.1f}th pct)"
        )

    print("\nFleet analysis done.")


if __name__ == "__main__":
    main()

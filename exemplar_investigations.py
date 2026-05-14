#!/usr/bin/env python3
"""Analysis I: Exemplar-specific spatial investigations.

Produces targeted diagnostic figures per exemplar using only data
available in the repository:

  GD198 (chr8, near 8p23-inv): possible / observed / expected spatial
        trace + recomb_male/_female at all scales.
  GD167 (chr6, recombination): recomb_male vs recomb_female divergence
        and overlay with z_adj/z_unadj.
  GD513 (chr15, highest z): cDNM_maternal_05M profile (the dominant
        LOFO driver) and z_adj/z_unadj overlay, plus dist2telo.

Summary stats per investigation are saved alongside the figures.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from utils.desert_utils import (
    DESERTS,
    FEATURE_SCALES,
    RESULTS_DIR,
    label_deserts,
    load_features,
    load_gnocchi,
)


def _mb(x, _):
    return f"{x / 1e6:.2f} Mb"


def _prep(name: str, df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["desert"] == name].sort_values("start").reset_index(drop=True)
    sub["pos_mid"] = (sub["start"] + sub["end"]) / 2
    return sub


def gd198_panel(df: pd.DataFrame) -> None:
    sub = _prep("GD198", df)
    chrom, s, e, note = DESERTS["GD198"]

    genome_possible_mean = df["possible"].mean()
    genome_possible_sd = df["possible"].std()
    possible_z = (sub["possible"] - genome_possible_mean) / genome_possible_sd

    summary = {
        "desert": "GD198",
        "n_windows": len(sub),
        "mean_possible": sub["possible"].mean(),
        "genome_mean_possible": genome_possible_mean,
        "std_dev_possible": float(possible_z.mean()),
        "frac_windows_possible_lt_median": float(
            (sub["possible"] < df["possible"].median()).mean()),
        "mean_recomb_male_1k": sub["recomb_male_1k"].mean(),
        "mean_recomb_female_1k": sub["recomb_female_1k"].mean(),
        "mean_obs_over_exp_adj": (sub["observed"] / sub["expected"]).mean(),
        "mean_obs_over_exp_unadj": (sub["observed"] / sub["expected_unadj"]).mean(),
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax = axes[0]
    ax.plot(sub["pos_mid"], sub["possible"], lw=0.8, color="tab:purple",
            label="possible")
    ax.axhline(genome_possible_mean, color="k", ls="--", lw=0.6,
               label=f"genome mean = {genome_possible_mean:.0f}")
    ax.set_ylabel("possible variants")
    ax.set_title(f"GD198  {chrom}:{s:,}-{e:,}  ({note})  — coverage/possible trace")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(sub["pos_mid"], sub["observed"], lw=0.7, color="tab:orange",
            label="observed")
    ax.plot(sub["pos_mid"], sub["expected"], lw=0.7, color="tab:blue",
            label="expected (adj)")
    ax.plot(sub["pos_mid"], sub["expected_unadj"], lw=0.7, color="tab:green",
            label="expected (unadj)")
    ax.set_ylabel("counts")
    ax.set_title("GD198 — Observed vs Expected tracks")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(sub["pos_mid"], sub["z_adj"], lw=0.7, label="z_adj", color="tab:blue")
    ax.plot(sub["pos_mid"], sub["z_unadj"], lw=0.7, label="z_unadj", color="tab:orange")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("z-score")
    ax.set_title("GD198 — z scores")
    ax.legend(fontsize=8)

    ax = axes[3]
    for scale, ls in zip(FEATURE_SCALES, ["-", "--", ":", "-."]):
        ax.plot(sub["pos_mid"], sub[f"recomb_male_{scale}"], lw=0.7, ls=ls,
                label=f"recomb_male_{scale}")
        ax.plot(sub["pos_mid"], sub[f"recomb_female_{scale}"], lw=0.7, ls=ls,
                alpha=0.6,
                label=f"recomb_female_{scale}")
    ax.set_ylabel("recomb rate")
    ax.set_xlabel(f"{chrom} position")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))
    ax.legend(fontsize=6, ncol=4)
    ax.set_title("GD198 — recombination rate (male/female) at 1k/10k/100k/1M")

    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_i_GD198_panel.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")
    return summary


def gd167_panel(df: pd.DataFrame) -> None:
    sub = _prep("GD167", df)
    chrom, s, e, note = DESERTS["GD167"]

    divergence_1k = sub["recomb_male_1k"] - sub["recomb_female_1k"]
    summary = {
        "desert": "GD167",
        "n_windows": len(sub),
        "mean_recomb_male_1M": sub["recomb_male_1M"].mean(),
        "mean_recomb_female_1M": sub["recomb_female_1M"].mean(),
        "mean_recomb_divergence_1k": divergence_1k.mean(),
        "genome_mean_recomb_divergence_1k":
            (df["recomb_male_1k"] - df["recomb_female_1k"]).mean(),
        "sd_recomb_divergence_1k_in_desert": divergence_1k.std(),
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    for scale, ls in zip(FEATURE_SCALES, ["-", "--", ":", "-."]):
        ax.plot(sub["pos_mid"], sub[f"recomb_male_{scale}"],
                lw=0.8, ls=ls, label=f"recomb_male_{scale}", color="tab:blue",
                alpha=0.7)
    ax.set_ylabel("recomb_male")
    ax.set_title(f"GD167  {chrom}:{s:,}-{e:,}  ({note})  — male recombination")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[1]
    for scale, ls in zip(FEATURE_SCALES, ["-", "--", ":", "-."]):
        ax.plot(sub["pos_mid"], sub[f"recomb_female_{scale}"],
                lw=0.8, ls=ls, label=f"recomb_female_{scale}", color="tab:red",
                alpha=0.7)
    ax.set_ylabel("recomb_female")
    ax.set_title("GD167 — female recombination")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[2]
    ax.plot(sub["pos_mid"], sub["z_adj"], lw=0.8, color="tab:blue", label="z_adj")
    ax.plot(sub["pos_mid"], sub["z_unadj"], lw=0.8, color="tab:orange",
            label="z_unadj")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("z-score")
    ax.set_xlabel(f"{chrom} position")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))
    ax.legend(fontsize=8)
    ax.set_title("GD167 — z trace")

    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_i_GD167_panel.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")
    return summary


def gd513_panel(df: pd.DataFrame) -> None:
    sub = _prep("GD513", df)
    chrom, s, e, note = DESERTS["GD513"]

    summary = {
        "desert": "GD513",
        "n_windows": len(sub),
        "mean_cDNM_maternal_05M_1k": sub["cDNM_maternal_05M_1k"].mean(),
        "genome_mean_cDNM_maternal_05M_1k": df["cDNM_maternal_05M_1k"].mean(),
        "mean_dist2telo_1M_Mb": sub["dist2telo_1M"].mean(),
        "mean_z_adj": sub["z_adj"].mean(),
        "mean_z_unadj": sub["z_unadj"].mean(),
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax = axes[0]
    ax.plot(sub["pos_mid"], sub["z_adj"], lw=0.8, color="tab:blue", label="z_adj")
    ax.plot(sub["pos_mid"], sub["z_unadj"], lw=0.8, color="tab:orange",
            label="z_unadj")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("z-score")
    ax.set_title(f"GD513  {chrom}:{s:,}-{e:,}  ({note})  — z scores")
    ax.legend(fontsize=8)

    ax = axes[1]
    for scale, ls in zip(FEATURE_SCALES, ["-", "--", ":", "-."]):
        col = f"cDNM_maternal_05M_{scale}"
        ax.plot(sub["pos_mid"], sub[col], lw=0.8, ls=ls,
                label=col, alpha=0.8)
    ax.set_ylabel("cDNM_maternal_05M")
    ax.set_title("GD513 — cDNM_maternal_05M (dominant LOFO driver)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for scale, ls in zip(FEATURE_SCALES, ["-", "--", ":", "-."]):
        col = f"dist2telo_{scale}"
        ax.plot(sub["pos_mid"], sub[col], lw=0.8, ls=ls,
                label=col, alpha=0.8)
    ax.set_ylabel("dist2telo")
    ax.set_title("GD513 — distance to telomere")
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(sub["pos_mid"], sub["possible"], lw=0.8, color="tab:purple",
            label="possible")
    ax.plot(sub["pos_mid"], sub["observed"], lw=0.8, color="tab:orange",
            label="observed", alpha=0.8)
    ax.plot(sub["pos_mid"], sub["expected"], lw=0.8, color="tab:blue",
            label="expected (adj)")
    ax.plot(sub["pos_mid"], sub["expected_unadj"], lw=0.8, color="tab:green",
            label="expected (unadj)")
    ax.set_ylabel("counts")
    ax.set_xlabel(f"{chrom} position")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))
    ax.legend(fontsize=7)
    ax.set_title("GD513 — possible/observed/expected tracks")

    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_i_GD513_panel.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")
    return summary


def main() -> None:
    print("Loading ...")
    gn = load_gnocchi()
    gn = label_deserts(gn)
    feats = load_features()
    df = gn.merge(feats, on="element_id", how="inner")
    print(f"  {len(df):,} merged rows")

    summaries = []
    for fn in (gd198_panel, gd167_panel, gd513_panel):
        summaries.append(fn(df))

    pd.DataFrame(summaries).to_csv(
        os.path.join(RESULTS_DIR, "analysis_i_summary.tsv"),
        sep="\t", index=False)
    print("  wrote results/analysis_i_summary.tsv")
    for s in summaries:
        print("\n", s)

    print("\nAnalysis I done.")


if __name__ == "__main__":
    main()

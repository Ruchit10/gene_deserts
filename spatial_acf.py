#!/usr/bin/env python3
"""Analysis C: Spatial autocorrelation of z_adj, z_unadj, and delta_z.

For each desert we sort 1kb windows by genomic coordinate and compute
the ACF at lags 1..100kb for z_adj, z_unadj, and delta_z. We compare
to a background of 200 same-length contiguous stretches sampled
genome-wide, using the 2.5/97.5 percentiles as a confidence band.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from utils.desert_utils import (
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts,
    load_gnocchi,
)

MAX_LAG = 100  # lags in 1kb units
N_BG = 200
RNG = np.random.default_rng(0)


def compute_acf(values: np.ndarray, nlags: int = MAX_LAG) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < nlags + 5:
        return np.full(nlags + 1, np.nan)
    return acf(v, nlags=nlags, fft=True, missing="drop")


def prepare_chrom_arrays(df: pd.DataFrame, cols: list[str]
                          ) -> dict[str, dict[str, np.ndarray]]:
    """Pre-sort each chromosome once and cache numpy arrays per response.

    Returns {chrom: {"start": np.ndarray, col: np.ndarray, ...}}.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for c, sub in df.groupby("chrom", sort=False):
        sub = sub.sort_values("start")
        d = {"start": sub["start"].to_numpy()}
        for col in cols:
            d[col] = sub[col].to_numpy(dtype=float)
        out[c] = d
    return out


def sample_background_acf(chrom_arrays: dict[str, dict[str, np.ndarray]],
                          length: int, col: str, n: int = N_BG) -> np.ndarray:
    """Sample n contiguous stretches of `length` windows and return stacked ACFs."""
    chroms = [c for c, d in chrom_arrays.items() if len(d["start"]) >= length + 5]
    if not chroms:
        return np.full((1, MAX_LAG + 1), np.nan)
    out = []
    attempts = 0
    while len(out) < n and attempts < 10 * n:
        attempts += 1
        c = chroms[RNG.integers(0, len(chroms))]
        starts = chrom_arrays[c]["start"]
        vals = chrom_arrays[c][col]
        i = RNG.integers(0, len(starts) - length)
        # Ensure the stretch is contiguous (gaps < 2kb between adjacent windows)
        seg = vals[i:i + length]
        # Require that starts at both ends span roughly `length` kb
        span = starts[i + length - 1] - starts[i]
        if span > (length * 2) * 1000:  # too gappy
            continue
        a = compute_acf(seg)
        if np.all(np.isfinite(a)):
            out.append(a)
    return np.vstack(out) if out else np.full((1, MAX_LAG + 1), np.nan)


def main() -> None:
    print("Loading gnocchi table ...")
    df = load_gnocchi(usecols=["chrom", "start", "end", "element_id",
                                "z_adj", "z_unadj", "delta_z"])
    df = label_deserts(df)
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    print(f"  {len(df):,} windows loaded")

    lag_kb = np.arange(MAX_LAG + 1)
    responses = ["z_adj", "z_unadj", "delta_z"]

    print("Preparing per-chromosome numpy arrays ...")
    chrom_arrays = prepare_chrom_arrays(df, responses)
    print(f"  {len(chrom_arrays)} chromosomes cached")

    rows = []
    acf_dict: dict[str, dict[str, np.ndarray]] = {}
    bg_dict: dict[str, dict[str, np.ndarray]] = {}

    for name in DESERT_ORDER:
        sub = df[df["desert"] == name].sort_values("start")
        length = len(sub)
        print(f"\n== {name}: {length} windows ==")
        acf_dict[name] = {}
        bg_dict[name] = {}
        for resp in responses:
            a = compute_acf(sub[resp].values)
            acf_dict[name][resp] = a
            print(f"  ACF({resp})   lag1={a[1]:+.3f}  lag10={a[10]:+.3f}  lag50={a[50]:+.3f}")
            bg = sample_background_acf(chrom_arrays, length=length, col=resp, n=N_BG)
            bg_dict[name][resp] = bg
            for lag in (1, 5, 10, 25, 50, 100):
                if lag <= MAX_LAG:
                    rows.append({"desert": name, "response": resp, "lag_kb": lag,
                                 "acf": a[lag],
                                 "bg_mean": float(np.nanmean(bg[:, lag])),
                                 "bg_lo": float(np.nanpercentile(bg[:, lag], 2.5)),
                                 "bg_hi": float(np.nanpercentile(bg[:, lag], 97.5))})

    acf_summary = pd.DataFrame(rows)
    tsv_path = os.path.join(RESULTS_DIR, "analysis_c_acf_summary.tsv")
    acf_summary.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  wrote {tsv_path}")

    # ── Plots: one figure per desert, 3 subplots (z_adj, z_unadj, delta_z) ──
    for name in DESERT_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, resp in zip(axes, responses):
            bg = bg_dict[name][resp]
            bg_mean = np.nanmean(bg, axis=0)
            bg_lo = np.nanpercentile(bg, 2.5, axis=0)
            bg_hi = np.nanpercentile(bg, 97.5, axis=0)
            ax.fill_between(lag_kb, bg_lo, bg_hi, color="lightgray", alpha=0.6,
                            label="background 95%")
            ax.plot(lag_kb, bg_mean, color="gray", lw=0.8, label="background mean")
            ax.plot(lag_kb, acf_dict[name][resp], color="tab:red", lw=1.2,
                    label=f"{name} ACF")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"ACF({resp}) — {name}")
            ax.set_xlabel("lag (kb)")
            if ax is axes[0]:
                ax.set_ylabel("autocorrelation")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f"analysis_c_acf_{name}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {fig_path}")

    # ── Combined z_adj ACF comparison across deserts ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in DESERT_ORDER:
        ax.plot(lag_kb, acf_dict[name]["z_adj"], label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (kb)")
    ax.set_ylabel("ACF(z_adj)")
    ax.set_title("z_adj ACF across deserts")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_c_acf_zadj_combined.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    print("\nAnalysis C done.")


if __name__ == "__main__":
    main()

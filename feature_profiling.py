#!/usr/bin/env python3
"""Analysis A: Feature profiling of deserts vs genome-wide background.

For each of the 52 feature-window columns, compute genome-wide mean
and SD (restricted to windows that survive in the merged Gnocchi
table) and the per-desert mean. Standardize per-desert deviations and
visualize as a 5 x 52 heatmap.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.desert_utils import (
    DESERTS,
    DESERT_ORDER,
    FEATURE_BASES,
    FEATURE_COLUMNS,
    FEATURE_SCALES,
    RESULTS_DIR,
    label_deserts,
    load_features,
    load_gnocchi,
)


def main() -> None:
    print("Loading gnocchi windows ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id"])
    gn = label_deserts(gn)
    print(f"  {len(gn):,} windows")

    print("Loading features ...")
    feats = load_features()
    df = gn.merge(feats, on="element_id", how="inner")
    print(f"  {len(df):,} merged rows")

    print("Computing genome-wide and per-desert stats ...")
    genome_mean = df[FEATURE_COLUMNS].mean()
    genome_sd = df[FEATURE_COLUMNS].std(ddof=1)
    genome_median = df[FEATURE_COLUMNS].median()

    # Summary long table
    rows = []
    for f in FEATURE_COLUMNS:
        rows.append({"feature": f, "scope": "genome",
                     "mean": genome_mean[f], "sd": genome_sd[f],
                     "median": genome_median[f], "std_dev": 0.0,
                     "n": len(df)})
    for name in DESERT_ORDER:
        sub = df[df["desert"] == name]
        n = len(sub)
        for f in FEATURE_COLUMNS:
            m = sub[f].mean()
            med = sub[f].median()
            sd = genome_sd[f]
            std_dev = (m - genome_mean[f]) / sd if sd else np.nan
            rows.append({"feature": f, "scope": name,
                         "mean": m, "sd": sub[f].std(ddof=1),
                         "median": med, "std_dev": std_dev, "n": n})
    long = pd.DataFrame(rows)
    tsv = os.path.join(RESULTS_DIR, "feature_profile_by_desert.tsv")
    long.to_csv(tsv, sep="\t", index=False)
    print(f"  wrote {tsv}")

    # Wide heatmap-ready pivot of standardized deviations
    desert_rows = long[long["scope"].isin(DESERT_ORDER)].pivot(
        index="scope", columns="feature", values="std_dev")
    desert_rows = desert_rows.reindex(DESERT_ORDER)[FEATURE_COLUMNS]

    # ── Heatmap ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 4))
    vmax = float(np.nanmax(np.abs(desert_rows.values)))
    vmax = min(vmax, 6.0)
    im = ax.imshow(desert_rows.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(DESERT_ORDER)))
    ax.set_yticklabels(DESERT_ORDER)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=6)
    for i, scale in enumerate(FEATURE_SCALES):
        ax.axvline(i * len(FEATURE_BASES) - 0.5, color="white", lw=1.2)
    ax.set_title("Desert feature profile: (desert_mean - genome_mean) / genome_sd")
    fig.colorbar(im, ax=ax, label="z vs genome", shrink=0.8)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_a_feature_heatmap.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # ── Top extreme features per desert (|std_dev| >= 2) ──────────────────
    print("\n=== Features with |std_dev| >= 2 per desert ===")
    for name in DESERT_ORDER:
        sub = long[long["scope"] == name].copy()
        sub["abs"] = sub["std_dev"].abs()
        extreme = sub[sub["abs"] >= 2.0].sort_values("abs", ascending=False)
        print(f"\n  {name} ({DESERTS[name][3]}): {len(extreme)} extreme features")
        if len(extreme):
            for r in extreme.head(12).itertuples():
                print(f"    {r.feature:30s}  std_dev={r.std_dev:+.2f}  mean={r.mean:.3g}  genome_mean={genome_mean[r.feature]:.3g}")

    print("\nAnalysis A done.")


if __name__ == "__main__":
    main()

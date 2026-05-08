#!/usr/bin/env python3
"""Analysis D: Feature/score correlations within deserts and genome-wide.

For each desert (and genome-wide) we compute Pearson and Spearman
correlations of `z_adj`, `z_unadj`, and `delta_z` with each of the 52
feature columns. Heatmaps per response variable summarize the result.

Note: the GD588 z-vs-GC targeted panel that was previously here has been
moved to `desert_gc_extrema.py`, which provides a broader dip/spike-aware
GC analysis for all five exemplar deserts and the full 633-desert fleet.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desert_utils import (
    DESERTS,
    DESERT_ORDER,
    FEATURE_COLUMNS,
    RESULTS_DIR,
    label_deserts,
    load_features,
    load_gnocchi,
)


def _corrs(df: pd.DataFrame, response: str, method: str) -> pd.Series:
    sub = df[[response] + FEATURE_COLUMNS].dropna()
    if len(sub) < 10:
        return pd.Series({f: np.nan for f in FEATURE_COLUMNS})
    return sub.corr(method=method)[response].drop(response)


def main() -> None:
    print("Loading gnocchi table ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id",
                                "z_adj", "z_unadj", "delta_z"])
    gn = label_deserts(gn)
    print(f"  {len(gn):,} windows loaded")

    print("Loading features ...")
    feats = load_features()
    print(f"  {len(feats):,} feature rows loaded")

    print("Merging ...")
    df = gn.merge(feats, on="element_id", how="inner")
    print(f"  {len(df):,} merged rows")

    # Subsample genome-wide for speed of Spearman on ~2M rows
    rng = np.random.default_rng(0)
    if len(df) > 200_000:
        genome_sample = df.sample(200_000, random_state=0)
    else:
        genome_sample = df

    responses = ["z_adj", "z_unadj", "delta_z"]
    methods = ["pearson", "spearman"]

    all_rows = []
    for method in methods:
        # Genome-wide
        for resp in responses:
            s = _corrs(genome_sample, resp, method)
            for f, v in s.items():
                all_rows.append({"scope": "genome", "desert": None,
                                 "method": method, "response": resp,
                                 "feature": f, "r": v})
        for name in DESERT_ORDER:
            sub = df[df["desert"] == name]
            for resp in responses:
                s = _corrs(sub, resp, method)
                for f, v in s.items():
                    all_rows.append({"scope": "desert", "desert": name,
                                     "method": method, "response": resp,
                                     "feature": f, "r": v})

    corr_long = pd.DataFrame(all_rows)
    corr_path = os.path.join(RESULTS_DIR, "analysis_d_feature_correlations.tsv")
    corr_long.to_csv(corr_path, sep="\t", index=False)
    print(f"  wrote {corr_path}")

    # ── Heatmaps (one per response, Pearson) ─────────────────────────────
    for resp in responses:
        pivot_rows = []
        labels = ["genome"] + DESERT_ORDER
        for label in labels:
            if label == "genome":
                s = corr_long[(corr_long["method"] == "pearson") &
                              (corr_long["response"] == resp) &
                              (corr_long["scope"] == "genome")]
            else:
                s = corr_long[(corr_long["method"] == "pearson") &
                              (corr_long["response"] == resp) &
                              (corr_long["desert"] == label)]
            row = {"label": label}
            for _, r in s.iterrows():
                row[r["feature"]] = r["r"]
            pivot_rows.append(row)
        pivot = pd.DataFrame(pivot_rows).set_index("label")[FEATURE_COLUMNS]

        fig, ax = plt.subplots(figsize=(16, 4))
        vmax = float(np.nanmax(np.abs(pivot.values)))
        vmax = max(0.3, min(vmax, 1.0))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(FEATURE_COLUMNS)))
        ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=6)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Pearson correlation of {resp} with 52 features")
        fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.9)
        fig.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, f"analysis_d_heatmap_{resp}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {fig_path}")

    # ── Summary print: top |r| features per desert per response ──────────
    print("\n=== Top |Pearson r| features per desert (z_adj) ===")
    for name in DESERT_ORDER:
        sub = corr_long[(corr_long["method"] == "pearson") &
                         (corr_long["response"] == "z_adj") &
                         (corr_long["desert"] == name)].copy()
        sub["abs_r"] = sub["r"].abs()
        top = sub.nlargest(5, "abs_r")
        print(f"  {name}: " + ", ".join(
            [f"{r.feature}({r.r:+.2f})" for r in top.itertuples()]))

    print("\n=== Top |Pearson r| features per desert (delta_z) ===")
    for name in DESERT_ORDER:
        sub = corr_long[(corr_long["method"] == "pearson") &
                         (corr_long["response"] == "delta_z") &
                         (corr_long["desert"] == name)].copy()
        sub["abs_r"] = sub["r"].abs()
        top = sub.nlargest(5, "abs_r")
        print(f"  {name}: " + ", ".join(
            [f"{r.feature}({r.r:+.2f})" for r in top.itertuples()]))

    print("\nAnalysis D done.")


if __name__ == "__main__":
    main()

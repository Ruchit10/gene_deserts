#!/usr/bin/env python3
"""Analysis F: Approximate leave-one-feature-out (LOFO).

Using the genome-wide ridge from Analysis E, zero out each feature's
standardized contribution, yielding a counterfactual `z_LOFO_k` for
each window, and rank features by |mean desert z-shift|. We also
collapse across window scales (1k/10k/100k/1M) for each of the 13
underlying features to produce a compact per-desert ranking.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils.desert_utils import (
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
    print("Loading ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id",
                                "z_adj", "z_unadj", "delta_z"])
    gn = label_deserts(gn)
    feats = load_features()
    df = gn.merge(feats, on="element_id", how="inner")

    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Xz = (X_raw - mu) / sd
    y = df["delta_z"].to_numpy(dtype=float)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(Xz, y)
    beta = ridge.coef_
    print(f"  ridge R^2 = {1 - np.var(y - ridge.predict(Xz))/np.var(y):.3f}")

    # Per-window full adjustment contribution
    full_contrib = Xz * beta  # (n, 52)
    delta_full = full_contrib.sum(axis=1) + ridge.intercept_  # ~= delta_z

    # ── Per-feature LOFO: shift in z_adj if we set beta_k to 0 ─────────────
    # z_adj = z_unadj - delta_z. Removing feature k from delta means the
    # 'adjusted' z becomes: z_LOFO_k = z_unadj - (delta_full - beta_k * xz_k)
    # Equivalently, shift in z_adj = (new z_adj) - z_adj = beta_k * xz_k.
    # Mean shift per desert is already computed in Analysis E's contrib table;
    # here we rank features and also collapse by feature base.

    per_feature_rows = []
    for name in DESERT_ORDER:
        mask = (df["desert"] == name).to_numpy()
        if mask.sum() == 0:
            continue
        for k, f in enumerate(FEATURE_COLUMNS):
            # shift in z_adj from removing feature k = + beta_k * xz_k
            shift = (Xz[mask, k] * beta[k]).mean()
            per_feature_rows.append({
                "desert": name, "feature": f, "coef": beta[k],
                "mean_xz_in_desert": float(Xz[mask, k].mean()),
                "mean_shift_z_adj": float(shift),
            })
    per_feat = pd.DataFrame(per_feature_rows)
    per_feat["abs_shift"] = per_feat["mean_shift_z_adj"].abs()
    per_feat = per_feat.sort_values(["desert", "abs_shift"],
                                     ascending=[True, False])
    path = os.path.join(RESULTS_DIR, "analysis_f_lofo_per_feature.tsv")
    per_feat.to_csv(path, sep="\t", index=False)
    print(f"  wrote {path}")

    # ── Collapse across scales to per-base ranking ────────────────────────
    collapsed_rows = []
    for name in DESERT_ORDER:
        mask = (df["desert"] == name).to_numpy()
        for base in FEATURE_BASES:
            cols = [f"{base}_{s}" for s in FEATURE_SCALES if f"{base}_{s}" in FEATURE_COLUMNS]
            idxs = [FEATURE_COLUMNS.index(c) for c in cols]
            sub_contrib = (Xz[mask][:, idxs] * beta[idxs]).sum(axis=1)
            collapsed_rows.append({
                "desert": name, "feature_base": base,
                "mean_shift_z_adj": float(sub_contrib.mean()),
            })
    collapsed = pd.DataFrame(collapsed_rows)
    collapsed["abs_shift"] = collapsed["mean_shift_z_adj"].abs()
    collapsed = collapsed.sort_values(["desert", "abs_shift"],
                                       ascending=[True, False])
    path = os.path.join(RESULTS_DIR, "analysis_f_lofo_collapsed.tsv")
    collapsed.to_csv(path, sep="\t", index=False)
    print(f"  wrote {path}")

    print("\n=== LOFO ranking per desert (collapsed by feature base) ===")
    print("  mean_shift_z_adj = shift in mean z_adj from removing that feature's effect")
    for name in DESERT_ORDER:
        sub = collapsed[collapsed["desert"] == name].head(13)
        print(f"\n  {name}:")
        for r in sub.itertuples():
            print(f"    {r.feature_base:22s}  shift = {r.mean_shift_z_adj:+.3f}")

    # ── Bar charts ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(DESERT_ORDER), 1, figsize=(10, 2.4 * len(DESERT_ORDER)))
    for ax, name in zip(axes, DESERT_ORDER):
        sub = collapsed[collapsed["desert"] == name]
        sub = sub.sort_values("mean_shift_z_adj")
        colors = ["tab:red" if v < 0 else "tab:blue" for v in sub["mean_shift_z_adj"]]
        ax.barh(sub["feature_base"], sub["mean_shift_z_adj"], color=colors)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{name}: z_adj shift if feature removed", fontsize=10)
        ax.set_xlabel("mean shift in z_adj (removal = unadjust)")
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_f_lofo_barcharts.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # ── Heatmap 5 x 13 ─────────────────────────────────────────────────────
    pivot = collapsed.pivot(index="desert", columns="feature_base",
                             values="mean_shift_z_adj").reindex(DESERT_ORDER)[FEATURE_BASES]
    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = float(np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(DESERT_ORDER)))
    ax.set_yticklabels(DESERT_ORDER)
    ax.set_xticks(range(len(FEATURE_BASES)))
    ax.set_xticklabels(FEATURE_BASES, rotation=45, ha="right")
    ax.set_title("LOFO: mean z_adj shift per desert when feature is removed")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_f_lofo_heatmap.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    print("\nAnalysis F done.")


if __name__ == "__main__":
    main()

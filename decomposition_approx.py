#!/usr/bin/env python3
"""Analysis E: Approximate r-decomposition via delta_z regression.

Fit a genome-wide ridge regression of `delta_z = z_unadj - z_adj` on
the 52 z-standardized feature columns. The per-feature contribution
for a window is `beta_k * standardized_x_k`; per-desert bar charts
show the mean contribution. Also saves per-desert local OLS for
comparison.

Caveat: this is an approximation in raw feature space, not in the
PCA space used by the original pipeline. Predictive R^2 is reported
as a diagnostic.
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
    print("Loading gnocchi ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id",
                                "z_adj", "z_unadj", "delta_z"])
    gn = label_deserts(gn)
    print("Loading features ...")
    feats = load_features()
    df = gn.merge(feats, on="element_id", how="inner")
    print(f"  {len(df):,} merged rows")

    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    finite_mask = np.all(np.isfinite(X_raw), axis=1) & np.isfinite(df["delta_z"].values)
    Xf = X_raw[finite_mask]
    yf = df.loc[finite_mask, "delta_z"].to_numpy(dtype=float)
    print(f"  {finite_mask.sum():,} rows with finite features + delta_z")

    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Xz_all = (X_raw - mu) / sd  # standardized on all rows (NaNs propagate)
    Xz_fit = (Xf - mu) / sd

    # ── Genome-wide ridge ─────────────────────────────────────────────────
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(Xz_fit, yf)
    yhat = ridge.predict(Xz_fit)
    r2 = 1 - np.var(yf - yhat) / np.var(yf)
    print(f"\nGenome-wide ridge R^2 on delta_z: {r2:.3f}")
    coefs = pd.Series(ridge.coef_, index=FEATURE_COLUMNS)
    coefs.to_csv(os.path.join(RESULTS_DIR, "analysis_e_ridge_coefs.tsv"),
                 sep="\t", header=["coef"])

    # Per-window per-feature contribution: beta_k * x_k^std
    df["delta_z_hat"] = np.where(finite_mask, yhat, np.nan) \
        if len(yhat) == finite_mask.sum() else np.nan
    # Safer: assign across full index
    df.loc[finite_mask, "delta_z_hat"] = yhat

    # Per-desert contributions (averaged)
    desert_contrib_rows = []
    for name in DESERT_ORDER:
        sub_mask = (df["desert"] == name).to_numpy() & finite_mask
        if sub_mask.sum() == 0:
            continue
        Xz_sub = Xz_all[sub_mask]
        contrib = Xz_sub * ridge.coef_  # shape (n, 52)
        mean_contrib = contrib.mean(axis=0)
        row = {"desert": name,
               "n": int(sub_mask.sum()),
               "mean_delta_z": float(df.loc[sub_mask, "delta_z"].mean()),
               "mean_delta_z_hat": float(df.loc[sub_mask, "delta_z_hat"].mean()),
               "sum_contrib": float(mean_contrib.sum() + ridge.intercept_),
               }
        for f, c in zip(FEATURE_COLUMNS, mean_contrib):
            row[f] = float(c)
        desert_contrib_rows.append(row)
    contrib_df = pd.DataFrame(desert_contrib_rows)
    contrib_path = os.path.join(RESULTS_DIR, "delta_z_feature_attribution.tsv")
    contrib_df.to_csv(contrib_path, sep="\t", index=False)
    print(f"  wrote {contrib_path}")

    # ── Per-desert local OLS (for comparison with global) ─────────────────
    local_rows = []
    for name in DESERT_ORDER:
        sub = df[(df["desert"] == name) & finite_mask]
        if len(sub) < len(FEATURE_COLUMNS) + 5:
            continue
        Xs = (sub[FEATURE_COLUMNS].to_numpy() - mu) / sd
        ys = sub["delta_z"].to_numpy()
        local = Ridge(alpha=1.0, fit_intercept=True)
        local.fit(Xs, ys)
        yh = local.predict(Xs)
        r2_local = 1 - np.var(ys - yh) / np.var(ys)
        for f, c in zip(FEATURE_COLUMNS, local.coef_):
            local_rows.append({"desert": name, "feature": f,
                               "local_coef": float(c),
                               "global_coef": float(coefs[f]),
                               "local_r2": float(r2_local)})
    local_df = pd.DataFrame(local_rows)
    local_path = os.path.join(RESULTS_DIR, "analysis_e_local_vs_global_coefs.tsv")
    local_df.to_csv(local_path, sep="\t", index=False)
    print(f"  wrote {local_path}")

    # ── Bar chart: per-desert top contributions ───────────────────────────
    fig, axes = plt.subplots(len(DESERT_ORDER), 1, figsize=(12, 3 * len(DESERT_ORDER)))
    for ax, name in zip(axes, DESERT_ORDER):
        row = contrib_df[contrib_df["desert"] == name]
        if row.empty:
            ax.set_title(f"{name}: no data")
            continue
        vals = row[FEATURE_COLUMNS].iloc[0].to_numpy(dtype=float)
        order = np.argsort(-np.abs(vals))
        top = order[:15]
        colors = ["tab:red" if v < 0 else "tab:blue" for v in vals[top]]
        ax.bar(range(len(top)), vals[top], color=colors)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(np.array(FEATURE_COLUMNS)[top], rotation=60,
                           ha="right", fontsize=8)
        mean_dz = row["mean_delta_z"].iloc[0]
        mean_dz_hat = row["mean_delta_z_hat"].iloc[0]
        ax.set_title(
            f"{name}: top feature contributions to delta_z  "
            f"(mean delta_z = {mean_dz:+.2f}, model pred = {mean_dz_hat:+.2f})",
            fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("beta_k * (x_k - mu)/sd (avg over desert)")
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_e_feature_contributions_per_desert.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # ── Collapsed by feature base (summing across scales) ──────────────────
    collapsed = pd.DataFrame(index=DESERT_ORDER, columns=FEATURE_BASES, dtype=float)
    for name in DESERT_ORDER:
        row = contrib_df[contrib_df["desert"] == name]
        if row.empty:
            continue
        for base in FEATURE_BASES:
            cols = [f"{base}_{s}" for s in FEATURE_SCALES if f"{base}_{s}" in FEATURE_COLUMNS]
            collapsed.loc[name, base] = float(row[cols].sum(axis=1).iloc[0])
    collapsed.to_csv(os.path.join(RESULTS_DIR,
                                   "analysis_e_feature_contributions_collapsed.tsv"),
                     sep="\t")
    fig, ax = plt.subplots(figsize=(12, 4))
    vmax = float(np.nanmax(np.abs(collapsed.values)))
    vmax = max(vmax, 0.1)
    im = ax.imshow(collapsed.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(DESERT_ORDER)))
    ax.set_yticklabels(DESERT_ORDER)
    ax.set_xticks(range(len(FEATURE_BASES)))
    ax.set_xticklabels(FEATURE_BASES, rotation=45, ha="right")
    ax.set_title("Sum of per-scale contributions to mean delta_z per desert")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "analysis_e_feature_contributions_collapsed.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    print("\n=== Per-desert mean delta_z vs model-predicted ===")
    for r in contrib_df.itertuples():
        print(f"  {r.desert}: observed={r.mean_delta_z:+.3f}  predicted={r.mean_delta_z_hat:+.3f}  "
              f"residual={r.mean_delta_z - r.mean_delta_z_hat:+.3f}")

    print("\nAnalysis E done.")


if __name__ == "__main__":
    main()

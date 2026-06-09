#!/usr/bin/env python3
"""Roulette-vs-Gnocchi comparison using direct Roulette rates (no calibration).

Implements four lenses on the gene-desert window subset:
  A) O/P vs mu regression residuals (primary) + O/mu summary (supplementary)
  B) Mutation-model concordance (mu_roulette vs expected_unadj)
  C) Rank-based concordance (mu rank vs expected_unadj rank)
  D) Poisson deviance residuals with log(mu) offset ("roulette z")
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils.desert_utils import (
    DESERT_ORDER,
    DESERTS,
    RESULTS_DIR,
    desert_palette,
    label_deserts,
    label_deserts_fleet,
    load_all_deserts,
    load_gnocchi,
)


ROULETTE_PATH = os.path.join("data", "roulette_gene_deserts_mu.tsv.bgz")


def _safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan)
    return pearsonr(x[mask], y[mask])


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan)
    return spearmanr(x[mask], y[mask])


def _deviance_residual(obs: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Poisson deviance residuals."""
    obs = np.asarray(obs, dtype=float)
    fit = np.asarray(fit, dtype=float)
    eps = 1e-12
    fit = np.clip(fit, eps, None)
    term = np.where(obs > 0, obs * np.log(obs / fit), 0.0)
    dev = 2.0 * (term - (obs - fit))
    dev = np.clip(dev, 0.0, None)
    return np.sign(obs - fit) * np.sqrt(dev)


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


def _chrom_sort_key(chrom: str) -> tuple[int, Any]:
    value = str(chrom).replace("chr", "")
    if value.isdigit():
        return (0, int(value))
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if value in special:
        return (0, special[value])
    return (1, value)


def _save_model_concordance(df: pd.DataFrame, out_path: str) -> None:
    pr, _ = _safe_corr(df["mu"], df["expected_unadj"])
    sr, _ = _safe_spearman(df["mu"], df["expected_unadj"])

    rng = np.random.default_rng(0)
    bg = df.sample(min(120_000, len(df)), random_state=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(bg["mu"], bg["expected_unadj"], s=2, alpha=0.12, color="lightgray", label="desert windows")

    palette = desert_palette()
    for desert in DESERT_ORDER:
        sub = df[df["desert_exemplar"] == desert]
        ax.scatter(sub["mu"], sub["expected_unadj"], s=10, alpha=0.6, color=palette[desert], label=desert)

    ax.set_xlabel("Roulette mu")
    ax.set_ylabel("Gnocchi expected_unadj")
    ax.set_title(f"Roulette vs gnomAD mutation model concordance\nPearson r={pr:.3f}, Spearman rho={sr:.3f}")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_op_vs_mu(df: pd.DataFrame, beta0: float, beta1: float, out_path: str) -> None:
    pr, _ = _safe_corr(df["mu"], df["op"])
    sr, _ = _safe_spearman(df["mu"], df["op"])
    bg = df.sample(min(120_000, len(df)), random_state=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(bg["mu"], bg["op"], s=2, alpha=0.12, color="lightgray", label="desert windows")
    palette = desert_palette()
    for desert in DESERT_ORDER:
        sub = df[df["desert_exemplar"] == desert]
        ax.scatter(sub["mu"], sub["op"], s=10, alpha=0.6, color=palette[desert], label=desert)

    xs = np.linspace(float(df["mu"].min()), float(df["mu"].max()), 200)
    ys = beta0 + beta1 * xs
    ax.plot(xs, ys, color="black", lw=1.5, ls="--", label="O/P ~ mu fit")

    ax.set_xlabel("Roulette mu")
    ax.set_ylabel("Observed / possible (O/P)")
    ax.set_title(f"O/P vs Roulette mu\nPearson r={pr:.3f}, Spearman rho={sr:.3f}")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_omu_boxplots(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    data = [df["omu"].dropna().values]
    labels = ["all_desert_windows"]
    for desert in DESERT_ORDER:
        data.append(df.loc[df["desert_exemplar"] == desert, "omu"].dropna().values)
        labels.append(desert)

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    palette = desert_palette()
    for i, patch in enumerate(bp["boxes"]):
        if i == 0:
            patch.set_facecolor("lightgray")
        else:
            patch.set_facecolor(palette[labels[i]])
        patch.set_alpha(0.6)
    ax.set_ylabel("O / mu")
    ax.set_title("Supplementary O/mu distributions")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    overall_mean_omu = float(df["omu"].mean())
    rows = []
    for desert in DESERT_ORDER:
        sub = df[df["desert_exemplar"] == desert]
        rows.append(
            {
                "desert_id": desert,
                "n_windows": int(len(sub)),
                "mean_omu": float(sub["omu"].mean()),
                "median_omu": float(sub["omu"].median()),
                "omu_percentile": float((sub["omu"].mean() > df["omu"]).mean() * 100.0),
                "mean_op": float(sub["op"].mean()),
                "median_op": float(sub["op"].median()),
                "mean_mu": float(sub["mu"].mean()),
                "mean_op_residual": float(sub["op_residual"].mean()),
                "median_op_residual": float(sub["op_residual"].median()),
                "mean_z_adj": float(sub["z_adj"].mean()),
                "median_z_adj": float(sub["z_adj"].median()),
                "mean_z_unadj": float(sub["z_unadj"].mean()),
                "median_z_unadj": float(sub["z_unadj"].median()),
                "mean_z_roulette_dev": float(sub["z_roulette_dev"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_rank_concordance(summary: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(summary["mean_rank_mu"], summary["mean_rank_exp_unadj"], s=28, alpha=0.7, color="tab:gray")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    for _, row in summary[summary["desert_id"].isin(DESERT_ORDER)].iterrows():
        ax.annotate(row["desert_id"], (row["mean_rank_mu"], row["mean_rank_exp_unadj"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean mu percentile rank")
    ax.set_ylabel("Mean expected_unadj percentile rank")
    ax.set_title("Rank-based model concordance (per desert)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_glm_scatter(df: pd.DataFrame, out_path: str) -> None:
    bg = df.sample(min(120_000, len(df)), random_state=3)
    pr, _ = _safe_corr(df["z_roulette_dev"], df["z_unadj"])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(bg["z_unadj"], bg["z_roulette_dev"], s=2, alpha=0.12, color="lightgray", label="desert windows")
    palette = desert_palette()
    for desert in DESERT_ORDER:
        sub = df[df["desert_exemplar"] == desert]
        ax.scatter(sub["z_unadj"], sub["z_roulette_dev"], s=10, alpha=0.6, color=palette[desert], label=desert)

    lo = float(np.nanpercentile(df["z_unadj"], 0.5))
    hi = float(np.nanpercentile(df["z_unadj"], 99.5))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("z_unadj")
    ax.set_ylabel("z_roulette_dev")
    ax.set_title(f"Roulette deviance residuals vs z_unadj\nPearson r={pr:.3f}")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_glm_distributions(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-10, 10, 201)
    ax.hist(df["z_unadj"].dropna(), bins=bins, alpha=0.5, density=True, label="z_unadj")
    ax.hist(df["z_roulette_dev"].dropna(), bins=bins, alpha=0.5, density=True, label="z_roulette_dev")
    ax.set_xlabel("Score")
    ax.set_ylabel("density")
    ax.set_title("Distribution overlay: z_unadj vs z_roulette_dev")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_spatial_profiles(df: pd.DataFrame) -> None:
    for name, (chrom, start, end, note) in DESERTS.items():
        sub = df[df["desert_exemplar"] == name].sort_values("start")
        pos = (sub["start"] + sub["end"]) / 2
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]},
        )

        ax = axes[0]
        ax.plot(pos, sub["z_adj"], lw=0.9, color="tab:blue", label="z_adj")
        ax.plot(pos, sub["z_unadj"], lw=0.9, color="tab:orange", label="z_unadj")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Gnocchi z")
        ax.set_title(f"{name}  {chrom}:{start:,}–{end:,}  ({note})")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(pos, sub["op_residual"], lw=0.9, color="tab:green", label="O/P residual")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("O/P residual")
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.plot(pos, sub["z_roulette_dev"], lw=0.9, color="tab:red", label="z_roulette_dev")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Roulette dev")
        ax.set_xlabel(f"{chrom} position")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f} Mb"))

        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"roulette_spatial_profile_{name}.png"), dpi=150)
        plt.close(fig)


def _save_fleet_scatter(summary: pd.DataFrame, out_path: str) -> None:
    colors = {
        "Inflated by adjustment": "tab:red",
        "Deflated by adjustment": "tab:blue",
        "Sign-flip": "tab:purple",
        "Neutral": "tab:green",
        "Intermediate": "tab:gray",
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    for cat, sub in summary.groupby("category"):
        ax.scatter(
            sub["mean_z_unadj"],
            sub["mean_z_roulette_dev"],
            s=30,
            alpha=0.7,
            color=colors.get(cat, "tab:gray"),
            label=f"{cat} (n={len(sub)})",
        )
    lo = min(summary["mean_z_unadj"].min(), summary["mean_z_roulette_dev"].min()) - 0.5
    hi = max(summary["mean_z_unadj"].max(), summary["mean_z_roulette_dev"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    for _, row in summary[summary["desert_id"].isin(DESERT_ORDER)].iterrows():
        ax.annotate(row["desert_id"], (row["mean_z_unadj"], row["mean_z_roulette_dev"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean z_unadj")
    ax.set_ylabel("Mean z_roulette_dev")
    ax.set_title("Fleet concordance: Roulette deviance vs z_unadj")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_fleet_op_residual(summary: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mean_z_adj"], summary["mean_op_residual"], s=28, alpha=0.7, color="tab:gray")
    ax.axhline(0, color="black", lw=0.8)
    for _, row in summary[summary["desert_id"].isin(DESERT_ORDER)].iterrows():
        ax.annotate(row["desert_id"], (row["mean_z_adj"], row["mean_op_residual"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean z_adj")
    ax.set_ylabel("Mean O/P regression residual")
    ax.set_title("Fleet O/P residual vs adjusted Gnocchi score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Roulette per-window rates ...")
    roulette = pd.read_csv(ROULETTE_PATH, sep="\t", compression="gzip")
    roulette["mu"] = pd.to_numeric(roulette["mu"], errors="coerce")
    roulette["n_sites"] = pd.to_numeric(roulette["n_sites"], errors="coerce")
    footer = roulette["element_id"].isna() | (roulette["element_id"] == "NA")
    if footer.any():
        print(f"  filtering {int(footer.sum())} footer row(s)")
    roulette = roulette[~footer].copy()

    print("Loading merged Gnocchi table ...")
    gn = load_gnocchi(
        usecols=[
            "chrom",
            "start",
            "end",
            "element_id",
            "possible",
            "observed",
            "expected_unadj",
            "z_adj",
            "z_unadj",
        ]
    )
    print(f"  Gnocchi rows: {len(gn):,}")
    print(f"  Roulette rows (after footer filter): {len(roulette):,}")

    print("Merging on element_id ...")
    df = gn.merge(roulette[["element_id", "mu", "n_sites"]], on="element_id", how="inner")
    print(f"  merged rows: {len(df):,}")

    valid = (
        np.isfinite(df["mu"])
        & np.isfinite(df["possible"])
        & np.isfinite(df["observed"])
        & (df["mu"] > 0)
        & (df["possible"] > 0)
    )
    before = len(df)
    df = df[valid].copy()
    print(f"  retained valid rows: {len(df):,} / {before:,}")

    print("Computing Lens A/C/D derived metrics ...")
    df["op"] = df["observed"] / df["possible"]
    df["omu"] = df["observed"] / df["mu"]
    df["mu_rank"] = df["mu"].rank(method="average", pct=True)
    df["exp_unadj_rank"] = df["expected_unadj"].rank(method="average", pct=True)

    beta1, beta0 = np.polyfit(df["mu"].values, df["op"].values, 1)
    df["op_fit"] = beta0 + beta1 * df["mu"]
    df["op_residual"] = df["op"] - df["op_fit"]

    scale = float(df["observed"].sum() / df["mu"].sum())
    intercept = float(np.log(scale))
    df["fit_poisson"] = df["mu"] * scale
    df["z_roulette_dev"] = _deviance_residual(df["observed"].values, df["fit_poisson"].values)
    pearson_chi2 = np.sum((df["observed"] - df["fit_poisson"]) ** 2 / np.clip(df["fit_poisson"], 1e-12, None))
    dispersion = float(pearson_chi2 / max(len(df) - 1, 1))
    print(f"  O/P~mu linear fit: op = {beta0:.5e} + {beta1:.5e}*mu")
    print(f"  Poisson offset intercept: {intercept:.6f}, dispersion={dispersion:.4f}")

    print("Labeling exemplar and fleet deserts ...")
    df = label_deserts(df)
    df = df.rename(columns={"desert": "desert_exemplar"})
    all_deserts = load_all_deserts()
    fleet = label_deserts_fleet(df, all_deserts)
    fleet = fleet[fleet["desert"].notna()].copy()
    print(f"  fleet-labeled rows: {len(fleet):,}")

    print("Lens B: model-concordance plot ...")
    _save_model_concordance(fleet, os.path.join(RESULTS_DIR, "roulette_model_concordance.png"))

    print("Lens A: O/P-vs-mu + O/mu summaries ...")
    _save_op_vs_mu(fleet, beta0=beta0, beta1=beta1, out_path=os.path.join(RESULTS_DIR, "roulette_op_vs_mu_scatter.png"))
    exemplar_summary = _save_omu_boxplots(fleet, os.path.join(RESULTS_DIR, "roulette_omu_exemplar_boxplots.png"))
    exemplar_summary.to_csv(os.path.join(RESULTS_DIR, "roulette_desert_summary.tsv"), sep="\t", index=False)

    print("Lens D: roulette deviance residual plots ...")
    _save_glm_scatter(fleet, os.path.join(RESULTS_DIR, "roulette_glm_scatter.png"))
    _save_glm_distributions(fleet, os.path.join(RESULTS_DIR, "roulette_glm_distributions.png"))

    print("Spatial profiles for exemplars ...")
    _save_spatial_profiles(fleet)

    print("Computing fleet summary table ...")
    rows = []
    omu_series = fleet["omu"].dropna()
    for desert_id, sub in fleet.groupby("desert", sort=False):
        mu_exp_r, _ = _safe_corr(sub["mu"], sub["expected_unadj"])
        mean_omu = float(sub["omu"].mean())
        rows.append(
            {
                "desert_id": desert_id,
                "n_windows": int(len(sub)),
                "mean_z_adj": float(sub["z_adj"].mean()),
                "mean_z_unadj": float(sub["z_unadj"].mean()),
                "mean_delta_z": float((sub["z_unadj"] - sub["z_adj"]).mean()),
                "mean_op": float(sub["op"].mean()),
                "mean_op_residual": float(sub["op_residual"].mean()),
                "mean_omu": mean_omu,
                "omu_percentile": float((mean_omu > omu_series).mean() * 100.0),
                "mean_z_roulette_dev": float(sub["z_roulette_dev"].mean()),
                "mean_rank_mu": float(sub["mu_rank"].mean()),
                "mean_rank_exp_unadj": float(sub["exp_unadj_rank"].mean()),
                "mean_rank_diff": float((sub["mu_rank"] - sub["exp_unadj_rank"]).mean()),
                "pearson_mu_expected_unadj": float(mu_exp_r),
            }
        )

    summary = pd.DataFrame(rows)
    meta = all_deserts.rename(columns={"start": "desert_start", "end": "desert_end"})[
        ["desert_id", "chrom", "desert_start", "desert_end", "panel_label"]
    ]
    summary = summary.merge(meta, on="desert_id", how="left", validate="one_to_one")
    summary["category"] = summary.apply(_category, axis=1)
    sign_match = np.sign(summary["mean_z_roulette_dev"]) == np.sign(summary["mean_z_unadj"])
    summary["concordance_flag"] = np.where(sign_match, "concordant", "discordant")
    summary = summary.sort_values(
        ["chrom", "desert_start"],
        key=lambda s: s.map(_chrom_sort_key) if s.name == "chrom" else s,
    ).reset_index(drop=True)

    print("Lens C: rank-concordance plot ...")
    _save_rank_concordance(summary, os.path.join(RESULTS_DIR, "roulette_rank_concordance.png"))

    print("Fleet-level scatters ...")
    _save_fleet_scatter(summary, os.path.join(RESULTS_DIR, "roulette_fleet_scatter.png"))
    _save_fleet_op_residual(summary, os.path.join(RESULTS_DIR, "roulette_fleet_op_residual.png"))

    out_summary = os.path.join(RESULTS_DIR, "roulette_fleet_summary.tsv")
    summary.to_csv(out_summary, sep="\t", index=False)
    print(f"  wrote {out_summary}")

    overall_mu_r, _ = _safe_corr(fleet["mu"], fleet["expected_unadj"])
    overall_mu_s, _ = _safe_spearman(fleet["mu"], fleet["expected_unadj"])
    concordance_frac = float(np.mean(summary["concordance_flag"] == "concordant"))
    discordant_n = int(np.sum(summary["concordance_flag"] == "discordant"))

    print("\n=== Key findings (summary stats) ===")
    print(f"  windows analyzed: {len(fleet):,}")
    print(f"  deserts analyzed: {summary['desert_id'].nunique():,}")
    print(f"  mu vs expected_unadj: Pearson r={overall_mu_r:.4f}, Spearman rho={overall_mu_s:.4f}")
    print(f"  Poisson offset intercept={intercept:.6f}, dispersion={dispersion:.4f}")
    print(f"  fleet sign concordance (roulette_dev vs z_unadj): {concordance_frac:.1%} ({len(summary)-discordant_n}/{len(summary)})")
    print(f"  discordant deserts: {discordant_n}")

    print("\n=== Exemplar means ===")
    exemplar_cols = ["desert_id", "mean_z_adj", "mean_z_unadj", "mean_op_residual", "mean_z_roulette_dev"]
    print(exemplar_summary[["desert_id", "mean_z_adj", "mean_z_unadj", "mean_op_residual", "mean_z_roulette_dev"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

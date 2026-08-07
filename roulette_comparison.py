#!/usr/bin/env python3
"""Roulette-vs-Gnocchi comparison using Roulette aggregated EXPECTED counts.

Roulette now provides per-1kb *expected* variant counts (`exp`), aggregated
from the Roulette relative mutation-rate model and calibrated to gnomAD v3,
plus `n_variants` (a coverage proxy: max 3000 = 1kb x 3 substitutions; lower
means Roulette data was missing for some possible substitutions).

Because actual expected counts are now available, the earlier "mu as a proxy
for expected" lenses (O/P~mu regression, rank concordance, Poisson offset) are
obsolete and removed. Instead we put Roulette expected on the same footing as
Gnocchi expected and recompute a Gnocchi-style z-score:

  1. Diploid -> haploid: Roulette rates are diploid, gnomAD is haploid, so we
     divide the Roulette per-genome expectation appropriately. Empirically the
     raw Roulette `exp` is ~0.55x the gnomAD expected; a factor of 2 (diploid)
     combined with the accessibility correction below lands it on the observed
     scale without any fitted fudge factor.
  2. Accessibility / coverage: Gnocchi counts observed variants over its QC
     *accessible* sites (`possible`, ~2670 +/- 490 of 3000), while Roulette
     `exp` is summed over `n_variants` covered substitutions (~3000). We convert
     Roulette to a per-covered-site rate and re-expand onto gnomAD's accessible
     count: exp_per_site = exp / n_variants; then multiply by `possible`.

  exp_roulette = DIPLOID_FACTOR * (exp / n_variants) * possible

Then z_roulette uses the same signed-chi formula as Gnocchi, and we compare
desert anomalies across the two mutation models.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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


# Diploid-calibrated Roulette expected counts (default); needs a factor of 2 to
# reach the haploid gnomAD basis.
ROULETTE_PATH_DIPLOID = os.path.join("data", "roulette_gd_relative_mu_agg_1kb.tsv.bgz")
# Haploid-calibrated Roulette expected counts; already on the haploid basis, so
# no factor of 2 is applied.
ROULETTE_PATH_HAPLOID = os.path.join("data", "roulette_v3_hap_gd_relative_mu_agg_1kb.tsv.bgz")

# Roulette enumerates 3 alternate alleles per base, so a fully-covered 1kb
# window has 1000 * 3 = 3000 possible substitutions.
MAX_COVERAGE = 3000

# Roulette rates in the default file are diploid; gnomAD/Gnocchi work on a
# haploid basis, so the diploid expected is multiplied by 2. The --haploid file
# is already haploid and uses a factor of 1.
DIPLOID_FACTOR = 2.0
HAPLOID_FACTOR = 1.0


def compute_gnocchi_z(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Signed chi deviation, identical to the Gnocchi z definition."""
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    exp = np.clip(exp, 1e-12, None)
    chi2 = (obs - exp) ** 2 / exp
    return np.where(obs < exp, np.sqrt(chi2), -np.sqrt(chi2))


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def _category(row: pd.Series) -> str:
    if np.sign(row["mean_z_adj"]) != np.sign(row["mean_z_unadj"]):
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


# ── Plots ────────────────────────────────────────────────────────────────────
def _save_zscore_distributions(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(-10, 10, 161)
    ax.hist(df["z_adj"].dropna(), bins=bins, density=True, histtype="step", lw=1.4, color="tab:blue", label="z_adj (gnomAD, adjusted)")
    ax.hist(df["z_unadj"].dropna(), bins=bins, density=True, histtype="step", lw=1.4, color="tab:orange", label="z_unadj (gnomAD, unadjusted)")
    ax.hist(df["z_roulette"].dropna(), bins=bins, density=True, histtype="step", lw=1.4, color="tab:red", label="z_roulette")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.set_title("Desert-window z-score distributions across mutation models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_scatter_vs_gnocchi(df: pd.DataFrame, out_path: str) -> None:
    bg = df.sample(min(120_000, len(df)), random_state=0)
    palette = desert_palette()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    specs = [("z_unadj", "z_unadj (gnomAD, unadjusted)"), ("z_adj", "z_adj (gnomAD, adjusted)")]
    for ax, (col, label) in zip(axes, specs):
        r = _safe_corr(df["z_roulette"], df[col])
        ax.scatter(bg[col], bg["z_roulette"], s=2, alpha=0.12, color="lightgray", label="desert windows")
        for desert in DESERT_ORDER:
            sub = df[df["desert_exemplar"] == desert]
            ax.scatter(sub[col], sub["z_roulette"], s=10, alpha=0.6, color=palette[desert], label=desert)
        lo = float(np.nanpercentile(df[col], 0.5))
        hi = float(np.nanpercentile(df[col], 99.5))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel("z_roulette")
        ax.set_title(f"z_roulette vs {col}  (Pearson r={r:.3f})")
        ax.legend(fontsize=7, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_oe_boxplots(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    groups = ["all_desert_windows"] + DESERT_ORDER
    palette = desert_palette()
    width = 0.27
    offsets = {"oe_adj": -width, "oe_unadj": 0.0, "oe_roulette": width}
    colors = {"oe_adj": "tab:blue", "oe_unadj": "tab:orange", "oe_roulette": "tab:red"}
    positions = np.arange(len(groups))
    for col, off in offsets.items():
        data = [df[col].dropna().values]
        for desert in DESERT_ORDER:
            data.append(df.loc[df["desert_exemplar"] == desert, col].dropna().values)
        bp = ax.boxplot(
            data,
            positions=positions + off,
            widths=width * 0.9,
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[col])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
    ax.axhline(1.0, color="grey", lw=0.7, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Observed / Expected")
    ax.set_ylim(0, 2.0)
    handles = [plt.Line2D([0], [0], color=colors[c], lw=6, alpha=0.55) for c in colors]
    ax.legend(handles, ["O/E adjusted", "O/E unadjusted", "O/E roulette"], fontsize=8)
    ax.set_title("O/E by mutation model (deserts vs all-desert background)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_spatial_profiles(df: pd.DataFrame) -> None:
    for name, (chrom, start, end, note) in DESERTS.items():
        sub = df[df["desert_exemplar"] == name].sort_values("start")
        if sub.empty:
            continue
        pos = (sub["start"] + sub["end"]) / 2
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2]},
        )

        ax = axes[0]
        ax.plot(pos, sub["z_adj"], lw=0.9, color="tab:blue", label="z_adj")
        ax.plot(pos, sub["z_unadj"], lw=0.9, color="tab:orange", label="z_unadj")
        ax.plot(pos, sub["z_roulette"], lw=0.9, color="tab:red", label="z_roulette")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("z-score")
        ax.set_title(f"{name}  {chrom}:{start:,}-{end:,}  ({note})")
        ax.legend(fontsize=8, ncol=3)

        ax = axes[1]
        ax.plot(pos, sub["oe_unadj"], lw=0.9, color="tab:orange", label="O/E unadjusted")
        ax.plot(pos, sub["oe_roulette"], lw=0.9, color="tab:red", label="O/E roulette")
        ax.axhline(1.0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("O/E")
        ax.set_xlabel(f"{chrom} position")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f} Mb"))
        ax.legend(fontsize=8, ncol=2)

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
        ax.scatter(sub["mean_z_unadj"], sub["mean_z_roulette"], s=30, alpha=0.7,
                   color=colors.get(cat, "tab:gray"), label=f"{cat} (n={len(sub)})")
    lo = min(summary["mean_z_unadj"].min(), summary["mean_z_roulette"].min()) - 0.5
    hi = max(summary["mean_z_unadj"].max(), summary["mean_z_roulette"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    for _, row in summary[summary["desert_id"].isin(DESERT_ORDER)].iterrows():
        ax.annotate(row["desert_id"], (row["mean_z_unadj"], row["mean_z_roulette"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean z_unadj (gnomAD)")
    ax.set_ylabel("Mean z_roulette")
    ax.set_title("Fleet concordance: Roulette vs gnomAD constraint")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--haploid",
        action="store_true",
        help="Use the haploid-calibrated Roulette file and skip the diploid x2 factor.",
    )
    args = parser.parse_args()

    if args.haploid:
        roulette_path = ROULETTE_PATH_HAPLOID
        scale_factor = HAPLOID_FACTOR
        basis = "haploid"
    else:
        roulette_path = ROULETTE_PATH_DIPLOID
        scale_factor = DIPLOID_FACTOR
        basis = "diploid"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Roulette basis: {basis}  (file={roulette_path}, scale_factor={scale_factor})")
    print("Loading Roulette aggregated expected counts ...")
    roulette = pd.read_csv(roulette_path, sep="\t", compression="gzip")
    numeric_cols = ["exp", "n_variants"]
    if "mu" in roulette.columns:
        numeric_cols.append("mu")
    for col in numeric_cols:
        roulette[col] = pd.to_numeric(roulette[col], errors="coerce")
    footer = roulette["element_id"].isna() | (roulette["element_id"].astype(str) == "NA")
    if footer.any():
        print(f"  filtering {int(footer.sum())} footer/total row(s)")
    roulette = roulette[~footer].copy()
    roulette = roulette.rename(columns={"exp": "exp_roulette_raw", "n_variants": "n_variants"})

    print("Loading merged Gnocchi table ...")
    gn = load_gnocchi(
        usecols=[
            "chrom", "start", "end", "element_id",
            "possible", "observed", "expected", "expected_unadj",
            "z_adj", "z_unadj",
        ]
    )
    print(f"  Gnocchi rows: {len(gn):,}")
    print(f"  Roulette rows (after footer filter): {len(roulette):,}")

    print("Merging on element_id ...")
    merge_cols = ["element_id", "exp_roulette_raw", "n_variants"]
    if "mu" in roulette.columns:
        merge_cols.append("mu")
    df = gn.merge(roulette[merge_cols], on="element_id", how="inner")
    if "mu" in df.columns:
        df = df.drop(columns=["mu"])
    print(f"  merged rows: {len(df):,}")

    valid = (
        np.isfinite(df["exp_roulette_raw"]) & (df["exp_roulette_raw"] > 0)
        & np.isfinite(df["n_variants"]) & (df["n_variants"] > 0)
        & np.isfinite(df["possible"]) & (df["possible"] > 0)
        & np.isfinite(df["observed"])
        & np.isfinite(df["expected_unadj"]) & (df["expected_unadj"] > 0)
    )
    before = len(df)
    df = df[valid].copy()
    print(f"  retained valid rows: {len(df):,} / {before:,}")

    n_low_cov = int((df["n_variants"] < MAX_COVERAGE).sum())
    print(f"  windows with incomplete Roulette coverage (n_variants<{MAX_COVERAGE}): {n_low_cov:,}")

    # ── Build comparable Roulette expected ───────────────────────────────────
    print(f"Building comparable Roulette expected ({basis} x{scale_factor:.0f} + accessibility) ...")
    df["coverage"] = df["n_variants"] / MAX_COVERAGE
    exp_per_site = df["exp_roulette_raw"] / df["n_variants"]
    df["exp_roulette"] = scale_factor * exp_per_site * df["possible"]

    df["oe_adj"] = df["observed"] / df["expected"]
    df["oe_unadj"] = df["observed"] / df["expected_unadj"]
    df["oe_roulette"] = df["observed"] / df["exp_roulette"]
    df["z_roulette"] = compute_gnocchi_z(df["observed"].values, df["exp_roulette"].values)
    df["delta_z"] = df["z_unadj"] - df["z_adj"]

    # ── Calibration diagnostics ──────────────────────────────────────────────
    sum_obs = float(df["observed"].sum())
    sum_unadj = float(df["expected_unadj"].sum())
    sum_raw = float(df["exp_roulette_raw"].sum())
    sum_rou = float(df["exp_roulette"].sum())
    implied_k = sum_obs / sum_rou
    print("\n=== Calibration diagnostics (desert windows) ===")
    print(f"  sum observed                 = {sum_obs:,.0f}")
    print(f"  sum expected_unadj (gnomAD)  = {sum_unadj:,.0f}   (/obs = {sum_unadj/sum_obs:.3f})")
    print(f"  sum exp_roulette_raw         = {sum_raw:,.0f}   (/obs = {sum_raw/sum_obs:.3f})")
    print(f"  sum exp_roulette (corrected) = {sum_rou:,.0f}   (/obs = {sum_rou/sum_obs:.3f})")
    print(f"  residual implied scale k (obs/corrected) = {implied_k:.4f}  "
          f"(≈1 means {basis} scaling + coverage reconcile; NOT applied, diagnostic only)")

    # ── Labeling ─────────────────────────────────────────────────────────────
    print("\nLabeling exemplar and fleet deserts ...")
    df = label_deserts(df)
    df = df.rename(columns={"desert": "desert_exemplar"})
    all_deserts = load_all_deserts()
    fleet = label_deserts_fleet(df, all_deserts)
    fleet = fleet[fleet["desert"].notna()].copy()
    print(f"  fleet-labeled rows: {len(fleet):,}  across {fleet['desert'].nunique():,} deserts")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("Generating comparison plots ...")
    _save_zscore_distributions(fleet, os.path.join(RESULTS_DIR, "roulette_zscore_distributions.png"))
    _save_scatter_vs_gnocchi(fleet, os.path.join(RESULTS_DIR, "roulette_scatter_vs_gnocchi.png"))
    _save_oe_boxplots(fleet, os.path.join(RESULTS_DIR, "roulette_oe_boxplots.png"))
    _save_spatial_profiles(fleet)

    # ── Exemplar summary ─────────────────────────────────────────────────────
    print("Building exemplar summary ...")
    ex_rows = []
    for name in DESERT_ORDER:
        sub = fleet[fleet["desert_exemplar"] == name]
        if sub.empty:
            continue
        ex_rows.append({
            "desert_id": name,
            "note": DESERTS[name][3],
            "n_windows": int(len(sub)),
            "mean_coverage": float(sub["coverage"].mean()),
            "mean_oe_unadj": float(sub["oe_unadj"].mean()),
            "mean_oe_roulette": float(sub["oe_roulette"].mean()),
            "mean_z_adj": float(sub["z_adj"].mean()),
            "median_z_adj": float(sub["z_adj"].median()),
            "mean_z_unadj": float(sub["z_unadj"].mean()),
            "median_z_unadj": float(sub["z_unadj"].median()),
            "mean_z_roulette": float(sub["z_roulette"].mean()),
            "median_z_roulette": float(sub["z_roulette"].median()),
            "corr_z_roulette_unadj": _safe_corr(sub["z_roulette"], sub["z_unadj"]),
        })
    exemplar_summary = pd.DataFrame(ex_rows)
    exemplar_summary.to_csv(os.path.join(RESULTS_DIR, "roulette_desert_summary.tsv"), sep="\t", index=False)

    # ── Fleet summary ────────────────────────────────────────────────────────
    print("Building fleet summary ...")
    rows = []
    for desert_id, sub in fleet.groupby("desert", sort=False):
        rows.append({
            "desert_id": desert_id,
            "n_windows": int(len(sub)),
            "mean_coverage": float(sub["coverage"].mean()),
            "mean_z_adj": float(sub["z_adj"].mean()),
            "mean_z_unadj": float(sub["z_unadj"].mean()),
            "mean_delta_z": float((sub["z_unadj"] - sub["z_adj"]).mean()),
            "mean_z_roulette": float(sub["z_roulette"].mean()),
            "median_z_roulette": float(sub["z_roulette"].median()),
            "mean_oe_unadj": float(sub["oe_unadj"].mean()),
            "mean_oe_roulette": float(sub["oe_roulette"].mean()),
            "corr_z_roulette_unadj": _safe_corr(sub["z_roulette"], sub["z_unadj"]),
        })
    summary = pd.DataFrame(rows)
    meta = all_deserts.rename(columns={"start": "desert_start", "end": "desert_end"})[
        ["desert_id", "chrom", "desert_start", "desert_end", "panel_label"]
    ]
    summary = summary.merge(meta, on="desert_id", how="left", validate="one_to_one")
    summary["category"] = summary.apply(_category, axis=1)
    sign_match = np.sign(summary["mean_z_roulette"]) == np.sign(summary["mean_z_unadj"])
    summary["concordance_flag"] = np.where(sign_match, "concordant", "discordant")
    summary = summary.sort_values(
        ["chrom", "desert_start"],
        key=lambda s: s.map(_chrom_sort_key) if s.name == "chrom" else s,
    ).reset_index(drop=True)

    _save_fleet_scatter(summary, os.path.join(RESULTS_DIR, "roulette_fleet_scatter.png"))
    out_summary = os.path.join(RESULTS_DIR, "roulette_fleet_summary.tsv")
    summary.to_csv(out_summary, sep="\t", index=False)
    print(f"  wrote {out_summary}")

    # ── Key findings ─────────────────────────────────────────────────────────
    overall_r_unadj = _safe_corr(fleet["z_roulette"], fleet["z_unadj"])
    overall_r_adj = _safe_corr(fleet["z_roulette"], fleet["z_adj"])
    concordance_frac = float(np.mean(summary["concordance_flag"] == "concordant"))
    discordant_n = int(np.sum(summary["concordance_flag"] == "discordant"))

    print("\n=== Key findings ===")
    print(f"  windows analyzed: {len(fleet):,}  | deserts: {summary['desert_id'].nunique():,}")
    print(f"  z_roulette vs z_unadj : Pearson r={overall_r_unadj:.4f}")
    print(f"  z_roulette vs z_adj   : Pearson r={overall_r_adj:.4f}")
    print(f"  fleet sign concordance (z_roulette vs z_unadj): {concordance_frac:.1%} "
          f"({len(summary)-discordant_n}/{len(summary)}); discordant={discordant_n}")

    print("\n=== Exemplar means ===")
    cols = ["desert_id", "mean_z_adj", "mean_z_unadj", "mean_z_roulette", "mean_oe_roulette", "corr_z_roulette_unadj"]
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(exemplar_summary[cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

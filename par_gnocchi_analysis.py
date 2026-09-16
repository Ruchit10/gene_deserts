#!/usr/bin/env python3
"""PAR1/PAR2 analog of the gene-desert Gnocchi diagnostics.

The 5 hand-curated gene deserts analysed elsewhere in this repo are all
autosomal by construction. chrX's pseudoautosomal regions never contain a
>=500kb gene-free interval (checked directly against GENCODE v39 -- PAR1's
largest gene-free gap is ~450kb, PAR2 has essentially none), so no
"gene desert" analog exists there. Instead this script treats PAR1 and PAR2
as whole regions, and additionally zooms into chrX:320,649-1,262,723 -- the
most gene-desert-like stretch of PAR1, flanked by gene-free gaps on both
sides -- which spans three genes: PPP2R3B, SHOX, and CRLF2.

Unlike the autosomal analyses, this uses `data/gnocchi_1kb_chrX_par_public.txt.gz`
(gnomAD's separate public PAR release) instead of
`constraint_z_genome_1kb.qc.download.txt.gz`. z_unadj is recomputed exactly
as in `unadjusted_gnocchi_analysis.py`, using `load_expected_unadj()` for the
matching chrX rows (same element_ids, 1:1).

z_adj comes from `results/par_regional_adjustment.tsv.gz`, produced by
`regional_corrections/compute_par_rr.py`. gnomAD's released PAR scores were
never regionally adjusted -- the published feature matrix is autosome-only, so
every chrX window hit the pipeline's `rr = 1` fallback and the shipped `gnocchi`
column equals the unadjusted score. That adjustment is now computed from a
PAR-specific feature matrix, so `delta_z` here means "effect of the regional
correction" (repo convention: z_unadj - z_adj).

Read every number below with one caveat: only 7 of the 13 regional features
have usable data in PAR. `recomb_male` -- selected in 31 of 32 contexts -- is
among the 6 masked, yet obligate male recombination is PAR1's defining feature.
This correction covers sequence and annotation composition, not recombination.
Measured on autosomes, masking those 6 retains r=0.85 with the full-feature
adjustment and biases expected by about +1.7%.

Not attempted here -- data unavailable for chrX (see README): feature
profiling / correlations / ridge decomposition / LOFO (Analyses A, D, E, F
all require the autosome-only 52-column matrix, and the PAR matrix populates
only the 7 unmasked features), mappability diagnostics
(`gnocchi.windows.mq.lcr.segdup.stats.tsv.gz` is autosome-only), and roulette
mutation-rate comparisons (chrX/Y unavailable there too).

Outputs (all in results/, `par_` prefixed):
  par_summary.tsv              - mean/median z_adj, z_unadj, delta_z, O/E for
                                  PAR1, PAR2, the highlighted interval, and
                                  each of the 3 genes
  par_gene_window_stats.tsv    - windows overlapping each gene vs the rest of
                                  the highlighted interval
  par_histograms.png           - z_adj vs z_unadj density, PAR1 vs PAR2
  par_scatter_adj_vs_unadj.png - z_adj vs z_unadj scatter, colored by region
  par_oe_boxplots.png          - O/E boxplots, adjusted vs unadjusted
  par1_spatial_profile.png     - full-PAR1 z trace + delta_z, highlight shaded
  par1_highlight_zoom.png      - zoomed trace + gene track for the 3 genes
  par2_spatial_profile.png     - full-PAR2 z trace + delta_z
  par1_acf.png / par2_acf.png  - spatial ACF vs autosomal genome-wide background
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from utils.desert_utils import DATA_DIR, RESULTS_DIR, load_expected_unadj, load_gnocchi
from distributions import z_stats
from spatial_acf import MAX_LAG, compute_acf, prepare_chrom_arrays, sample_background_acf

PAR_GNOCCHI_TABLE = os.path.join(DATA_DIR, "gnocchi_1kb_chrX_par_public.txt.gz")
PAR_ADJUSTMENT_TABLE = os.path.join(RESULTS_DIR, "par_regional_adjustment.tsv.gz")

# hg38 pseudoautosomal region boundaries
PAR_REGIONS: dict[str, tuple[str, int, int, str]] = {
    "PAR1": ("chrX", 10_001, 2_781_479, "pseudoautosomal region 1"),
    "PAR2": ("chrX", 155_701_383, 156_030_895, "pseudoautosomal region 2"),
}
PAR_ORDER: list[str] = list(PAR_REGIONS.keys())
PAR_PALETTE: dict[str, str] = {"PAR1": "tab:blue", "PAR2": "tab:green"}

# Most gene-desert-like stretch of PAR1: gene-free gaps flank PPP2R3B, SHOX,
# and CRLF2 on both sides (largest single gap in this span is ~450kb).
HIGHLIGHT_REGION: tuple[str, int, int, str] = (
    "chrX", 320_649, 1_262_723, "PPP2R3B-SHOX-CRLF2 interval",
)

# GENCODE v39 gene-level coordinates (0-based start, matching Gnocchi windows)
GENES: dict[str, tuple[str, int, int, str]] = {
    "PPP2R3B": ("chrX", 333_932, 386_955, "-"),
    "SHOX":    ("chrX", 624_343, 659_411, "+"),
    "CRLF2":   ("chrX", 1_187_548, 1_212_723, "-"),
}
GENE_ORDER: list[str] = list(GENES.keys())
GENE_COLORS: dict[str, str] = {"PPP2R3B": "tab:red", "SHOX": "tab:purple", "CRLF2": "tab:orange"}


def compute_gnocchi_z(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Same chi-square-signed z used in unadjusted_gnocchi_analysis.py."""
    chi2 = (obs - exp) ** 2 / exp
    return np.where(obs < exp, np.sqrt(chi2), -np.sqrt(chi2))


def load_par_gnocchi() -> pd.DataFrame:
    """Load chrX PAR windows and recompute z_unadj.

    Mirrors unadjusted_gnocchi_analysis.py's merge/recompute, but against
    gnomAD's separate PAR-specific public release (which already carries an
    adjusted z-score under the `gnocchi` column) instead of the autosome-only
    constraint_z_genome_1kb table.
    """
    print("Loading chrX PAR Gnocchi table ...")
    par = pd.read_csv(PAR_GNOCCHI_TABLE, sep="\t")
    n_total = len(par)
    par = par[par["pass_qc"]].copy()
    print(f"  {n_total:,} PAR windows loaded, {len(par):,} pass QC")

    print("Loading unadjusted expected sums (chrX rows) ...")
    unadj = load_expected_unadj()
    unadj = unadj[unadj["element_id"].str.startswith("chrX-")]
    print(f"  {len(unadj):,} chrX windows loaded")

    df = par.merge(unadj, on="element_id", how="inner")
    print(f"  {len(df):,} windows after inner join")

    print("Loading regional adjustment ...")
    if not os.path.exists(PAR_ADJUSTMENT_TABLE):
        raise FileNotFoundError(
            f"{PAR_ADJUSTMENT_TABLE} not found -- run "
            "regional_corrections/compute_par_rr.py first."
        )
    adj = pd.read_csv(PAR_ADJUSTMENT_TABLE, sep="\t",
                      usecols=["element_id", "expected_adj", "rr_mean", "z_adj_new"])
    df = df.merge(adj, on="element_id", how="inner")
    print(f"  {len(df):,} windows with a regional adjustment "
          f"(mean rr {df['rr_mean'].mean():.4f})")

    # The shipped `gnocchi`/`expected` columns are the unadjusted scores, since
    # PAR hit the pipeline's rr = 1 fallback; z_adj now comes from our own
    # PAR-specific regional adjustment instead.
    df["z_adj"] = df["z_adj_new"]
    df["z_unadj"] = compute_gnocchi_z(df["observed"].values, df["expected_unadj"].values)
    df["delta_z"] = df["z_unadj"] - df["z_adj"]
    df["oe_adj"] = df["observed"] / df["expected_adj"]
    df["oe_unadj"] = df["observed"] / df["expected_unadj"]
    return df


def label_par_regions(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag each window with PAR1/PAR2 (else NaN) -- same pattern as label_deserts."""
    out = frame.copy()
    out["region"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    for name, (chrom, start, end, _) in PAR_REGIONS.items():
        mask = (out["chrom"] == chrom) & (out["start"] >= start) & (out["end"] <= end)
        out.loc[mask, "region"] = name
    return out


def _overlap_mask(frame: pd.DataFrame, chrom: str, start: int, end: int) -> pd.Series:
    return (frame["chrom"] == chrom) & (frame["start"] < end) & (frame["end"] > start)


def _summary_row(scope: str, name: str, note: str, sub: pd.DataFrame) -> dict:
    adj = z_stats(sub["z_adj"])
    unadj = z_stats(sub["z_unadj"])
    return {
        "scope": scope,
        "name": name,
        "note": note,
        "n_windows": len(sub),
        **{f"adj_{k}": v for k, v in adj.items() if k != "n"},
        **{f"unadj_{k}": v for k, v in unadj.items() if k != "n"},
        "mean_delta_z": float(sub["delta_z"].mean()) if len(sub) else np.nan,
        "mean_oe_adj": float(sub["oe_adj"].mean()) if len(sub) else np.nan,
        "mean_oe_unadj": float(sub["oe_unadj"].mean()) if len(sub) else np.nan,
    }


def build_summary(df: pd.DataFrame, highlight_mask: pd.Series) -> pd.DataFrame:
    rows = [_summary_row("par_all", "PAR1+PAR2", "", df)]
    for name in PAR_ORDER:
        _, _, _, note = PAR_REGIONS[name]
        rows.append(_summary_row("region", name, note, df[df["region"] == name]))
    _, _, _, hnote = HIGHLIGHT_REGION
    rows.append(_summary_row("highlight", "PPP2R3B-SHOX-CRLF2", hnote, df[highlight_mask]))
    for gene in GENE_ORDER:
        chrom, gs, ge, strand = GENES[gene]
        rows.append(_summary_row("gene", gene, f"strand {strand}", df[_overlap_mask(df, chrom, gs, ge)]))
    return pd.DataFrame(rows)


def build_gene_window_stats(df: pd.DataFrame, highlight_mask: pd.Series) -> pd.DataFrame:
    """Windows overlapping each gene vs the rest of the highlighted interval."""
    highlight_df = df[highlight_mask]
    rows = []
    for gene in GENE_ORDER:
        chrom, gs, ge, _ = GENES[gene]
        in_gene = _overlap_mask(highlight_df, chrom, gs, ge)
        for group_name, sub in [("in_gene", highlight_df[in_gene]), ("rest_of_interval", highlight_df[~in_gene])]:
            row = _summary_row("gene_window", f"{gene}:{group_name}", group_name, sub)
            row["gene"] = gene
            row["group"] = group_name
            rows.append(row)
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(len(PAR_ORDER), 1, figsize=(8, 7), sharex=True)
    bins = np.linspace(-8, 8, 65)
    for ax, name in zip(axes, PAR_ORDER):
        sub = df[df["region"] == name]
        ax.hist(sub["z_adj"], bins=bins, alpha=0.6, label="adjusted", density=True)
        ax.hist(sub["z_unadj"], bins=bins, alpha=0.6, label="unadjusted", density=True)
        ax.set_title(f"{name} ({PAR_REGIONS[name][3]}, n={len(sub):,})", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel("density")
    axes[-1].set_xlabel("Gnocchi z-score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for name in PAR_ORDER:
        sub = df[df["region"] == name]
        ax.scatter(sub["z_adj"], sub["z_unadj"], s=6, alpha=0.4,
                   color=PAR_PALETTE[name], label=name)
    lims = [-10, 10]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("z (adjusted)")
    ax.set_ylabel("z (unadjusted)")
    ax.set_title("Adjusted vs Unadjusted Gnocchi -- chrX PAR windows")
    ax.legend(markerscale=2, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_oe_boxplots(df: pd.DataFrame, highlight_mask: pd.Series, out_path: str) -> None:
    groups = PAR_ORDER + ["highlight"]
    data_by_group = {name: df[df["region"] == name] for name in PAR_ORDER}
    data_by_group["highlight"] = df[highlight_mask]
    colors = [PAR_PALETTE[n] for n in PAR_ORDER] + ["tab:gray"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for ax, col, title in [(axes[0], "oe_adj", "O/E (adjusted)"),
                            (axes[1], "oe_unadj", "O/E (unadjusted)")]:
        data = [data_by_group[g][col].dropna().values for g in groups]
        bp = ax.boxplot(data, labels=groups, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(1.0, color="k", ls="--", lw=0.7)
        ax.set_title(title)
        ax.set_ylabel("O/E ratio")
        ax.set_ylim(0, 2.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _gene_track_panel(ax: plt.Axes, region_start: int, region_end: int) -> None:
    for i, gene in enumerate(GENE_ORDER):
        chrom, gs, ge, strand = GENES[gene]
        ax.broken_barh([(gs, ge - gs)], (i - 0.4, 0.8),
                       facecolors=GENE_COLORS[gene], edgecolors="none", alpha=0.85)
        ax.text((gs + ge) / 2, i, f"{gene} ({strand})", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(range(len(GENE_ORDER)))
    ax.set_yticklabels([])
    ax.set_ylim(-0.8, len(GENE_ORDER) - 0.2)
    ax.set_xlim(region_start, region_end)
    ax.set_ylabel("genes", fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}"))
    ax.set_xlabel("chrX position (Mb)", fontsize=9)


def plot_region_profile(sub: pd.DataFrame, chrom: str, start: int, end: int,
                         title: str, out_path: str,
                         shade_region: tuple[int, int] | None = None) -> None:
    """2-panel spatial profile: z_adj/z_unadj trace + delta_z fill."""
    sub = sub.sort_values("start")
    pos = (sub["start"] + sub["end"]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(pos, sub["z_adj"], lw=0.9, color="tab:blue", label="z adjusted")
    ax.plot(pos, sub["z_unadj"], lw=0.9, color="tab:orange", label="z unadjusted")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("Gnocchi z-score")
    ax.set_title(f"{title}  {chrom}:{start:,}-{end:,}", fontsize=11)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    delta = sub["delta_z"]
    ax2.fill_between(pos, delta, 0, where=(delta >= 0), color="tab:orange", alpha=0.4, label="delta_z > 0")
    ax2.fill_between(pos, delta, 0, where=(delta < 0), color="tab:blue", alpha=0.4, label="delta_z < 0")
    ax2.axhline(0, color="grey", lw=0.5, ls="--")
    ax2.set_ylabel("delta_z (unadj - adj)")
    ax2.set_xlabel(f"{chrom} position")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f} Mb"))
    ax2.legend(fontsize=8)

    if shade_region is not None:
        for a in axes:
            a.axvspan(shade_region[0], shade_region[1], color="tab:gray", alpha=0.15, lw=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_highlight_zoom(sub: pd.DataFrame, out_path: str) -> None:
    chrom, start, end, note = HIGHLIGHT_REGION
    sub = sub.sort_values("start")
    pos = (sub["start"] + sub["end"]) / 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 1]})
    ax = axes[0]
    ax.plot(pos, sub["z_adj"], lw=1.0, color="tab:blue", label="z adjusted")
    ax.plot(pos, sub["z_unadj"], lw=1.0, color="tab:orange", label="z unadjusted")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("Gnocchi z-score")
    ax.set_title(f"{note}  {chrom}:{start:,}-{end:,}", fontsize=11)
    ax.legend(fontsize=9)
    for g in GENE_ORDER:
        _, gs, ge, _ = GENES[g]
        ax.axvspan(gs, ge, color=GENE_COLORS[g], alpha=0.12, lw=0)

    ax2 = axes[1]
    delta = sub["delta_z"]
    ax2.fill_between(pos, delta, 0, where=(delta >= 0), color="tab:orange", alpha=0.4, label="delta_z > 0")
    ax2.fill_between(pos, delta, 0, where=(delta < 0), color="tab:blue", alpha=0.4, label="delta_z < 0")
    ax2.axhline(0, color="grey", lw=0.5, ls="--")
    ax2.set_ylabel("delta_z")
    ax2.legend(fontsize=8)

    _gene_track_panel(axes[2], start, end)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acf(name: str, sub: pd.DataFrame, chrom_arrays: dict, out_path: str) -> None:
    sub = sub.sort_values("start")
    responses = ["z_adj", "z_unadj", "delta_z"]
    lag_kb = np.arange(MAX_LAG + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, resp in zip(axes, responses):
        a = compute_acf(sub[resp].values)
        bg = sample_background_acf(chrom_arrays, length=len(sub), col=resp)
        bg_mean = np.nanmean(bg, axis=0)
        bg_lo = np.nanpercentile(bg, 2.5, axis=0)
        bg_hi = np.nanpercentile(bg, 97.5, axis=0)
        ax.fill_between(lag_kb, bg_lo, bg_hi, color="lightgray", alpha=0.6, label="autosomal background 95%")
        ax.plot(lag_kb, bg_mean, color="gray", lw=0.8, label="background mean")
        ax.plot(lag_kb, a, color="tab:red", lw=1.2, label=f"{name} ACF")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"ACF({resp}) -- {name}")
        ax.set_xlabel("lag (kb)")
        if ax is axes[0]:
            ax.set_ylabel("autocorrelation")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_par_gnocchi()
    df = label_par_regions(df)
    hchrom, hstart, hend, _ = HIGHLIGHT_REGION
    highlight_mask = _overlap_mask(df, hchrom, hstart, hend)

    print("\n=== PAR summary ===")
    summary = build_summary(df, highlight_mask)
    summary_path = os.path.join(RESULTS_DIR, "par_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                            "display.float_format", "{:,.3f}".format):
        print(summary.to_string(index=False))
    print(f"  wrote {summary_path}")

    gene_stats = build_gene_window_stats(df, highlight_mask)
    gene_stats_path = os.path.join(RESULTS_DIR, "par_gene_window_stats.tsv")
    gene_stats.to_csv(gene_stats_path, sep="\t", index=False)
    print(f"  wrote {gene_stats_path}")

    print("\nGenerating region-level plots ...")
    plot_histograms(df, os.path.join(RESULTS_DIR, "par_histograms.png"))
    plot_scatter(df, os.path.join(RESULTS_DIR, "par_scatter_adj_vs_unadj.png"))
    plot_oe_boxplots(df, highlight_mask, os.path.join(RESULTS_DIR, "par_oe_boxplots.png"))
    print("  wrote par_histograms.png, par_scatter_adj_vs_unadj.png, par_oe_boxplots.png")

    print("Generating spatial profiles ...")
    par1_chrom, par1_start, par1_end, _ = PAR_REGIONS["PAR1"]
    plot_region_profile(
        df[df["region"] == "PAR1"], par1_chrom, par1_start, par1_end,
        "PAR1", os.path.join(RESULTS_DIR, "par1_spatial_profile.png"),
        shade_region=(hstart, hend),
    )
    par2_chrom, par2_start, par2_end, _ = PAR_REGIONS["PAR2"]
    plot_region_profile(
        df[df["region"] == "PAR2"], par2_chrom, par2_start, par2_end,
        "PAR2", os.path.join(RESULTS_DIR, "par2_spatial_profile.png"),
    )
    plot_highlight_zoom(df[highlight_mask], os.path.join(RESULTS_DIR, "par1_highlight_zoom.png"))
    print("  wrote par1_spatial_profile.png, par2_spatial_profile.png, par1_highlight_zoom.png")

    print("\nLoading autosomal genome-wide table for ACF background ...")
    genome_bg = load_gnocchi(usecols=["chrom", "start", "end", "z_adj", "z_unadj", "delta_z"])
    chrom_arrays = prepare_chrom_arrays(genome_bg, ["z_adj", "z_unadj", "delta_z"])
    print(f"  {len(chrom_arrays)} autosomal chromosomes cached for background sampling")

    print("Computing spatial ACF (PAR1, PAR2 vs autosomal background) ...")
    plot_acf("PAR1", df[df["region"] == "PAR1"], chrom_arrays, os.path.join(RESULTS_DIR, "par1_acf.png"))
    plot_acf("PAR2", df[df["region"] == "PAR2"], chrom_arrays, os.path.join(RESULTS_DIR, "par2_acf.png"))
    print("  wrote par1_acf.png, par2_acf.png (note: PAR2 has only "
          f"{int((df['region'] == 'PAR2').sum())} windows -- limited statistical power)")

    print("\nDone.")


if __name__ == "__main__":
    main()

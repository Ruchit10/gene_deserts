#!/usr/bin/env python3
"""Reproduce Gnocchi-paper benchmark figures, extended to compare Roulette.

Ports the benchmark/validation analyses from the paper's own fig_utils.py /
efig_utils.py (github.com/atgu/gnomad_nc_constraint) against the data staged
in data/gnocchi_benchmark_data/, and extends every one of them to also
evaluate z_roulette alongside Gnocchi's z_adj/z_unadj. Any benchmark whose
required data file isn't present in data/gnocchi_benchmark_data/ is skipped
(printed, not stubbed) -- see utils/benchmark_utils.py's BENCHMARK_MANIFEST.

Deferred (no supporting data file yet): score-score correlation matrix,
UKBB trait-specific enrichment, exonic/percentile cross-mapping panels,
locus-specific track plots (PLG/IHH/DD-recurrent CNV), CNV/developmental-
delay case-control panels, APS-vs-Gnocchi, delta-PIP fine-mapping update,
enhancer-tissue-expression correlation, and the mutation-model QC panels.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from utils.desert_utils import RESULTS_DIR
from utils.benchmark_utils import (
    COMPARATOR_SCORES,
    GNOCCHI_COLORS,
    GWAS_ANN_COLORS,
    NEGATIVE_SETS,
    POSITIVE_SETS,
    REFERENCE_LINE_COLOR,
    REGULATORY_ANN_COLORS,
    Z_BIN_LABELS,
    attach_roulette_score,
    attach_roulette_to_annot,
    build_roulette_windows,
    check_benchmark_data,
    coerce_bool,
    compute_enhancer_z,
    enrichment_by_zbin,
    filter_noncoding_qc,
    load_annot_table,
    load_comparisons_table,
    load_enh_gene_roadmaplinks,
    load_enhz_loeuf_pred,
    parse_element_id,
    roc_auc_by_score,
    score_color,
    sem,
    style_axes,
    validate_enhancer_rollup,
    validate_gene_rollup_rule,
)

GNOCCHI_SCORES = ["z_adj", "z_unadj", "z_roulette"]
ALL_SCORE_COLS = GNOCCHI_SCORES + COMPARATOR_SCORES

REGULATORY_COLS = [
    "ENCODE cCRE-PLS", "ENCODE cCRE-pELS", "ENCODE cCRE-dELS",
    "ENCODE CTCF-only", "FANTOM enhancers", "Super enhancers",
]
GWAS_COLS = ["GWAS Catalog", "GWAS Catalog repl (ext)", "GWAS fine-mapping"]
GENE_SET_COLS = [
    "Haploinsufficient", "MGI essential", "OMIM dominant",
    "LOEUF constrained", "Olfactory", "LOEUF unconstrained", "LOEUF underpowered",
]


def _outpath(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"benchmark_{name}")


# ── A: score-distribution / O-E sanity checks ───────────────────────────────

def benchmark_score_distributions(annot: pd.DataFrame) -> None:
    is_coding = annot["coding_prop"] > 0
    bins = np.linspace(-10, 10, 81)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, score in zip(axes, GNOCCHI_SCORES):
        nc = annot.loc[~is_coding, score].dropna()
        cd = annot.loc[is_coding, score].dropna()
        color = score_color(score)
        ax.hist(nc, bins=bins, density=True, alpha=0.7, color=color, label="non-coding")
        ax.axvline(nc.median(), color=color, lw=1, ls="--")
        ax2 = ax.twinx()
        ax2.hist(cd, bins=bins, density=True, histtype="step", lw=1.5, color="#613969", label="coding")
        ax2.axvline(cd.median(), color="#613969", lw=1, ls="--")
        ax.set_title(score, fontsize=10)
        ax.set_xlabel("z-score")
        ax.set_ylabel("density (non-coding)", color=color)
        ax2.set_ylabel("density (coding)", color="#613969")
        style_axes(ax)
    fig.suptitle("Score distributions: coding vs non-coding windows")
    fig.tight_layout()
    out = _outpath("score_distributions.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_oe_scatter(annot: pd.DataFrame) -> None:
    sub = annot.sample(min(150_000, len(annot)), random_state=0)
    specs = [("z_adj", "expected"), ("z_roulette", "exp_roulette")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (score, exp_col) in zip(axes, specs):
        s = sub.dropna(subset=[exp_col, "observed", score])
        sc = ax.scatter(s[exp_col], s["observed"], c=s[score], cmap="RdBu_r",
                         vmin=-6, vmax=6, s=3, alpha=0.5)
        lo = float(s[exp_col].quantile(0.01))
        hi = float(s[exp_col].quantile(0.99))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlabel(f"expected ({exp_col})")
        ax.set_ylabel("observed")
        ax.set_title(f"O/E colored by {score}")
        fig.colorbar(sc, ax=ax, label=score)
        style_axes(ax)
    fig.tight_layout()
    out = _outpath("oe_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_prop_constrained_by_coding(annot: pd.DataFrame) -> None:
    bins = np.linspace(0, 1, 11)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins) - 1)]
    annot = annot.copy()
    annot["_cdbin"] = pd.cut(annot["coding_prop"].clip(0, 1), bins=bins, labels=labels, include_lowest=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for score in GNOCCHI_SCORES:
        rows = []
        for label in labels:
            sub = annot[annot["_cdbin"] == label]
            n = len(sub)
            x = int((sub[score] >= 4).sum())
            rows.append((x / n if n else np.nan, sem(x, n) if n else np.nan))
        frac = [r[0] for r in rows]
        se = [r[1] for r in rows]
        ax.errorbar(range(len(labels)), frac, yerr=[1.96 * v for v in se],
                     marker="o", label=score, color=score_color(score))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("% coding bases in window")
    ax.set_ylabel("fraction of windows with Z >= 4")
    ax.legend(fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    out = _outpath("prop_constrained_by_coding.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ── B: enrichment of constrained windows in functional annotations ─────────

def _plot_enrichment_grid(annot: pd.DataFrame, annot_cols: list[str], out_name: str, ncols: int) -> None:
    nrows = int(np.ceil(len(annot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, annot_col in zip(axes.flat, annot_cols):
        for score in GNOCCHI_SCORES:
            enr = enrichment_by_zbin(annot, score, annot_col)
            yerr = [enr["odds_ratio"] - enr["ci_lo"], enr["ci_hi"] - enr["odds_ratio"]]
            ax.errorbar(enr["bin_idx"], enr["odds_ratio"], yerr=yerr,
                        marker="o", ms=3, label=score, color=score_color(score))
        ax.axhline(1.0, color=REFERENCE_LINE_COLOR, ls="--", lw=1)
        ax.set_title(annot_col, fontsize=9)
        ax.set_xticks(range(len(Z_BIN_LABELS)))
        ax.set_xticklabels(Z_BIN_LABELS, rotation=60, fontsize=7)
        ax.set_ylabel("odds ratio")
        style_axes(ax)
    for ax in axes.flat[len(annot_cols):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    out = _outpath(out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


DODGE_WIDTH = 0.6


def _plot_enrichment_fig2_style(
    annot: pd.DataFrame, annot_cols: list[str], ann_colors: dict[str, tuple], out_name: str,
) -> None:
    """Fig. 2a/2b-style layout: one subplot per score (z_adj/z_unadj/
    z_roulette), with every annotation overlaid on the same axis --
    mirrors fig_utils.py's plt_enrichment_re/plt_enrichment_gwas (each of
    which puts all annotations on a single plot for one score), tiled into
    a 3-panel row so all three scores are shown side by side. Annotations
    are dodged evenly within each Z-bin (DODGE_WIDTH split across
    len(annot_cols)) since their odds ratios/CIs otherwise overlap and
    become unreadable when >=2 annotations track each other closely."""
    offsets = (
        np.linspace(-DODGE_WIDTH / 2, DODGE_WIDTH / 2, len(annot_cols))
        if len(annot_cols) > 1 else [0.0]
    )
    fig, axes = plt.subplots(1, len(GNOCCHI_SCORES), figsize=(6 * len(GNOCCHI_SCORES), 4.5))
    for ax, score in zip(axes, GNOCCHI_SCORES):
        for annot_col, offset in zip(annot_cols, offsets):
            enr = enrichment_by_zbin(annot, score, annot_col)
            yerr = [enr["odds_ratio"] - enr["ci_lo"], enr["ci_hi"] - enr["odds_ratio"]]
            ax.errorbar(enr["bin_idx"] + offset, enr["odds_ratio"], yerr=yerr,
                        marker="o", ms=4, ls="-", lw=1, elinewidth=1.5, alpha=0.8,
                        label=annot_col, color=ann_colors[annot_col])
        ax.axhline(1.0, color=REFERENCE_LINE_COLOR, ls="--", lw=1)
        ax.set_title(score, fontsize=10)
        ax.set_xticks(range(len(Z_BIN_LABELS)))
        ax.set_xticklabels(Z_BIN_LABELS, rotation=60, fontsize=7)
        ax.set_ylabel("odds ratio")
        style_axes(ax)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = _outpath(out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_enrichment_regulatory_elements(annot: pd.DataFrame) -> None:
    nc = filter_noncoding_qc(annot)
    _plot_enrichment_grid(nc, REGULATORY_COLS, "enrichment_regulatory_elements.png", ncols=3)
    _plot_enrichment_fig2_style(
        nc, REGULATORY_COLS, REGULATORY_ANN_COLORS, "enrichment_regulatory_elements_fig2a_style.png")


def benchmark_enrichment_gwas(annot: pd.DataFrame) -> None:
    nc = filter_noncoding_qc(annot)
    _plot_enrichment_grid(nc, GWAS_COLS, "enrichment_gwas.png", ncols=3)
    _plot_enrichment_fig2_style(nc, GWAS_COLS, GWAS_ANN_COLORS, "enrichment_gwas_fig2b_style.png")


def benchmark_enrichment_gwas_vs_ccre(annot: pd.DataFrame) -> None:
    annot = filter_noncoding_qc(annot)
    ccre_cols = ["ENCODE cCRE-PLS", "ENCODE cCRE-pELS", "ENCODE cCRE-dELS"]
    in_ccre = pd.concat([coerce_bool(annot[c]) for c in ccre_cols], axis=1).any(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (label, mask) in zip(axes, [("within cCREs", in_ccre), ("outside cCREs", ~in_ccre)]):
        sub = annot[mask]
        for score in GNOCCHI_SCORES:
            enr = enrichment_by_zbin(sub, score, "GWAS Catalog")
            yerr = [enr["odds_ratio"] - enr["ci_lo"], enr["ci_hi"] - enr["odds_ratio"]]
            ax.errorbar(enr["bin_idx"], enr["odds_ratio"], yerr=yerr,
                        marker="o", ms=3, label=score, color=score_color(score))
        ax.axhline(1.0, color=REFERENCE_LINE_COLOR, ls="--", lw=1)
        ax.set_title(f"GWAS Catalog enrichment, {label}", fontsize=9)
        ax.set_xticks(range(len(Z_BIN_LABELS)))
        ax.set_xticklabels(Z_BIN_LABELS, rotation=60, fontsize=7)
        style_axes(ax)
    axes[0].set_ylabel("odds ratio")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = _outpath("enrichment_gwas_vs_ccre.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_prop_roadmaplinks(annot: pd.DataFrame) -> None:
    annot = filter_noncoding_qc(annot)
    fig, ax = plt.subplots(figsize=(8, 5))
    for score in GNOCCHI_SCORES:
        enr = enrichment_by_zbin(annot, score, "RoadmapLinks")
        se = np.sqrt(enr["frac_annot"] * (1 - enr["frac_annot"]) / enr["n"])
        ax.errorbar(enr["bin_idx"], enr["frac_annot"], yerr=1.96 * se,
                     marker="o", label=score, color=score_color(score))
    ax.set_xticks(range(len(Z_BIN_LABELS)))
    ax.set_xticklabels(Z_BIN_LABELS, rotation=45, ha="right")
    ax.set_ylabel("fraction of windows with a RoadmapLinks enhancer")
    ax.legend(fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    out = _outpath("prop_roadmaplinks.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ── E: head-to-head ROC/AUC vs. other constraint/conservation scores ────────

def benchmark_roc_auc(roulette_windows: pd.DataFrame) -> None:
    neg = load_comparisons_table("topmed_maf5")
    if neg is None:
        return
    neg = attach_roulette_score(neg, roulette_windows)

    fig, axes = plt.subplots(1, len(POSITIVE_SETS), figsize=(5 * len(POSITIVE_SETS), 4.5))
    for ax, pos_name in zip(np.atleast_1d(axes), POSITIVE_SETS):
        pos = load_comparisons_table(pos_name)
        if pos is None:
            ax.set_title(f"{pos_name}: no data")
            continue
        pos = attach_roulette_score(pos, roulette_windows)
        roc = roc_auc_by_score(pos, neg, ALL_SCORE_COLS)
        for score, res in sorted(roc.items(), key=lambda kv: -kv[1]["auc"]):
            ax.plot(res["fpr"], res["tpr"], color=score_color(score),
                     label=f"{score} ({res['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE_LINE_COLOR)
        ax.set_title(pos_name, fontsize=9)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=6, loc="lower right")
        style_axes(ax)
    fig.suptitle("ROC vs. TopMed common variants (MAF>5%) background")
    fig.tight_layout()
    out = _outpath("roc_auc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_auc_vs_af(roulette_windows: pd.DataFrame) -> None:
    pos = load_comparisons_table("gwas_fine-mapping_pip09")
    if pos is None:
        return
    pos = attach_roulette_score(pos, roulette_windows)

    neg_sizes, labels = [], []
    auc_by_score = {c: [] for c in ALL_SCORE_COLS}
    for neg_name in NEGATIVE_SETS:
        neg = load_comparisons_table(neg_name)
        if neg is None:
            continue
        neg = attach_roulette_score(neg, roulette_windows)
        labels.append(neg_name)
        neg_sizes.append(len(neg))
        roc = roc_auc_by_score(pos, neg, ALL_SCORE_COLS, n_boot=5)
        for c in ALL_SCORE_COLS:
            auc_by_score[c].append(roc.get(c, {}).get("auc", np.nan))
    if not labels:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.5]})
    ax1.bar(range(len(labels)), neg_sizes, color=REFERENCE_LINE_COLOR)
    ax1.set_ylabel("n (negative set)")
    for c in ALL_SCORE_COLS:
        ax2.plot(range(len(labels)), auc_by_score[c], marker="o", label=c, color=score_color(c))
    ax2.axhline(0.5, color=REFERENCE_LINE_COLOR, ls="--")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("AUC")
    ax2.legend(fontsize=6, bbox_to_anchor=(1, 0.75))
    style_axes(ax1)
    style_axes(ax2)
    fig.suptitle("AUC vs. negative-set allele-frequency stratum\n(positive set: GWAS fine-mapping PIP>0.9)")
    fig.tight_layout()
    out = _outpath("auc_vs_af.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_gnocchi_self_comparison(roulette_windows: pd.DataFrame) -> None:
    pos = load_comparisons_table("gwas_fine-mapping_pip09_hc")
    neg = load_comparisons_table("topmed_maf5")
    if pos is None or neg is None:
        return
    pos = attach_roulette_score(pos, roulette_windows)
    neg = attach_roulette_score(neg, roulette_windows)

    variant_groups = {
        "mutation model": ["z_adj", "z_sliding100", "z_trimer", "z_heptamer", "z_roulette"],
        "window size": ["z_100bp", "z_500bp", "z_adj", "z_2kb", "z_3kb"],
        "population": ["z_adj", "z_global", "z_nfe"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (group, cols) in zip(axes, variant_groups.items()):
        roc = roc_auc_by_score(pos, neg, cols, n_boot=5)
        for i, (score, res) in enumerate(sorted(roc.items(), key=lambda kv: -kv[1]["auc"])):
            ax.plot(res["fpr"], res["tpr"], label=f"{score} ({res['auc']:.3f})",
                     color=score_color(score, fallback_index=i))
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE_LINE_COLOR)
        ax.set_title(group, fontsize=10)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
        style_axes(ax)
    fig.suptitle("Gnocchi self-comparison (positive: GWAS fine-mapping, high-confidence)")
    fig.tight_layout()
    out = _outpath("gnocchi_self_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_dominance_analysis(roulette_windows: pd.DataFrame) -> None:
    try:
        from dominance_analysis import Dominance
    except ImportError:
        print("  skipping: `dominance_analysis` not importable in this environment "
              "(pip install dominance-analysis; already installed in the hail conda env)")
        return

    scores = ALL_SCORE_COLS
    score_type = {s: "Human lineage-specific constraint" for s in GNOCCHI_SCORES + ["Orion", "CDTS", "gwRVIS", "DR"]}
    score_type.update({s: "Interspecies conservation" for s in ["phastCons", "phyloP", "GERP"]})
    score_label = {s: s for s in scores}
    score_label.update({"z_adj": "Gnocchi (adjusted)", "z_unadj": "Gnocchi (unadjusted)", "z_roulette": "Roulette"})

    configs = [
        ("gwas", ["gwas_catalog_repl", "gwas_fine-mapping_pip09"], "topmed_maf5", "GWAS"),
        ("clinvar_plp_hgmd", ["clinvar_hgmd"], "topmed_mac1", "Likely pathogenic"),
    ]
    for key, pos_names, neg_name, ylabel in configs:
        pos_frames = [load_comparisons_table(n) for n in pos_names]
        if any(p is None for p in pos_frames):
            print(f"  skipping dominance/{key}: missing positive set data")
            continue
        neg = load_comparisons_table(neg_name)
        if neg is None:
            print(f"  skipping dominance/{key}: missing negative set data")
            continue

        pos = pd.concat(pos_frames).drop_duplicates(subset=["locus"])
        pos = attach_roulette_score(pos, roulette_windows)
        neg = attach_roulette_score(neg, roulette_windows)
        pos = pos.copy()
        neg = neg.copy()
        pos["group"] = 1
        neg["group"] = 0
        if len(neg) > 10 * len(pos):
            neg = neg.sample(n=10 * len(pos), random_state=714)

        df01 = pd.concat([pos, neg]).drop_duplicates(subset=["locus"])
        for s in scores:
            if s not in df01.columns:
                continue
            df01[f"{s}_"] = df01[s] - np.nanmin(df01[s])
        available = [s for s in scores if f"{s}_" in df01.columns]
        df01 = df01.dropna(subset=available)
        if df01["group"].nunique() < 2 or len(df01) < 20:
            print(f"  skipping dominance/{key}: insufficient overlap after dropna")
            continue

        dfx = df01[[f"{s}_" for s in available]]
        dfy = df01["group"]
        dft = pd.concat([dfx, dfy], axis=1)
        dom = Dominance(data=dft, target="group", objective=0, pseudo_r2="mcfadden", top_k=dfx.shape[1])
        dom.incremental_rsquare()
        dfp = dom.dominance_stats()
        dfp = dfp.copy()
        dfp["score"] = ["_".join(i.split("_")[:-1]) for i in dfp.index]
        dfp["score_label"] = dfp["score"].map(score_label)
        dfp["score_type"] = dfp["score"].map(score_type)
        dfp["row_number"] = np.arange(len(dfp))

        fig, ax = plt.subplots(figsize=(6, 4))
        for stype, color in [("Human lineage-specific constraint", GNOCCHI_COLORS["z_adj"]),
                              ("Interspecies conservation", "#613969")]:
            sub = dfp[dfp["score_type"] == stype]
            ax.bar(sub["row_number"], sub["Percentage Relative Importance"],
                    width=0.8, color=color, alpha=0.7, label=stype)
        ax.set_xticks(range(len(dfp)))
        ax.set_xticklabels(dfp["score_label"], rotation=90)
        ax.set_ylabel(f"Relative contribution in\n{ylabel} variant classification (%)")
        ax.legend(fontsize=9)
        style_axes(ax)
        fig.tight_layout()
        out = _outpath(f"dominance_{key}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


# ── G: enhancer-level gene-essentiality benchmarks ──────────────────────────

def benchmark_enhancer_geneset(roulette_windows: pd.DataFrame) -> None:
    enh = load_enh_gene_roadmaplinks()
    if enh is None:
        return
    enh_parsed = parse_element_id(enh, "enhancer")
    corr = validate_enhancer_rollup(enh, roulette_windows)
    print(f"  Gnocchi enhancer-Z rollup validation: r={corr:.3f} vs the file's existing "
          "enhancer_constraint_Z (sum obs/exp over overlapping windows, then chi-sq)")

    enh = enh.copy()
    enh["roulette_enhancer_Z"] = compute_enhancer_z(
        enh_parsed, roulette_windows, obs_col="observed", exp_col="exp_roulette").to_numpy()

    fig, axes = plt.subplots(1, len(GENE_SET_COLS), figsize=(3.2 * len(GENE_SET_COLS), 4.5), sharey=True)
    for ax, col in zip(axes, GENE_SET_COLS):
        flag = coerce_bool(enh[col])
        positions = [0, 1, 2.2, 3.2]
        data = [
            enh.loc[~flag, "enhancer_constraint_Z"].dropna(),
            enh.loc[flag, "enhancer_constraint_Z"].dropna(),
            enh.loc[~flag, "roulette_enhancer_Z"].dropna(),
            enh.loc[flag, "roulette_enhancer_Z"].dropna(),
        ]
        colors = [GNOCCHI_COLORS["z_adj"], GNOCCHI_COLORS["z_adj"],
                  GNOCCHI_COLORS["z_roulette"], GNOCCHI_COLORS["z_roulette"]]
        alphas = [0.35, 0.8, 0.35, 0.8]
        bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=False, patch_artist=True)
        for patch, color, alpha in zip(bp["boxes"], colors, alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
        ax.set_xticks(positions)
        ax.set_xticklabels(["no", "yes", "no", "yes"], fontsize=7)
        ax.set_title(col, fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel("Enhancer constraint Z")
    fig.suptitle("Enhancer constraint Z by target-gene set membership (Gnocchi vs Roulette)")
    fig.tight_layout()
    out = _outpath("enhancer_geneset.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def benchmark_enhancer_loeuf_roc(roulette_windows: pd.DataFrame) -> None:
    import statsmodels.api as sm

    enhz = load_enhz_loeuf_pred()
    enh = load_enh_gene_roadmaplinks()
    if enhz is None or enh is None:
        return

    enh_parsed = parse_element_id(enh, "enhancer")
    enh = enh.copy()
    enh["roulette_enhancer_Z"] = compute_enhancer_z(
        enh_parsed, roulette_windows, obs_col="observed", exp_col="exp_roulette").to_numpy()

    rollup_corr = validate_gene_rollup_rule(enh, enhz)
    print(f"  gene<-enhancer rollup check (mean/max/min agreeing => each gene has exactly "
          f"one linked enhancer): {rollup_corr}")
    gene_roulette = enh.groupby("gene")["roulette_enhancer_Z"].mean()
    enhz = enhz.merge(gene_roulette.rename("roulette_enhancer_Z"), on="gene", how="left")

    need = ["LOEUF", "enhancer_constraint_Z", "roulette_enhancer_Z", "constrained"]
    train = enhz[enhz["train_test"] == "train"].dropna(subset=need)
    test = enhz[(enhz["train_test"] == "test") & enhz["LOEUF_underpowered"].astype(bool)].dropna(subset=need)
    if train.empty or test.empty:
        print("  skipping enhancer_loeuf_roc: empty train/test split after dropna")
        return

    model_specs = {
        "LOEUF only": ["LOEUF"],
        "LOEUF + Gnocchi enhancer Z": ["LOEUF", "enhancer_constraint_Z"],
        "LOEUF + Roulette enhancer Z": ["LOEUF", "roulette_enhancer_Z"],
    }
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (label, cols) in enumerate(model_specs.items()):
        x_train = sm.add_constant(train[cols])
        logit = sm.Logit(train["constrained"], x_train).fit_regularized(disp=0)
        x_test = sm.add_constant(test[cols], has_constant="add")
        pred = logit.predict(x_test)
        auc = roc_auc_score(test["constrained"], pred)
        fpr, tpr, _ = roc_curve(test["constrained"], pred)
        ax.plot(fpr, tpr, label=f"{label} ({auc:.3f})", color=score_color(label, fallback_index=i))
    ax.plot([0, 1], [0, 1], "--", color=REFERENCE_LINE_COLOR)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Predicting constrained genes\n(LOEUF-underpowered test set)")
    ax.legend(fontsize=8, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    out = _outpath("enhancer_loeuf_roc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    skipped: list[str] = []

    def run(name: str, fn, *args) -> None:
        print(f"\n=== {name} ===")
        missing = check_benchmark_data(name)
        if missing:
            print(f"  skipped: missing {missing}")
            skipped.append(name)
            return
        fn(*args)

    print("Loading genome-wide Roulette window table (for z_roulette joins + enhancer rollups) ...")
    roulette_windows = build_roulette_windows()

    print("Loading + joining the annotated genome-wide constraint track ...")
    annot = load_annot_table()
    if annot is not None:
        annot = attach_roulette_to_annot(annot, roulette_windows)

    run("score_distributions", benchmark_score_distributions, annot)
    run("oe_scatter", benchmark_oe_scatter, annot)
    run("prop_constrained_by_coding", benchmark_prop_constrained_by_coding, annot)
    run("enrichment_regulatory_elements", benchmark_enrichment_regulatory_elements, annot)
    run("enrichment_gwas", benchmark_enrichment_gwas, annot)
    run("enrichment_gwas_vs_ccre", benchmark_enrichment_gwas_vs_ccre, annot)
    run("prop_roadmaplinks", benchmark_prop_roadmaplinks, annot)
    run("roc_auc", benchmark_roc_auc, roulette_windows)
    run("auc_vs_af", benchmark_auc_vs_af, roulette_windows)
    run("gnocchi_self_comparison", benchmark_gnocchi_self_comparison, roulette_windows)
    run("dominance_analysis", benchmark_dominance_analysis, roulette_windows)
    run("enhancer_geneset", benchmark_enhancer_geneset, roulette_windows)
    run("enhancer_loeuf_roc", benchmark_enhancer_loeuf_roc, roulette_windows)

    print("\n=== Summary ===")
    if skipped:
        print("  Skipped (missing data):", ", ".join(skipped))
    else:
        print("  All benchmarks ran.")
    print("\nDone.")


if __name__ == "__main__":
    main()

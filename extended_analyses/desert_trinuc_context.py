#!/usr/bin/env python3
"""Trinucleotide context composition diagnostics for gene-desert anomalies."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from utils.desert_utils import (
    DATA_DIR,
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_features,
    load_gnocchi,
)

CONTEXT_TABLE = os.path.join(DATA_DIR, "expected_counts_per_context_methyl_genome_1kb.txt.gz")
TOP_CONTEXTS = 20


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _build_context_tables(labeled_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    element_to_desert = labeled_windows.set_index("element_id")["desert_id"].to_dict()
    desert_elements = set(element_to_desert.keys())

    genome_ctx = defaultdict(float)
    desert_ctx = defaultdict(float)  # (desert, context) -> expected sum

    reader = pd.read_csv(
        CONTEXT_TABLE,
        sep="\t",
        usecols=["element_id", "context", "expected"],
        chunksize=1_000_000,
        low_memory=False,
    )
    for chunk in reader:
        chunk["expected"] = pd.to_numeric(chunk["expected"], errors="coerce").fillna(0.0)
        g = chunk.groupby("context", observed=True)["expected"].sum()
        for ctx, val in g.items():
            genome_ctx[str(ctx)] += float(val)

        sub = chunk[chunk["element_id"].isin(desert_elements)].copy()
        if sub.empty:
            continue
        sub["desert_id"] = sub["element_id"].map(element_to_desert)
        d = sub.groupby(["desert_id", "context"], observed=True)["expected"].sum()
        for (desert_id, context), val in d.items():
            desert_ctx[(str(desert_id), str(context))] += float(val)

    genome_df = pd.DataFrame(
        {"context": list(genome_ctx.keys()), "genome_expected_sum": list(genome_ctx.values())}
    ).sort_values("genome_expected_sum", ascending=False)
    desert_df = pd.DataFrame(
        [
            {"desert_id": k[0], "context": k[1], "desert_expected_sum": v}
            for k, v in desert_ctx.items()
        ]
    )
    return genome_df, desert_df


def _plot_exemplar(
    desert_id: str,
    desert_ctx: pd.DataFrame,
    genome_ctx: pd.DataFrame,
    desert_stats: pd.DataFrame,
    out_path: str,
) -> None:
    sub = desert_ctx[desert_ctx["desert_id"] == desert_id].copy()
    if sub.empty:
        return

    # Build full (all-context) distribution tables for this desert.
    all_ctx = sorted(set(genome_ctx["context"]) | set(sub["context"]))
    p_d = (
        sub.set_index("context")["desert_expected_sum"]
        .reindex(all_ctx, fill_value=0.0)
    )
    p_g = (
        genome_ctx.set_index("context")["genome_expected_sum"]
        .reindex(all_ctx, fill_value=0.0)
    )
    full = pd.DataFrame({"context": all_ctx,
                         "p_desert": p_d.values / p_d.values.sum().clip(1e-12),
                         "p_genome": p_g.values / p_g.values.sum().clip(1e-12)})
    full["prop_shift"] = full["p_desert"] - full["p_genome"]
    full["kl_contrib"] = (full["p_desert"] + 1e-12) * np.log(
        (full["p_desert"] + 1e-12) / (full["p_genome"] + 1e-12)
    )
    full["abs_shift"] = full["prop_shift"].abs()
    # Central base: works for standard 3-mers (e.g. "ACG") and longer strings.
    full["central_base"] = full["context"].str[1].where(full["context"].str.len() >= 3,
                                                         full["context"].str[0])

    stat = desert_stats[desert_stats["desert_id"] == desert_id].iloc[0]
    oe_val = float(stat["oe_unadj_desert"])
    kl_val = float(stat["kl_desert_vs_genome"])
    chi2_stat_val = float(stat["chi2_stat"])
    chi2_p = float(stat["chi2_pvalue"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ── [0,0]: Composition shift scatter (all contexts) ──────────────────
    ax = axes[0, 0]
    ax.scatter(full["p_genome"], full["p_desert"], s=18, alpha=0.5,
               color="tab:blue", linewidths=0)
    lim = max(float(full["p_genome"].max()), float(full["p_desert"].max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    for _, row in full.nlargest(6, "abs_shift").iterrows():
        ax.annotate(row["context"],
                    (row["p_genome"], row["p_desert"]),
                    fontsize=7, color="0.25",
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Genome context proportion")
    ax.set_ylabel(f"{desert_id} context proportion")
    ax.set_title(f"Composition shift  (O/E unadj = {oe_val:.3f})")

    # ── [0,1]: Per-context KL contribution (top 15) ──────────────────────
    ax = axes[0, 1]
    top_kl = full.reindex(full["kl_contrib"].abs().nlargest(15).index).sort_values("kl_contrib")
    bar_colors = ["tab:red" if v < 0 else "tab:blue" for v in top_kl["kl_contrib"]]
    ax.barh(top_kl["context"], top_kl["kl_contrib"], color=bar_colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$p_k \log(p_k / q_k)$  (contribution to KL)")
    ax.set_title(f"Per-context KL contribution  (total KL = {kl_val:.4f})")
    ax.tick_params(axis="y", labelsize=8)

    # ── [1,0]: Fleet KL histogram — where does this desert sit? ──────────
    ax = axes[1, 0]
    fleet_kl = desert_stats["kl_desert_vs_genome"].dropna()
    pct = float((fleet_kl < kl_val).mean()) * 100
    ax.hist(fleet_kl, bins=40, color="tab:gray", edgecolor="white", alpha=0.75)
    ax.axvline(kl_val, color="tab:red", lw=2,
               label=f"{desert_id}  KL={kl_val:.4f}  ({pct:.0f}th pct)")
    ax.set_xlabel("KL divergence (desert || genome)")
    ax.set_ylabel("Number of deserts")
    ax.set_title(f"Divergence in fleet context  ($\\chi^2$ p = {chi2_p:.2e})")
    ax.legend(fontsize=8)

    # ── [1,1]: Composition by central base ───────────────────────────────
    ax = axes[1, 1]
    base_grp = (full.groupby("central_base", sort=True)[["p_desert", "p_genome"]]
                .sum().reset_index())
    x = np.arange(len(base_grp))
    w = 0.35
    ax.bar(x - w / 2, base_grp["p_desert"], width=w,
           label=desert_id, color="tab:blue", alpha=0.85)
    ax.bar(x + w / 2, base_grp["p_genome"], width=w,
           label="Genome", color="tab:gray", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(base_grp["central_base"], fontsize=11)
    ax.set_ylabel("Summed context proportion")
    ax.set_title("Composition by central trinucleotide base")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{desert_id} trinucleotide context diagnostics"
        f"   |   KL = {kl_val:.4f}   $\\chi^2$ = {chi2_stat_val:.1f}   p = {chi2_p:.2e}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet_overview(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ── [0,0]: KL divergence distribution ────────────────────────────────
    ax = axes[0, 0]
    ax.hist(summary["kl_desert_vs_genome"].dropna(), bins=35,
            color="tab:blue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("KL divergence (desert || genome)")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Context divergence distribution across deserts")

    # ── [0,1]: KL vs mean z_unadj — key hypothesis ───────────────────────
    ax = axes[0, 1]
    plot_df = summary.dropna(subset=["kl_desert_vs_genome", "mean_z_unadj"])
    ax.scatter(plot_df["kl_desert_vs_genome"], plot_df["mean_z_unadj"],
               s=16, alpha=0.55, color="tab:gray")
    for desert_id in DESERT_ORDER:
        row = plot_df[plot_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id,
                    (row["kl_desert_vs_genome"].iloc[0], row["mean_z_unadj"].iloc[0]),
                    fontsize=8)
    ax.set_xlabel("KL divergence")
    ax.set_ylabel("Mean z_unadj")
    ax.set_title("Context divergence vs unadjusted constraint anomaly")

    # ── [1,0]: GC content vs KL — confounder check ───────────────────────
    ax = axes[1, 0]
    plot_df = summary.dropna(subset=["kl_desert_vs_genome", "mean_gc_1k"])
    ax.scatter(plot_df["mean_gc_1k"], plot_df["kl_desert_vs_genome"],
               s=16, alpha=0.55, color="tab:green")
    for desert_id in DESERT_ORDER:
        row = plot_df[plot_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id,
                    (row["mean_gc_1k"].iloc[0], row["kl_desert_vs_genome"].iloc[0]),
                    fontsize=8)
    ax.set_xlabel("Mean GC content (1 kb windows)")
    ax.set_ylabel("KL divergence")
    ax.set_title("GC content vs context divergence (confounder check)")

    # ── [1,1]: KL vs −log10(χ² p) significance volcano ───────────────────
    ax = axes[1, 1]
    vol_df = summary.dropna(subset=["kl_desert_vs_genome", "chi2_pvalue"]).copy()
    vol_df["neg_log10_p"] = -np.log10(vol_df["chi2_pvalue"].clip(lower=1e-300))
    ax.scatter(vol_df["kl_desert_vs_genome"], vol_df["neg_log10_p"],
               s=16, alpha=0.55, color="tab:orange")
    for desert_id in DESERT_ORDER:
        row = vol_df[vol_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id,
                    (row["kl_desert_vs_genome"].iloc[0], row["neg_log10_p"].iloc[0]),
                    fontsize=8)
    ax.set_xlabel("KL divergence")
    ax.set_ylabel(r"$-\log_{10}$($\chi^2$ p-value)")
    ax.set_title(r"Divergence magnitude vs $\chi^2$ significance")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading merged Gnocchi windows ...")
    gn = load_gnocchi(
        usecols=[
            "element_id",
            "chrom",
            "start",
            "end",
            "observed",
            "expected_unadj",
            "z_adj",
            "z_unadj",
        ]
    )
    gn["start"] = gn["start"].astype(np.int64)
    gn["end"] = gn["end"].astype(np.int64)

    print("Labeling all deserts ...")
    deserts_df = load_all_deserts()
    labeled = label_deserts_fleet(gn, deserts_df)
    in_deserts = labeled[labeled["desert"].notna()].copy().rename(columns={"desert": "desert_id"})

    print("Scanning context expected-count table ...")
    genome_ctx, desert_ctx = _build_context_tables(in_deserts[["element_id", "desert_id"]])

    # Build per-desert context matrices.
    all_contexts = sorted(set(genome_ctx["context"]) | set(desert_ctx["context"]))
    genome_vec = genome_ctx.set_index("context")["genome_expected_sum"].reindex(all_contexts, fill_value=0.0).to_numpy()

    print("Computing divergence metrics ...")
    rows = []
    desert_totals = (
        in_deserts.groupby("desert_id", observed=True)
        .agg(
            n_windows=("element_id", "size"),
            observed_sum=("observed", "sum"),
            expected_unadj_sum=("expected_unadj", "sum"),
            mean_z_unadj=("z_unadj", "mean"),
            mean_z_adj=("z_adj", "mean"),
        )
        .reset_index()
    )
    desert_totals["oe_unadj_desert"] = desert_totals["observed_sum"] / desert_totals["expected_unadj_sum"].clip(lower=1e-12)

    feat = load_features(element_ids=in_deserts["element_id"].tolist())[["element_id", "GC_content_1k"]]
    gc_means = (
        in_deserts[["desert_id", "element_id"]]
        .merge(feat, on="element_id", how="left")
        .groupby("desert_id", observed=True)["GC_content_1k"]
        .mean()
        .rename("mean_gc_1k")
        .reset_index()
    )

    for desert_id, sub in desert_ctx.groupby("desert_id", observed=True):
        d_vec = (
            sub.set_index("context")["desert_expected_sum"]
            .reindex(all_contexts, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        if d_vec.sum() <= 0:
            continue
        p = d_vec / d_vec.sum()
        q = genome_vec / genome_vec.sum()
        kl = _kl_divergence(p, q)
        chi = chisquare(f_obs=d_vec, f_exp=q * d_vec.sum())
        abs_shift = np.abs(p - q)
        top_idx = int(np.argmax(abs_shift))
        rows.append(
            {
                "desert_id": desert_id,
                "kl_desert_vs_genome": kl,
                "chi2_stat": float(chi.statistic),
                "chi2_pvalue": float(chi.pvalue),
                "top_shift_context": all_contexts[top_idx],
                "top_shift_abs_delta_prop": float(abs_shift[top_idx]),
            }
        )

    summary = pd.DataFrame(rows).merge(desert_totals, on="desert_id", how="left").merge(gc_means, on="desert_id", how="left")
    summary = summary.merge(
        deserts_df[["desert_id", "chrom", "start", "end", "panel_label"]],
        on="desert_id",
        how="left",
    )

    out_tsv = os.path.join(RESULTS_DIR, "trinuc_context_desert_summary.tsv")
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"  wrote {out_tsv}")

    print("Generating exemplar panels ...")
    for desert_id in DESERT_ORDER:
        if desert_id not in set(summary["desert_id"]):
            continue
        out_png = os.path.join(RESULTS_DIR, f"trinuc_context_exemplar_{desert_id}.png")
        _plot_exemplar(desert_id, desert_ctx, genome_ctx, summary, out_png)
        print(f"  wrote {out_png}")

    print("Generating fleet overview ...")
    fleet_png = os.path.join(RESULTS_DIR, "trinuc_context_fleet_overview.png")
    _plot_fleet_overview(summary, fleet_png)
    print(f"  wrote {fleet_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()

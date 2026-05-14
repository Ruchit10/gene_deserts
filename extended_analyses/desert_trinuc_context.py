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
    merged = sub.merge(genome_ctx, on="context", how="left")
    merged["p_desert"] = merged["desert_expected_sum"] / merged["desert_expected_sum"].sum()
    merged["p_genome"] = merged["genome_expected_sum"] / merged["genome_expected_sum"].sum()
    merged["log2_enrich"] = np.log2((merged["p_desert"] + 1e-12) / (merged["p_genome"] + 1e-12))
    merged = merged.sort_values("p_desert", ascending=False).head(TOP_CONTEXTS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    x = np.arange(len(merged))
    ax.bar(x - 0.2, merged["p_desert"], width=0.4, label=desert_id, color="tab:blue")
    ax.bar(x + 0.2, merged["p_genome"], width=0.4, label="genome", color="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["context"], rotation=75, fontsize=7)
    ax.set_ylabel("Expected-count proportion")
    ax.set_title("Top contexts by desert composition")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(x, merged["log2_enrich"], color="tab:purple", alpha=0.85)
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["context"], rotation=75, fontsize=7)
    ax.set_ylabel("log2(desert/genome)")
    ax.set_title("Context enrichment")

    ax = axes[1, 0]
    stat = desert_stats[desert_stats["desert_id"] == desert_id].iloc[0]
    oe_val = stat["oe_unadj_desert"]
    ax.scatter(
        merged["p_genome"],
        merged["p_desert"],
        s=45,
        alpha=0.7,
        color="tab:blue",
    )
    lim = max(float(merged["p_genome"].max()), float(merged["p_desert"].max()))
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Genome context proportion")
    ax.set_ylabel(f"{desert_id} context proportion")
    ax.set_title(f"Composition shift (desert O/E_unadj={oe_val:.3f})")

    ax = axes[1, 1]
    ax.scatter(merged["log2_enrich"], np.log10(merged["p_desert"] + 1e-12), color="tab:orange", alpha=0.8)
    for _, row in merged.head(8).iterrows():
        ax.annotate(row["context"], (row["log2_enrich"], np.log10(row["p_desert"] + 1e-12)), fontsize=7)
    ax.set_xlabel("log2 enrichment")
    ax.set_ylabel("log10(desert proportion)")
    ax.set_title("Most shifted contexts")

    fig.suptitle(f"{desert_id} trinucleotide context diagnostics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fleet_overview(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(summary["kl_desert_vs_genome"].dropna(), bins=35, color="tab:blue", edgecolor="white")
    ax.set_xlabel("KL divergence (desert || genome)")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Context divergence across deserts")

    ax = axes[0, 1]
    plot_df = summary.dropna(subset=["kl_desert_vs_genome", "mean_z_unadj"])
    ax.scatter(plot_df["kl_desert_vs_genome"], plot_df["mean_z_unadj"], s=16, alpha=0.65, color="tab:gray")
    for desert_id in DESERT_ORDER:
        row = plot_df[plot_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id, (row["kl_desert_vs_genome"].iloc[0], row["mean_z_unadj"].iloc[0]), fontsize=8)
    ax.set_xlabel("KL divergence")
    ax.set_ylabel("mean z_unadj")
    ax.set_title("Divergence vs unadjusted anomaly")

    ax = axes[1, 0]
    plot_df = summary.dropna(subset=["kl_desert_vs_genome", "mean_gc_1k"])
    ax.scatter(plot_df["mean_gc_1k"], plot_df["kl_desert_vs_genome"], s=16, alpha=0.65, color="tab:green")
    for desert_id in DESERT_ORDER:
        row = plot_df[plot_df["desert_id"] == desert_id]
        if row.empty:
            continue
        ax.annotate(desert_id, (row["mean_gc_1k"].iloc[0], row["kl_desert_vs_genome"].iloc[0]), fontsize=8)
    ax.set_xlabel("mean GC_content_1k")
    ax.set_ylabel("KL divergence")
    ax.set_title("GC strata vs context divergence")

    ax = axes[1, 1]
    top = summary.nlargest(12, "kl_desert_vs_genome")
    ax.barh(top["desert_id"], top["kl_desert_vs_genome"], color="tab:purple", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("KL divergence")
    ax.set_title("Top context-divergent deserts")

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

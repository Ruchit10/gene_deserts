#!/usr/bin/env python3
"""Map non-coding RNA annotations inside gene deserts.

Parses GENCODE v39 gene-level records for seven ncRNA biotypes and asks
whether their positions co-localise with the constraint anomalies (spikes /
dips in Gnocchi z-scores) seen in the exemplar deserts and across the full
fleet of 633 deserts.

Target biotypes (gene_type field):
  lncRNA     – long non-coding RNA (includes former lincRNA in GENCODE v39)
  miRNA      – micro RNA
  snRNA      – small nuclear RNA
  snoRNA     – small nucleolar RNA
  misc_RNA   – miscellaneous RNA
  Mt_tRNA    – mitochondrial tRNA (only nuclear tRNA class in GENCODE v39)
  scaRNA     – small Cajal body-associated RNA

Analysis levels:
  Exemplar   – per-exemplar spatial profiles: z_adj/z_unadj trace + ncRNA
               gene-track panel drawn as coloured rectangles on a shared axis.
  Fleet      – per-desert ncRNA counts, type breakdown, and comparison of
               mean z-score in deserts with vs without each biotype.
  Window     – within each exemplar: z_adj distributions for 1kb windows
               that overlap an ncRNA vs those that do not.

Outputs (all in results/):
  ncrna_desert_catalog.tsv      – every ncRNA overlapping any of the 633 deserts
  ncrna_desert_summary.tsv      – per-desert counts by biotype + total + fraction
  ncrna_exemplar_<ID>.png       – 5 per-exemplar spatial figures
  ncrna_fleet_summary.png       – fleet-level bar/histogram/scatter panels
  ncrna_zscore_context.png      – z_adj: ncRNA-overlapping vs non-overlapping windows
"""

from __future__ import annotations

import gzip
import os
import re
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from utils.desert_utils import (
    DATA_DIR,
    DESERTS,
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    load_gnocchi,
    label_deserts,
)

# ── Constants ──────────────────────────────────────────────────────────────────
GTF_FILE = os.path.join(DATA_DIR, "gencode.v39.annotation.gtf.gz")

TARGET_BIOTYPES: dict[str, str] = {
    "lncRNA":   "tab:blue",
    "miRNA":    "tab:red",
    "snRNA":    "tab:orange",
    "snoRNA":   "tab:green",
    "misc_RNA": "tab:purple",
    "Mt_tRNA":  "tab:brown",
    "scaRNA":   "tab:pink",
}
BIOTYPE_ORDER = list(TARGET_BIOTYPES.keys())


# ── GTF parser ─────────────────────────────────────────────────────────────────

def _parse_gtf() -> pd.DataFrame:
    """Extract gene-level records for the target biotypes from GENCODE GTF.

    GTF coordinates are 1-based closed; we convert start to 0-based so all
    interval arithmetic uses the same half-open convention as Gnocchi windows.
    """
    _attr_re = re.compile(r'(\w+) "([^"]+)"')
    rows: list[dict[str, Any]] = []
    target_set = set(TARGET_BIOTYPES.keys())

    with gzip.open(GTF_FILE, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = dict(_attr_re.findall(fields[8]))
            gene_type = attrs.get("gene_type", "")
            if gene_type not in target_set:
                continue
            rows.append(
                {
                    "chrom":      fields[0],
                    "start":      int(fields[3]) - 1,  # convert to 0-based
                    "end":        int(fields[4]),
                    "strand":     fields[6],
                    "gene_id":    attrs.get("gene_id", ""),
                    "gene_name":  attrs.get("gene_name", ""),
                    "gene_type":  gene_type,
                }
            )

    df = pd.DataFrame(rows)
    df["length_kb"] = (df["end"] - df["start"]) / 1000
    return df.sort_values(["chrom", "start"]).reset_index(drop=True)


# ── Overlap helpers ────────────────────────────────────────────────────────────

def _overlapping_ncrnas(deserts_df: pd.DataFrame, ncrna_df: pd.DataFrame) -> pd.DataFrame:
    """Return a catalog of all ncRNAs that overlap any desert (any % overlap)."""
    catalog_rows: list[dict[str, Any]] = []
    for _, d in deserts_df.iterrows():
        chrom, ds, de = d["chrom"], int(d["start"]), int(d["end"])
        hits = ncrna_df[
            (ncrna_df["chrom"] == chrom)
            & (ncrna_df["start"] < de)
            & (ncrna_df["end"] > ds)
        ]
        for _, g in hits.iterrows():
            overlap_bp = min(g["end"], de) - max(g["start"], ds)
            catalog_rows.append(
                {
                    "desert_id":   d["desert_id"],
                    "gene_id":     g["gene_id"],
                    "gene_name":   g["gene_name"],
                    "gene_type":   g["gene_type"],
                    "chrom":       g["chrom"],
                    "start":       g["start"],
                    "end":         g["end"],
                    "strand":      g["strand"],
                    "length_kb":   g["length_kb"],
                    "overlap_bp":  overlap_bp,
                }
            )
    return pd.DataFrame(catalog_rows)


def _tag_ncrna_windows(windows: pd.DataFrame, ncrna_sub: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True for each window overlapping any ncRNA."""
    if ncrna_sub.empty:
        return pd.Series(False, index=windows.index)
    ncrna_arr = ncrna_sub[["start", "end"]].values  # (M, 2)
    w_start = windows["start"].values[:, None]
    w_end   = windows["end"].values[:, None]
    g_start = ncrna_arr[:, 0]
    g_end   = ncrna_arr[:, 1]
    overlap = (w_start < g_end) & (w_end > g_start)  # (N, M)
    return pd.Series(overlap.any(axis=1), index=windows.index)


# ── Per-desert summary ─────────────────────────────────────────────────────────

def _desert_ncrna_summary(
    deserts_df: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    desert_len = {
        r["desert_id"]: (r["end"] - r["start"])
        for _, r in deserts_df.iterrows()
    }
    for did in deserts_df["desert_id"]:
        sub = catalog[catalog["desert_id"] == did]
        row: dict[str, Any] = {"desert_id": did, "total_ncrna": len(sub)}
        for bt in BIOTYPE_ORDER:
            row[f"n_{bt}"] = int((sub["gene_type"] == bt).sum())
        covered_bp = min(sub["overlap_bp"].sum(), desert_len.get(did, 1)) if not sub.empty else 0
        row["frac_desert_covered"] = covered_bp / max(desert_len.get(did, 1), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _mb_fmt(x: float, _: Any) -> str:
    return f"{x/1e6:.2f}"


def _save_exemplar_profile(
    name: str,
    windows: pd.DataFrame,
    ncrna_in_desert: pd.DataFrame,
    out_path: str,
) -> None:
    """2-panel figure: z-score trace (top) + ncRNA gene track (bottom)."""
    chrom, ds, de, note = DESERTS[name]
    sub = windows.sort_values("start").copy()
    pos = (sub["start"] + sub["end"]) / 2

    has_z = "z_adj" in sub.columns
    n_panels = 2 if not ncrna_in_desert.empty else 2

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [3, 1.5]},
        sharex=True,
    )

    # Panel 1 — z-score trace
    ax = axes[0]
    if has_z:
        ax.plot(pos, sub["z_adj"],   lw=0.8, color="tab:blue",   alpha=0.85, label="z_adj")
        ax.plot(pos, sub["z_unadj"], lw=0.8, color="tab:orange", alpha=0.75, label="z_unadj")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Gnocchi z-score")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "z-scores not available (run unadjusted_gnocchi_analysis.py first)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(
        f"{name}  {chrom}:{ds:,}–{de:,}  ({note})\n"
        f"ncRNAs in desert: {len(ncrna_in_desert)}",
        fontsize=11,
    )

    # Shade windows that overlap any ncRNA
    if not ncrna_in_desert.empty and has_z:
        tagged = _tag_ncrna_windows(sub, ncrna_in_desert)
        for _, w in sub[tagged].iterrows():
            ax.axvspan(w["start"], w["end"], alpha=0.12, color="tab:gray", lw=0)

    # Panel 2 — ncRNA gene track
    ax2 = axes[1]
    if ncrna_in_desert.empty:
        ax2.text(0.5, 0.5, "No ncRNAs annotated in this desert",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
    else:
        biotype_y = {bt: i for i, bt in enumerate(BIOTYPE_ORDER)}
        for _, g in ncrna_in_desert.iterrows():
            bt = g["gene_type"]
            y = biotype_y.get(bt, len(BIOTYPE_ORDER))
            color = TARGET_BIOTYPES.get(bt, "tab:gray")
            ax2.broken_barh(
                [(g["start"], g["end"] - g["start"])],
                (y - 0.4, 0.8),
                facecolors=color, edgecolors="none", alpha=0.85,
            )
            # Label short enough names
            if (g["end"] - g["start"]) > (de - ds) * 0.01:
                ax2.text(
                    (g["start"] + g["end"]) / 2, y,
                    g["gene_name"], ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold",
                )
        ax2.set_yticks(list(biotype_y.values()))
        ax2.set_yticklabels(list(biotype_y.keys()), fontsize=8)
        ax2.set_ylim(-0.8, len(BIOTYPE_ORDER) - 0.2)
        ax2.set_ylabel("ncRNA type", fontsize=9)

    ax2.set_xlabel(f"{chrom} position (Mb)", fontsize=9)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(_mb_fmt))

    # Legend patches for ncRNA types
    handles = [mpatches.Patch(color=c, label=bt) for bt, c in TARGET_BIOTYPES.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=7, ncol=4,
               title="ncRNA biotype", title_fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_fleet_summary(
    deserts_df: pd.DataFrame,
    summary: pd.DataFrame,
    catalog: pd.DataFrame,
    fleet_z: pd.DataFrame | None,
    out_path: str,
) -> None:
    """3-panel fleet figure: type breakdown | ncRNA count histogram | z vs ncRNA."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — how many deserts contain each biotype
    ax = axes[0]
    counts = [(bt, int((summary[f"n_{bt}"] > 0).sum())) for bt in BIOTYPE_ORDER]
    labels, vals = zip(*counts)
    colors = [TARGET_BIOTYPES[bt] for bt in labels]
    ax.barh(labels, vals, color=colors, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of deserts containing ≥1 gene")
    ax.set_title("Deserts with each ncRNA biotype")
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)

    # Panel 2 — distribution of total ncRNA count per desert
    ax = axes[1]
    ax.hist(summary["total_ncrna"], bins=range(0, summary["total_ncrna"].max() + 2),
            color="tab:gray", edgecolor="white", alpha=0.8)
    ax.set_xlabel("ncRNAs per desert")
    ax.set_ylabel("Number of deserts")
    ax.set_title(f"ncRNA count per desert (n={len(summary)})")
    n_with = int((summary["total_ncrna"] > 0).sum())
    ax.text(0.97, 0.97, f"{n_with}/{len(summary)} deserts\nhave ≥1 ncRNA",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)

    # Panel 3 — z_adj: deserts with vs without ncRNA (if fleet z available)
    ax = axes[2]
    if fleet_z is not None:
        merged = summary.merge(fleet_z[["desert_id", "mean_z_adj", "mean_delta_z"]],
                                on="desert_id", how="left")
        has_ncrna = merged["total_ncrna"] > 0
        for flag, label, color in [(True, "≥1 ncRNA", "tab:blue"),
                                    (False, "No ncRNA", "tab:gray")]:
            vals_z = merged.loc[merged["total_ncrna"].gt(0) == flag, "mean_z_adj"].dropna()
            ax.hist(vals_z, bins=25, alpha=0.6, color=color, label=f"{label} (n={len(vals_z)})",
                    density=True)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Per-desert mean z_adj")
        ax.set_ylabel("Density")
        ax.set_title("Mean z_adj: deserts with vs without ncRNA")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Fleet z-scores not available\n(run analysis_fleet_deserts.py first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Mean z_adj: deserts with vs without ncRNA")

    fig.suptitle("ncRNA landscape across 633 gene deserts", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_zscore_context(
    exemplar_windows: dict[str, pd.DataFrame],
    exemplar_ncrnas: dict[str, pd.DataFrame],
    out_path: str,
) -> None:
    """Per-exemplar box/strip plots: z_adj for ncRNA-overlapping vs non-overlapping windows."""
    n = len(DESERT_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)

    for ax, name in zip(axes, DESERT_ORDER):
        windows = exemplar_windows.get(name, pd.DataFrame())
        ncrnas  = exemplar_ncrnas.get(name, pd.DataFrame())
        if windows.empty or "z_adj" not in windows.columns:
            ax.set_title(f"{name}\nno data")
            continue

        tagged = _tag_ncrna_windows(windows, ncrnas)
        grp_ncrna    = windows.loc[tagged,  "z_adj"].dropna().values
        grp_no_ncrna = windows.loc[~tagged, "z_adj"].dropna().values

        data   = [grp_no_ncrna, grp_ncrna]
        labels = [f"No ncRNA\n(n={len(grp_no_ncrna)})", f"ncRNA\n(n={len(grp_ncrna)})"]
        colors = ["tab:gray", "tab:blue"]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5, notch=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Jitter strip
        rng = np.random.default_rng(42)
        for i, (group, color) in enumerate(zip(data, colors), start=1):
            if len(group) == 0:
                continue
            jitter = rng.normal(0, 0.07, size=len(group))
            ax.scatter(i + jitter, group, s=6, alpha=0.35, color=color)

        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(name, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("z_adj (1kb window)", fontsize=10)

    fig.suptitle("z_adj in ncRNA-overlapping vs non-overlapping windows (exemplar deserts)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Parsing GENCODE GTF for ncRNA genes ...")
    ncrna_df = _parse_gtf()
    counts_by_type = ncrna_df["gene_type"].value_counts()
    for bt in BIOTYPE_ORDER:
        print(f"  {bt:12s}  {counts_by_type.get(bt, 0):5d} genes")
    print(f"  Total:        {len(ncrna_df):5d} genes")

    print("\nLoading all 633 desert coordinates ...")
    deserts_df = load_all_deserts()
    print(f"  {len(deserts_df):,} deserts")

    print("Finding ncRNA–desert overlaps (fleet) ...")
    catalog = _overlapping_ncrnas(deserts_df, ncrna_df)
    print(f"  {len(catalog):,} ncRNA–desert overlap records across "
          f"{catalog['desert_id'].nunique()} deserts")

    out_catalog = os.path.join(RESULTS_DIR, "ncrna_desert_catalog.tsv")
    catalog.to_csv(out_catalog, sep="\t", index=False)
    print(f"  wrote {out_catalog}")

    summary = _desert_ncrna_summary(deserts_df, catalog)
    out_summary = os.path.join(RESULTS_DIR, "ncrna_desert_summary.tsv")
    summary.to_csv(out_summary, sep="\t", index=False)
    print(f"  wrote {out_summary}")

    # Load fleet z-score summary if available
    fleet_z_path = os.path.join(RESULTS_DIR, "fleet_summary.tsv")
    if os.path.exists(fleet_z_path):
        fleet_z = pd.read_csv(fleet_z_path, sep="\t",
                              usecols=["desert_id", "mean_z_adj", "mean_delta_z"])
        print(f"  loaded fleet z-score summary ({len(fleet_z)} rows)")
    else:
        fleet_z = None
        print("  fleet_summary.tsv not found — skipping z-score comparisons")

    # Load Gnocchi z-scores for exemplar windows
    print("\nLoading Gnocchi z-scores for exemplar windows ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id", "z_adj", "z_unadj"])
    gn = label_deserts(gn)
    print(f"  {len(gn):,} windows loaded")

    exemplar_windows: dict[str, pd.DataFrame] = {}
    exemplar_ncrnas:  dict[str, pd.DataFrame] = {}
    for name, (chrom, ds, de, _) in DESERTS.items():
        exemplar_windows[name] = gn[gn["desert"] == name].copy()
        exemplar_ncrnas[name]  = ncrna_df[
            (ncrna_df["chrom"] == chrom)
            & (ncrna_df["start"] < de)
            & (ncrna_df["end"]   > ds)
        ].copy()

    # ── Exemplar spatial profiles ─────────────────────────────────────────────
    print("\nGenerating exemplar profiles ...")
    for name in DESERT_ORDER:
        out_path = os.path.join(RESULTS_DIR, f"ncrna_exemplar_{name}.png")
        _save_exemplar_profile(
            name,
            exemplar_windows[name],
            exemplar_ncrnas[name],
            out_path,
        )
        n_nc = len(exemplar_ncrnas[name])
        print(f"  wrote {out_path}  ({n_nc} ncRNAs annotated)")

    # ── Fleet summary figure ──────────────────────────────────────────────────
    print("\nGenerating fleet summary figure ...")
    _save_fleet_summary(
        deserts_df, summary, catalog, fleet_z,
        os.path.join(RESULTS_DIR, "ncrna_fleet_summary.png"),
    )
    print("  wrote ncrna_fleet_summary.png")

    # ── Z-score context figure ────────────────────────────────────────────────
    print("Generating z-score context figure ...")
    _save_zscore_context(
        exemplar_windows, exemplar_ncrnas,
        os.path.join(RESULTS_DIR, "ncrna_zscore_context.png"),
    )
    print("  wrote ncrna_zscore_context.png")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Fleet ncRNA summary ===")
    n_with = int((summary["total_ncrna"] > 0).sum())
    print(f"  Deserts with ≥1 ncRNA: {n_with} / {len(summary)}")
    print(f"  {'Biotype':12s}  {'ncRNA genes':>11}  {'deserts w/ ≥1':>13}")
    for bt in BIOTYPE_ORDER:
        col = f"n_{bt}"
        n_genes   = int(catalog[catalog["gene_type"] == bt]["gene_id"].nunique()) if not catalog.empty else 0
        n_deserts = int((summary[col] > 0).sum())
        print(f"  {bt:12s}  {n_genes:11d}  {n_deserts:13d}")

    print("\n=== Exemplar ncRNA content ===")
    for name in DESERT_ORDER:
        nc = exemplar_ncrnas[name]
        if nc.empty:
            print(f"  {name}: no ncRNAs")
        else:
            type_str = ", ".join(
                f"{bt}×{int((nc['gene_type']==bt).sum())}"
                for bt in BIOTYPE_ORDER if (nc["gene_type"] == bt).any()
            )
            names_str = ", ".join(nc["gene_name"].tolist())
            print(f"  {name}: {len(nc)} ncRNA(s)  [{type_str}]  — {names_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mappability diagnostics for gene desert Gnocchi windows.

Assesses whether gene deserts have mapping quality (MQ), low-complexity
region (LCR), or segmental duplication (Segdup) patterns that could
compromise Gnocchi score reliability.  Analysis at two resolutions:

  Exemplar level  -- spatial profiles of MQ / LCR / Segdup per 1kb window
                     for each of the 5 hand-curated exemplar deserts,
                     overlaid with z_adj for context.

  Fleet level     -- per-desert aggregated stats for all 633 deserts, with
                     correlation to z_adj / delta_z and a flagging table
                     identifying potentially unreliable deserts.

Outputs (all in results/):
  mappability_desert_summary.tsv
  mappability_exemplar_profiles.png      (3 metrics × 5 exemplars)
  mappability_fleet_distributions.png   (histograms across 633)
  mappability_fleet_vs_zscore.png        (scatter: mappability ↔ z-score)
  mappability_flagged_deserts.tsv        (deserts that fail any threshold)
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from desert_utils import (
    DATA_DIR,
    DESERTS,
    DESERT_ORDER,
    GNOCCHI_TABLE,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAPPABILITY_FILE = os.path.join(DATA_DIR, "gnocchi.windows.mq.lcr.segdup.stats.tsv.gz")
RAW_GNOCCHI_FILE = os.path.join(DATA_DIR, "constraint_z_genome_1kb.qc.download.txt.gz")

# Columns to keep from the mappability file
MQ_COL = "MQ.mean"
LCR_COL = "LCR"
SEGDUP_COL = "Segdup"
METRIC_COLS = [MQ_COL, LCR_COL, SEGDUP_COL]

# Flagging thresholds (per-desert mean values)
FLAG_THRESHOLDS = {
    MQ_COL:     ("lt", 40.0),   # mean MQ below 40 → low confidence
    LCR_COL:    ("gt", 0.30),   # >30% of bases in LCR
    SEGDUP_COL: ("gt", 0.10),   # >10% of bases in segmental duplication
}

METRIC_LABELS = {
    MQ_COL:     "Mean mapping quality (MQ)",
    LCR_COL:    "LCR fraction",
    SEGDUP_COL: "Segdup fraction",
}
METRIC_COLORS = {
    MQ_COL:     "tab:blue",
    LCR_COL:    "tab:orange",
    SEGDUP_COL: "tab:red",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_mappability() -> pd.DataFrame:
    df = pd.read_csv(
        MAPPABILITY_FILE, sep="\t",
        usecols=["element_id", MQ_COL, LCR_COL, SEGDUP_COL],
    )
    return df


def _load_positions() -> pd.DataFrame:
    """Load chrom/start/end/element_id from raw Gnocchi table (lightweight)."""
    df = pd.read_csv(
        RAW_GNOCCHI_FILE, sep="\t",
        usecols=["chrom", "start", "end", "element_id"],
    )
    return df


def _load_z_scores() -> pd.DataFrame | None:
    """Load z_adj / z_unadj / delta_z from the merged Gnocchi table if present."""
    if not os.path.exists(GNOCCHI_TABLE):
        return None
    return pd.read_csv(
        GNOCCHI_TABLE, sep="\t",
        usecols=["element_id", "z_adj", "z_unadj", "delta_z"],
    )


# ── Per-desert summary ────────────────────────────────────────────────────────

def _desert_stats(df: pd.DataFrame, z_df: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for desert_id, sub in df.groupby("desert", sort=False):
        row: dict[str, Any] = {"desert_id": desert_id, "n_windows": len(sub)}
        for col in METRIC_COLS:
            vals = sub[col].dropna().values
            row[f"mean_{col}"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"median_{col}"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"frac_windows_missing_{col}"] = float(sub[col].isna().mean())

        if z_df is not None:
            sub_z = sub.merge(z_df, on="element_id", how="left")
            for zcol in ["z_adj", "z_unadj", "delta_z"]:
                if zcol in sub_z.columns:
                    row[f"mean_{zcol}"] = float(sub_z[zcol].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _flag_deserts(summary: pd.DataFrame) -> pd.DataFrame:
    flag_cols = []
    for col, (direction, threshold) in FLAG_THRESHOLDS.items():
        mean_col = f"mean_{col}"
        if mean_col not in summary.columns:
            continue
        flag_name = f"flag_{col}"
        if direction == "lt":
            summary[flag_name] = summary[mean_col] < threshold
        else:
            summary[flag_name] = summary[mean_col] > threshold
        flag_cols.append(flag_name)
    summary["any_flag"] = summary[flag_cols].any(axis=1)
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_exemplar_profiles(df: pd.DataFrame, z_df: pd.DataFrame | None, out_path: str) -> None:
    """4-row × 5-column grid: MQ, LCR, Segdup, z_adj (if available) per exemplar."""
    n_rows = 4 if z_df is not None else 3
    fig, axes = plt.subplots(n_rows, len(DESERT_ORDER), figsize=(4 * len(DESERT_ORDER), 3 * n_rows),
                             sharey="row")

    for col_idx, name in enumerate(DESERT_ORDER):
        chrom, start, end, note = DESERTS[name]
        sub = df[df["desert"] == name].sort_values("start").copy()
        if sub.empty:
            for r in range(n_rows):
                axes[r, col_idx].set_title(f"{name}\nno data")
            continue

        pos = (sub["start"] + sub["end"]) / 2

        fmt = ticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}")

        for row_idx, col in enumerate(METRIC_COLS):
            ax = axes[row_idx, col_idx]
            color = METRIC_COLORS[col]
            ax.plot(pos, sub[col], lw=0.7, color=color, alpha=0.85)
            ax.axhline(FLAG_THRESHOLDS[col][1], color="grey", lw=0.6, ls="--", alpha=0.7)
            if row_idx == 0:
                ax.set_title(f"{name}\n({note})", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[col], fontsize=8)
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(labelsize=7)

        if z_df is not None:
            sub_z = sub.merge(z_df, on="element_id", how="left")
            pos_z = (sub_z["start"] + sub_z["end"]) / 2
            ax = axes[3, col_idx]
            ax.plot(pos_z, sub_z["z_adj"], lw=0.7, color="tab:blue", alpha=0.8, label="z_adj")
            ax.plot(pos_z, sub_z["z_unadj"], lw=0.7, color="tab:orange", alpha=0.8, label="z_unadj")
            ax.axhline(0, color="grey", lw=0.5, ls="--")
            if col_idx == 0:
                ax.set_ylabel("Gnocchi z", fontsize=8)
                ax.legend(fontsize=7)
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(labelsize=7)

        axes[n_rows - 1, col_idx].set_xlabel("Position (Mb)", fontsize=8)

    fig.suptitle("Mappability metrics and Gnocchi z-scores across exemplar deserts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_fleet_distributions(summary: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, METRIC_COLS):
        mean_col = f"mean_{col}"
        vals = summary[mean_col].dropna()
        ax.hist(vals, bins=40, color=METRIC_COLORS[col], alpha=0.7, edgecolor="white")
        threshold = FLAG_THRESHOLDS[col][1]
        ax.axvline(threshold, color="black", lw=1.2, ls="--", label=f"flag={threshold}")

        # Annotate exemplar positions
        ex_summary = summary[summary["desert_id"].isin(set(DESERTS.keys()))]
        for _, row in ex_summary.iterrows():
            if pd.notna(row[mean_col]):
                ax.axvline(row[mean_col], color="tab:purple", lw=0.8, ls=":", alpha=0.8)

        ax.set_xlabel(METRIC_LABELS[col])
        ax.set_ylabel("Number of deserts")
        ax.set_title(f"Fleet distribution: {METRIC_LABELS[col]}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_fleet_vs_zscore(summary: pd.DataFrame, out_path: str) -> None:
    z_cols = [c for c in ["mean_z_adj", "mean_delta_z"] if c in summary.columns]
    if not z_cols:
        print("  Skipping fleet-vs-zscore plot (no z-score columns in summary)")
        return

    n_metric = len(METRIC_COLS)
    n_z = len(z_cols)
    fig, axes = plt.subplots(n_metric, n_z, figsize=(5 * n_z, 4 * n_metric), squeeze=False)

    flag_cols = [f"flag_{c}" for c in METRIC_COLS if f"flag_{c}" in summary.columns]
    any_flag = summary[flag_cols].any(axis=1) if flag_cols else pd.Series(False, index=summary.index)

    for r, metric_col in enumerate(METRIC_COLS):
        mean_col = f"mean_{metric_col}"
        if mean_col not in summary.columns:
            continue
        for c, zcol in enumerate(z_cols):
            ax = axes[r, c]
            x = summary[mean_col]
            y = summary[zcol]
            mask = x.notna() & y.notna()

            ax.scatter(x[mask & ~any_flag], y[mask & ~any_flag],
                       s=18, alpha=0.5, color="tab:gray", label="ok")
            ax.scatter(x[mask & any_flag], y[mask & any_flag],
                       s=28, alpha=0.8, color="tab:red", label="flagged")
            ax.axvline(FLAG_THRESHOLDS[metric_col][1], color="grey", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(0, color="black", lw=0.6)

            # Fit trend line
            valid = mask.values
            if valid.sum() > 5:
                coef = np.polyfit(x[mask].values, y[mask].values, 1)
                xs = np.linspace(x[mask].min(), x[mask].max(), 80)
                ax.plot(xs, np.polyval(coef, xs), color="tab:orange", lw=1.2,
                        label=f"slope={coef[0]:+.3f}")

            # Label exemplars
            ex = summary[summary["desert_id"].isin(set(DESERTS.keys()))]
            for _, row in ex.iterrows():
                if pd.notna(row[mean_col]) and pd.notna(row[zcol]):
                    ax.annotate(row["desert_id"], (row[mean_col], row[zcol]),
                                xytext=(3, 3), textcoords="offset points", fontsize=7)

            ax.set_xlabel(METRIC_LABELS[metric_col], fontsize=9)
            ax.set_ylabel(zcol.replace("_", " "), fontsize=9)
            ax.set_title(f"{METRIC_LABELS[metric_col]} vs {zcol}", fontsize=9)
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading mappability stats ...")
    mq = _load_mappability()
    print(f"  {len(mq):,} windows loaded")

    print("Loading genomic positions ...")
    pos = _load_positions()
    print(f"  {len(pos):,} windows loaded")

    print("Merging positions + mappability ...")
    df = pos.merge(mq, on="element_id", how="inner")
    print(f"  {len(df):,} windows after merge")

    print("Loading z-scores (optional) ...")
    z_df = _load_z_scores()
    if z_df is not None:
        print(f"  {len(z_df):,} windows with z-scores")
    else:
        print("  Merged Gnocchi table not found; skipping z-score correlations")

    print("Loading all 633 desert coordinates ...")
    deserts_df = load_all_deserts()
    print(f"  {len(deserts_df):,} deserts")

    print("Labeling windows with desert IDs ...")
    labeled = label_deserts_fleet(df, deserts_df)
    in_desert = labeled[labeled["desert"].notna()].copy()
    print(f"  {len(in_desert):,} windows assigned to deserts ({in_desert['desert'].nunique()} unique deserts)")

    print("Computing per-desert mappability stats ...")
    summary = _desert_stats(in_desert, z_df)
    summary = _flag_deserts(summary)
    summary = summary.merge(
        deserts_df[["desert_id", "chrom", "start", "end"]].rename(
            columns={"start": "desert_start", "end": "desert_end"}
        ),
        on="desert_id", how="left",
    )

    out_summary = os.path.join(RESULTS_DIR, "mappability_desert_summary.tsv")
    summary.to_csv(out_summary, sep="\t", index=False)
    print(f"  wrote {out_summary}")

    flagged = summary[summary["any_flag"]]
    out_flagged = os.path.join(RESULTS_DIR, "mappability_flagged_deserts.tsv")
    flagged.to_csv(out_flagged, sep="\t", index=False)
    print(f"  wrote {out_flagged} ({len(flagged)} flagged deserts)")

    print("Generating visualizations ...")

    _save_exemplar_profiles(
        in_desert, z_df,
        os.path.join(RESULTS_DIR, "mappability_exemplar_profiles.png"),
    )
    print("  wrote mappability_exemplar_profiles.png")

    _save_fleet_distributions(
        summary,
        os.path.join(RESULTS_DIR, "mappability_fleet_distributions.png"),
    )
    print("  wrote mappability_fleet_distributions.png")

    _save_fleet_vs_zscore(
        summary,
        os.path.join(RESULTS_DIR, "mappability_fleet_vs_zscore.png"),
    )
    print("  wrote mappability_fleet_vs_zscore.png")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Fleet mappability summary (633 deserts) ===")
    for col in METRIC_COLS:
        mean_col = f"mean_{col}"
        vals = summary[mean_col].dropna()
        flag_col = f"flag_{col}"
        n_flagged = int(summary[flag_col].sum()) if flag_col in summary.columns else "n/a"
        print(f"  {METRIC_LABELS[col]:40s}  "
              f"fleet mean={vals.mean():.3f}  "
              f"fleet sd={vals.std():.3f}  "
              f"flagged={n_flagged}")

    print("\n=== Exemplar deserts ===")
    ex_rows = summary[summary["desert_id"].isin(set(DESERTS.keys()))].copy()
    for _, row in ex_rows.iterrows():
        flags = []
        for col in METRIC_COLS:
            fc = f"flag_{col}"
            if fc in row and row[fc]:
                flags.append(col)
        flag_str = ", ".join(flags) if flags else "none"
        parts = [f"  {row['desert_id']}:"]
        for col in METRIC_COLS:
            mean_col = f"mean_{col}"
            if mean_col in row:
                parts.append(f"{col}={row[mean_col]:.3f}")
        parts.append(f"flags=[{flag_str}]")
        print("  ".join(parts))

    print("\nDone.")


if __name__ == "__main__":
    main()

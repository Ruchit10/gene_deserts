#!/usr/bin/env python3
"""GC-content analysis at Gnocchi score extrema in gene deserts.

Asks: do the localised dips and spikes in z_adj within a desert coincide
with atypical GC content — which would implicate the GC adjustment step
as the source of anomalous scores?

Approach
--------
1. Classify every 1kb window inside a desert as a "dip", "spike", or
   "normal" using both absolute and within-desert relative thresholds.
2. Compute, per desert, the GC content difference between extreme and
   normal windows (effect size and t-test).
3. Produce spatial profiles for the 5 exemplars (z-trace + GC overlay
   coloured by class, plus a z-vs-GC scatter per exemplar).
4. Scan all 633 deserts to flag those with notable dips/spikes, then
   summarise how GC co-varies with those extrema fleet-wide.

Tunable thresholds (top of file)
---------------------------------
  ABS_DIP_THRESH    z_adj below this  → absolute dip   (default -2.0)
  ABS_SPIKE_THRESH  z_adj above this  → absolute spike  (default +2.0)
  REL_N_SD          also flag windows > N sd from desert mean (default 1.5)
  MIN_EXTREME_WIN   minimum dip/spike windows to call a desert "notable"

Outputs (all in results/)
--------------------------
  gc_extrema_desert_flags.tsv       per-desert flagging table (all 633)
  gc_extrema_comparison.tsv         GC stats for dip/spike/normal per notable desert
  gc_extrema_exemplar_{name}.png    3-panel spatial + z-vs-GC scatter for each exemplar
  gc_extrema_fleet_overview.png     fleet-level summary figure
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

from desert_utils import (
    DATA_DIR,
    DESERTS,
    DESERT_ORDER,
    RESULTS_DIR,
    label_deserts,
    label_deserts_fleet,
    load_all_deserts,
    load_features,
    load_gnocchi,
)

# ── Tunable thresholds ────────────────────────────────────────────────────────
ABS_DIP_THRESH    = -2.0   # absolute z_adj floor for "dip"
ABS_SPIKE_THRESH  =  2.0   # absolute z_adj ceiling for "spike"
REL_N_SD          =  1.5   # within-desert SD multiplier for relative classification
MIN_EXTREME_WIN   =  5     # min windows in a class to consider a desert "notable"
MIN_DESERT_WINDOWS = 10    # deserts with fewer windows are skipped
GC_SCALES         = ["1k", "10k"]   # GC feature scales to include
PRIMARY_GC_COL    = "GC_content_1k" # main GC column used for most comparisons


# ── Window classification ─────────────────────────────────────────────────────

def _classify(sub: pd.DataFrame) -> pd.Series:
    """Assign each row a class label using absolute + relative thresholds.

    Priority: absolute thresholds first; relative thresholds extend the
    classification for within-desert outliers that don't cross the genome-wide
    absolute cutoff (e.g. a desert that is uniformly at z = -1 may still have
    relative dips worth investigating).
    """
    z = sub["z_adj"].values
    n = len(z)
    classes = np.full(n, "normal", dtype=object)

    abs_dip   = z < ABS_DIP_THRESH
    abs_spike = z > ABS_SPIKE_THRESH
    classes[abs_dip]   = "dip"
    classes[abs_spike] = "spike"

    # Relative: only applied where not already classified
    normal_mask = classes == "normal"
    if normal_mask.sum() >= MIN_DESERT_WINDOWS:
        mu = np.nanmean(z[normal_mask])
        sd = np.nanstd(z[normal_mask], ddof=1)
        if sd > 0.1:
            rel_dip   = normal_mask & (z < mu - REL_N_SD * sd)
            rel_spike = normal_mask & (z > mu + REL_N_SD * sd)
            classes[rel_dip]   = "rel_dip"
            classes[rel_spike] = "rel_spike"

    return pd.Series(classes, index=sub.index, name="window_class")


# ── Per-desert statistics ─────────────────────────────────────────────────────

def _desert_stats(sub: pd.DataFrame) -> dict[str, Any]:
    """Compute per-desert flagging and GC stats for a classified window set."""
    z = sub["z_adj"].dropna()
    gc = sub[PRIMARY_GC_COL].dropna() if PRIMARY_GC_COL in sub.columns else pd.Series([], dtype=float)
    n = len(sub)

    n_dip        = int((sub["window_class"] == "dip").sum())
    n_spike      = int((sub["window_class"] == "spike").sum())
    n_rel_dip    = int((sub["window_class"] == "rel_dip").sum())
    n_rel_spike  = int((sub["window_class"] == "rel_spike").sum())
    within_sd    = float(z.std(ddof=1)) if len(z) > 1 else np.nan
    z_range      = float(z.max() - z.min()) if len(z) > 1 else np.nan

    row: dict[str, Any] = {
        "n_windows":    n,
        "n_dip":        n_dip,
        "n_spike":      n_spike,
        "n_rel_dip":    n_rel_dip,
        "n_rel_spike":  n_rel_spike,
        "dip_frac":     n_dip / max(n, 1),
        "spike_frac":   n_spike / max(n, 1),
        "within_sd":    within_sd,
        "z_range":      z_range,
        "has_dips":     (n_dip + n_rel_dip) >= MIN_EXTREME_WIN,
        "has_spikes":   (n_spike + n_rel_spike) >= MIN_EXTREME_WIN,
        "heterogeneous": (not np.isnan(within_sd)) and (within_sd > 1.0) and
                          ((n_dip + n_rel_dip + n_spike + n_rel_spike) >= MIN_EXTREME_WIN),
    }

    # GC correlation with z_adj
    if len(gc) >= 20 and PRIMARY_GC_COL in sub.columns:
        combined = sub[["z_adj", PRIMARY_GC_COL]].dropna()
        if len(combined) >= 20:
            r_p, p_p = pearsonr(combined["z_adj"], combined[PRIMARY_GC_COL])
            r_s, p_s = spearmanr(combined["z_adj"], combined[PRIMARY_GC_COL])
            row["gc_r_pearson"]  = float(r_p)
            row["gc_p_pearson"]  = float(p_p)
            row["gc_r_spearman"] = float(r_s)
            row["gc_p_spearman"] = float(p_s)
        else:
            row.update(gc_r_pearson=np.nan, gc_p_pearson=np.nan,
                       gc_r_spearman=np.nan, gc_p_spearman=np.nan)
    else:
        row.update(gc_r_pearson=np.nan, gc_p_pearson=np.nan,
                   gc_r_spearman=np.nan, gc_p_spearman=np.nan)

    # GC comparison between classes
    if PRIMARY_GC_COL in sub.columns:
        for cls in ["dip", "rel_dip", "spike", "rel_spike", "normal"]:
            vals = sub.loc[sub["window_class"] == cls, PRIMARY_GC_COL].dropna()
            row[f"mean_gc_{cls}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"n_gc_{cls}"]    = len(vals)

        # Effect size: (mean_gc_dip - mean_gc_normal) / sd_gc_normal
        all_dip    = sub.loc[sub["window_class"].isin(["dip", "rel_dip"]), PRIMARY_GC_COL].dropna()
        all_spike  = sub.loc[sub["window_class"].isin(["spike", "rel_spike"]), PRIMARY_GC_COL].dropna()
        normal_gc  = sub.loc[sub["window_class"] == "normal", PRIMARY_GC_COL].dropna()
        normal_sd  = float(normal_gc.std(ddof=1)) if len(normal_gc) > 1 else np.nan

        if len(all_dip) >= 3 and len(normal_gc) >= 3:
            stat, p = ttest_ind(all_dip, normal_gc, equal_var=False)
            row["gc_effect_dip"]  = float((all_dip.mean() - normal_gc.mean()) /
                                          max(normal_sd or 1e-9, 1e-9))
            row["gc_ttest_p_dip"] = float(p)
        else:
            row.update(gc_effect_dip=np.nan, gc_ttest_p_dip=np.nan)

        if len(all_spike) >= 3 and len(normal_gc) >= 3:
            stat, p = ttest_ind(all_spike, normal_gc, equal_var=False)
            row["gc_effect_spike"]  = float((all_spike.mean() - normal_gc.mean()) /
                                            max(normal_sd or 1e-9, 1e-9))
            row["gc_ttest_p_spike"] = float(p)
        else:
            row.update(gc_effect_spike=np.nan, gc_ttest_p_spike=np.nan)

    return row


# ── Plots ─────────────────────────────────────────────────────────────────────

_CLASS_COLOR = {
    "dip":       "tab:red",
    "rel_dip":   "tab:orange",
    "spike":     "tab:blue",
    "rel_spike": "tab:cyan",
    "normal":    "tab:gray",
}
_CLASS_ALPHA = {"dip": 0.18, "rel_dip": 0.10, "spike": 0.18, "rel_spike": 0.10}


def _mb(x: float, _: Any) -> str:
    return f"{x/1e6:.2f}"


def _save_exemplar_profile(name: str, sub: pd.DataFrame, out_path: str) -> None:
    """3-panel figure: z-trace + GC trace (coloured by class) + z vs GC scatter."""
    chrom, ds, de, note = DESERTS[name]
    sub = sub.sort_values("start").copy()
    pos = (sub["start"] + sub["end"]) / 2
    has_gc = PRIMARY_GC_COL in sub.columns

    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.3)
    ax_z   = fig.add_subplot(gs[0, 0])
    ax_gc  = fig.add_subplot(gs[1, 0], sharex=ax_z)
    ax_dz  = fig.add_subplot(gs[2, 0], sharex=ax_z)
    ax_scat = fig.add_subplot(gs[:, 1])

    # ── Panel 1: z-score trace with class shading ────────────────────────
    ax_z.plot(pos, sub["z_adj"],   lw=0.8, color="tab:blue",   alpha=0.9, label="z_adj",   zorder=3)
    ax_z.plot(pos, sub["z_unadj"], lw=0.8, color="tab:orange", alpha=0.75, label="z_unadj", zorder=3)
    ax_z.axhline(0, color="grey", lw=0.5, ls="--", zorder=1)
    ax_z.axhline(ABS_DIP_THRESH,   color="tab:red",  lw=0.6, ls=":", alpha=0.7)
    ax_z.axhline(ABS_SPIKE_THRESH, color="tab:blue", lw=0.6, ls=":", alpha=0.7)
    for cls in ["dip", "rel_dip", "spike", "rel_spike"]:
        for _, w in sub[sub["window_class"] == cls].iterrows():
            ax_z.axvspan(w["start"], w["end"],
                         alpha=_CLASS_ALPHA.get(cls, 0.1),
                         color=_CLASS_COLOR[cls], lw=0, zorder=2)
    ax_z.set_ylabel("Gnocchi z-score")
    ax_z.set_title(f"{name}  {chrom}:{ds:,}–{de:,}  ({note})", fontsize=10)
    ax_z.legend(fontsize=8, loc="best")
    plt.setp(ax_z.get_xticklabels(), visible=False)

    # ── Panel 2: GC_content_1k coloured by class ─────────────────────────
    if has_gc:
        for cls in _CLASS_COLOR:
            mask = sub["window_class"] == cls
            ax_gc.scatter(pos[mask], sub.loc[mask, PRIMARY_GC_COL],
                          s=5, alpha=0.7, color=_CLASS_COLOR[cls],
                          label=cls if mask.any() else None)
        ax_gc.set_ylabel("GC content (1 kb)")
        ax_gc.legend(fontsize=7, markerscale=2, loc="best")
    else:
        ax_gc.text(0.5, 0.5, "GC features not available",
                   ha="center", va="center", transform=ax_gc.transAxes)
    plt.setp(ax_gc.get_xticklabels(), visible=False)

    # ── Panel 3: delta_z = z_unadj - z_adj ───────────────────────────────
    if "delta_z" in sub.columns:
        dz = sub["delta_z"]
    else:
        dz = sub["z_unadj"] - sub["z_adj"]
    ax_dz.plot(pos, dz, lw=0.8, color="tab:purple", alpha=0.85)
    ax_dz.axhline(0, color="grey", lw=0.5, ls="--")
    ax_dz.set_ylabel("delta_z\n(unadj − adj)")
    ax_dz.set_xlabel(f"{chrom} (Mb)")
    ax_dz.xaxis.set_major_formatter(ticker.FuncFormatter(_mb))

    # ── Right panel: z_adj vs GC scatter, coloured by class ──────────────
    if has_gc:
        for cls in _CLASS_COLOR:
            mask = sub["window_class"] == cls
            if not mask.any():
                continue
            ax_scat.scatter(sub.loc[mask, PRIMARY_GC_COL], sub.loc[mask, "z_adj"],
                            s=7, alpha=0.6, color=_CLASS_COLOR[cls], label=cls)
        # trend line for all windows
        combined = sub[[PRIMARY_GC_COL, "z_adj"]].dropna()
        if len(combined) > 10:
            coef = np.polyfit(combined[PRIMARY_GC_COL], combined["z_adj"], 1)
            xs = np.linspace(combined[PRIMARY_GC_COL].min(),
                             combined[PRIMARY_GC_COL].max(), 60)
            r, _ = pearsonr(combined[PRIMARY_GC_COL], combined["z_adj"])
            ax_scat.plot(xs, np.polyval(coef, xs), color="black", lw=1.2,
                         label=f"all  r={r:+.2f}")
        ax_scat.axhline(0, color="grey", lw=0.5, ls="--")
        ax_scat.axhline(ABS_DIP_THRESH,   color="tab:red",  lw=0.6, ls=":", alpha=0.7)
        ax_scat.axhline(ABS_SPIKE_THRESH, color="tab:blue", lw=0.6, ls=":", alpha=0.7)
        ax_scat.set_xlabel(PRIMARY_GC_COL)
        ax_scat.set_ylabel("z_adj")
        ax_scat.set_title(f"{name}: z_adj vs GC\n(coloured by class)")
        ax_scat.legend(fontsize=7, markerscale=2)
    else:
        ax_scat.set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_fleet_overview(flags: pd.DataFrame, out_path: str) -> None:
    """4-panel fleet overview: dip/spike counts, GC effects, heterogeneity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    exemplars = set(DESERT_ORDER)

    # ── Panel A: scatter n_dip vs n_spike (size = n_windows) ─────────────
    ax = axes[0, 0]
    het = flags["heterogeneous"]
    size = 20 + 80 * (flags["n_windows"] - flags["n_windows"].min()) / max(
        1, flags["n_windows"].max() - flags["n_windows"].min())
    ax.scatter(flags.loc[~het, "n_dip"], flags.loc[~het, "n_spike"],
               s=size[~het], alpha=0.5, color="tab:gray", label="not heterogeneous")
    ax.scatter(flags.loc[het,  "n_dip"], flags.loc[het,  "n_spike"],
               s=size[het],  alpha=0.7, color="tab:purple", label="heterogeneous")
    for _, row in flags[flags["desert_id"].isin(exemplars)].iterrows():
        ax.annotate(row["desert_id"], (row["n_dip"], row["n_spike"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("n_dip windows (z < −2 or relative)")
    ax.set_ylabel("n_spike windows (z > +2 or relative)")
    ax.set_title("Dip vs spike window counts per desert")
    ax.legend(fontsize=8)

    # ── Panel B: within-desert SD distribution ────────────────────────────
    ax = axes[0, 1]
    ax.hist(flags["within_sd"].dropna(), bins=40, color="tab:gray",
            edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="tab:purple", lw=1, ls="--",
               label=f"heterogeneity cut (SD=1.0)")
    for _, row in flags[flags["desert_id"].isin(exemplars)].iterrows():
        if pd.notna(row["within_sd"]):
            ax.axvline(row["within_sd"], color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("Within-desert SD of z_adj")
    ax.set_ylabel("Number of deserts")
    ax.set_title("Internal z_adj variability across all deserts")
    ax.legend(fontsize=8)

    # ── Panel C: GC effect at dips across all deserts ─────────────────────
    ax = axes[1, 0]
    has_gc_effect = flags["gc_effect_dip"].notna()
    vals = flags.loc[has_gc_effect, "gc_effect_dip"].sort_values()
    colors = ["tab:red" if v < 0 else "tab:blue" for v in vals]
    ax.barh(np.arange(len(vals)), vals, color=colors, alpha=0.7, height=0.8)
    # Annotate exemplars
    for name in exemplars:
        match = flags[flags["desert_id"] == name]
        if not match.empty and pd.notna(match["gc_effect_dip"].iloc[0]):
            eff = match["gc_effect_dip"].iloc[0]
            idx = (vals == eff).argmax()
            ax.text(eff, idx, f" {name}", va="center", fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("GC effect at dips vs normal windows\n(Cohen's d: mean_dip_GC − mean_normal_GC) / sd_normal_GC")
    ax.set_yticks([])
    ax.set_title(f"GC difference at dip windows ({has_gc_effect.sum()} deserts)")

    # ── Panel D: Pearson r(z~GC_1k) per desert, vs within_sd ─────────────
    ax = axes[1, 1]
    both = flags[flags["gc_r_pearson"].notna() & flags["within_sd"].notna()]
    ax.scatter(both["gc_r_pearson"], both["within_sd"],
               s=18, alpha=0.5, color="tab:gray")
    for _, row in both[both["desert_id"].isin(exemplars)].iterrows():
        ax.annotate(row["desert_id"], (row["gc_r_pearson"], row["within_sd"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r  (z_adj ~ GC_content_1k)")
    ax.set_ylabel("Within-desert SD of z_adj")
    ax.set_title("GC–z correlation vs internal variability")

    fig.suptitle("Fleet-wide GC extrema analysis (all deserts)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_exemplar_windows(gn: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return per-exemplar window DataFrames from an already-labeled Gnocchi table."""
    return {name: gn[gn["desert"] == name].copy() for name in DESERT_ORDER}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading Gnocchi z-scores ...")
    gn = load_gnocchi(usecols=["chrom", "start", "end", "element_id", "z_adj", "z_unadj"])
    gn["delta_z"] = gn["z_unadj"] - gn["z_adj"]
    print(f"  {len(gn):,} windows")

    print("Loading all desert coordinates ...")
    deserts_df = load_all_deserts()

    # ── Label all windows with fleet desert IDs ───────────────────────────
    print("Labeling windows (fleet) ...")
    labeled_fleet = label_deserts_fleet(gn, deserts_df)
    desert_windows = labeled_fleet[labeled_fleet["desert"].notna()].copy()
    print(f"  {len(desert_windows):,} windows in deserts "
          f"({desert_windows['desert'].nunique()} unique deserts)")

    # ── Load GC features for desert windows only ──────────────────────────
    print("Loading GC features for desert windows ...")
    gc_cols = ["element_id"] + [f"GC_content_{s}" for s in GC_SCALES]
    feats = load_features(element_ids=desert_windows["element_id"].tolist())
    feats = feats[[c for c in gc_cols if c in feats.columns]]
    desert_windows = desert_windows.merge(feats, on="element_id", how="left")
    print(f"  GC joined; missing fraction: "
          f"{desert_windows[PRIMARY_GC_COL].isna().mean():.3f}")

    # ── Classify windows and build per-desert flag table ──────────────────
    print("Classifying windows and computing per-desert stats ...")
    class_parts: list[pd.Series] = []
    flag_rows: list[dict[str, Any]] = []
    for desert_id, sub in desert_windows.groupby("desert", sort=False):
        if len(sub) < MIN_DESERT_WINDOWS:
            continue
        cls = _classify(sub)
        class_parts.append(cls)
        row = {"desert_id": desert_id}
        row.update(_desert_stats(sub.assign(window_class=cls)))
        flag_rows.append(row)

    if class_parts:
        desert_windows = desert_windows.join(
            pd.concat(class_parts).rename("window_class")
        )
    else:
        desert_windows["window_class"] = "normal"

    flags = pd.DataFrame(flag_rows)
    flags = flags.merge(
        deserts_df[["desert_id", "chrom", "start", "end"]].rename(
            columns={"start": "desert_start", "end": "desert_end"}),
        on="desert_id", how="left",
    )

    out_flags = os.path.join(RESULTS_DIR, "gc_extrema_desert_flags.tsv")
    flags.to_csv(out_flags, sep="\t", index=False)
    print(f"  wrote {out_flags}")

    # ── GC comparison table (notable deserts only) ────────────────────────
    notable = flags[flags["has_dips"] | flags["has_spikes"]].copy()
    notable_ids = set(notable["desert_id"])
    comp_cols = [c for c in flags.columns
                 if c.startswith(("mean_gc_", "n_gc_", "gc_effect", "gc_ttest",
                                  "gc_r_", "gc_p_"))]
    out_comp = os.path.join(RESULTS_DIR, "gc_extrema_comparison.tsv")
    notable[["desert_id"] + comp_cols].to_csv(out_comp, sep="\t", index=False)
    print(f"  wrote {out_comp} ({len(notable)} notable deserts)")

    # ── Exemplar spatial + scatter profiles ──────────────────────────────
    print("\nGenerating exemplar profiles ...")
    gn_exemplar = label_deserts(gn)
    ex_feats = load_features(element_ids=gn_exemplar["element_id"].tolist())
    ex_feats = ex_feats[[c for c in gc_cols if c in ex_feats.columns]]
    gn_exemplar = gn_exemplar.merge(ex_feats, on="element_id", how="left")

    for name in DESERT_ORDER:
        sub = gn_exemplar[gn_exemplar["desert"] == name].copy()
        if sub.empty:
            print(f"  {name}: no windows found, skipping")
            continue
        sub["window_class"] = _classify(sub).values
        out_path = os.path.join(RESULTS_DIR, f"gc_extrema_exemplar_{name}.png")
        _save_exemplar_profile(name, sub, out_path)
        stats = _desert_stats(sub)
        n_ext = stats["n_dip"] + stats["n_spike"] + stats["n_rel_dip"] + stats["n_rel_spike"]
        r_gc  = stats.get("gc_r_pearson", np.nan)
        eff   = stats.get("gc_effect_dip", np.nan)
        print(f"  {name}: {n_ext} extreme windows, "
              f"r(z~GC_1k)={r_gc:+.3f}, GC effect at dips={eff:+.3f} SD")
        print(f"    wrote {out_path}")

    # ── Fleet overview figure ─────────────────────────────────────────────
    print("\nGenerating fleet overview figure ...")
    _save_fleet_overview(flags, os.path.join(RESULTS_DIR, "gc_extrema_fleet_overview.png"))
    print("  wrote gc_extrema_fleet_overview.png")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n=== Fleet summary ({len(flags)} deserts) ===")
    print(f"  Has notable dips  : {flags['has_dips'].sum()}")
    print(f"  Has notable spikes: {flags['has_spikes'].sum()}")
    print(f"  Heterogeneous     : {flags['heterogeneous'].sum()}")
    print(f"  Top |r(z~GC_1k)| deserts:")
    top_r = flags.dropna(subset=["gc_r_pearson"]).nlargest(10, "gc_r_pearson")
    for _, row in top_r.iterrows():
        marker = " <-- exemplar" if row["desert_id"] in DESERT_ORDER else ""
        print(f"    {row['desert_id']:8s}  r={row['gc_r_pearson']:+.3f}  "
              f"effect_dip={row.get('gc_effect_dip', np.nan):+.3f} SD{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()

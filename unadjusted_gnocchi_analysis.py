#!/usr/bin/env python3
"""
Recompute Gnocchi z-scores using unadjusted expected counts and compare
to the original (regional-feature-adjusted) scores for 5 exemplar gene deserts.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.desert_utils import DESERTS as deserts, DATA_DIR as DATA, RESULTS_DIR as OUT

# ── Step 1+2: Load, merge, recompute ─────────────────────────────────────────
print("Loading unadjusted expected sums …")
unadj = pd.read_csv(
    os.path.join(DATA, "expected_unadj_sum_by_region.txt"),
    sep="\t", header=None, names=["element_id", "expected_unadj"],
)
print(f"  {len(unadj):,} windows loaded")

print("Loading original Gnocchi table …")
gnocchi = pd.read_csv(
    os.path.join(DATA, "constraint_z_genome_1kb.qc.download.txt.gz"),
    sep="\t",
)
print(f"  {len(gnocchi):,} windows loaded")

print("Merging …")
df = gnocchi.merge(unadj, on="element_id", how="inner")
print(f"  {len(df):,} windows after inner join")

def compute_gnocchi_z(obs, exp):
    chi2 = (obs - exp) ** 2 / exp
    return np.where(obs < exp, np.sqrt(chi2), -np.sqrt(chi2))

df["z_unadj"] = compute_gnocchi_z(df["observed"].values, df["expected_unadj"].values)
df["z_adj"] = df["z"].values  # original adjusted z
df["delta_z"] = df["z_unadj"] - df["z_adj"]

combined_path = os.path.join(OUT, "gnocchi_adj_vs_unadj.tsv.gz")
df.to_csv(combined_path, sep="\t", index=False, compression="gzip")
print(f"  Saved combined table → {combined_path}")

# ── Helper: assign desert labels ─────────────────────────────────────────────
def label_deserts(frame):
    frame = frame.copy()
    frame["desert"] = None
    for name, (chrom, start, end, _) in deserts.items():
        mask = (
            (frame["chrom"] == chrom)
            & (frame["start"] >= start)
            & (frame["end"] <= end)
        )
        frame.loc[mask, "desert"] = name
    return frame

df = label_deserts(df)

# ── Step 3: Desert comparison ────────────────────────────────────────────────
print("\n=== Desert summary ===")
rows = []
for name, (chrom, start, end, note) in deserts.items():
    sub = df[df["desert"] == name]
    n = len(sub)
    rows.append({
        "desert": name,
        "note": note,
        "n_windows": n,
        "mean_z_adj": sub["z_adj"].mean(),
        "median_z_adj": sub["z_adj"].median(),
        "mean_z_unadj": sub["z_unadj"].mean(),
        "median_z_unadj": sub["z_unadj"].median(),
        "mean_delta_z": sub["delta_z"].mean(),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT, "desert_summary.tsv"), sep="\t", index=False)

# ── Plot A: side-by-side histograms per desert ───────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(8, 16), sharex=True)
for ax, (name, (chrom, start, end, note)) in zip(axes, deserts.items()):
    sub = df[df["desert"] == name]
    bins = np.linspace(-8, 8, 65)
    ax.hist(sub["z_adj"], bins=bins, alpha=0.6, label="adjusted", density=True)
    ax.hist(sub["z_unadj"], bins=bins, alpha=0.6, label="unadjusted", density=True)
    ax.set_title(f"{name} ({note})", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylabel("density")
axes[-1].set_xlabel("Gnocchi z-score")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "desert_histograms.png"), dpi=150)
plt.close(fig)
print("  Saved desert_histograms.png")

# ── Plot B: scatter z_adj vs z_unadj per desert ─────────────────────────────
desert_df = df[df["desert"].notna()].copy()
fig, ax = plt.subplots(figsize=(7, 7))
colors = dict(zip(deserts.keys(), plt.cm.tab10.colors[:5]))
for name in deserts:
    sub = desert_df[desert_df["desert"] == name]
    ax.scatter(sub["z_adj"], sub["z_unadj"], s=4, alpha=0.3,
               color=colors[name], label=name)
lims = [-10, 10]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("z (adjusted)")
ax.set_ylabel("z (unadjusted)")
ax.set_title("Adjusted vs Unadjusted Gnocchi — desert windows")
ax.legend(markerscale=3, fontsize=8)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "desert_scatter_adj_vs_unadj.png"), dpi=150)
plt.close(fig)
print("  Saved desert_scatter_adj_vs_unadj.png")

# ── Plot C: spatial profiles — one figure per desert ─────────────────────────
for name, (chrom, start, end, note) in deserts.items():
    sub = df[df["desert"] == name].sort_values("start")
    pos = (sub["start"] + sub["end"]) / 2

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(pos, sub["z_adj"],   lw=0.9, color="tab:blue",   label="z adjusted")
    ax.plot(pos, sub["z_unadj"], lw=0.9, color="tab:orange", label="z unadjusted")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("Gnocchi z-score")
    ax.set_title(f"{name}  {chrom}:{start:,}–{end:,}  ({note})", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    delta = sub["delta_z"] if "delta_z" in sub.columns else sub["z_unadj"] - sub["z_adj"]
    ax.fill_between(pos, delta, 0,
                    where=(delta >= 0), color="tab:orange", alpha=0.4, label="Δz > 0")
    ax.fill_between(pos, delta, 0,
                    where=(delta < 0),  color="tab:blue",   alpha=0.4, label="Δz < 0")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("Δz (unadj − adj)")
    ax.set_xlabel(f"{chrom} position")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f} Mb"))
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(OUT, f"desert_spatial_profile_{name}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")

# ── Step 4: Genome-wide sanity check ────────────────────────────────────────
print("\n=== Genome-wide sanity check ===")
for col, label in [("z_adj", "Adjusted"), ("z_unadj", "Unadjusted")]:
    vals = df[col].dropna()
    print(f"  {label:12s}  mean={vals.mean():.4f}  median={vals.median():.4f}  sd={vals.std():.4f}  n={len(vals):,}")

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-10, 10, 201)
ax.hist(df["z_adj"].dropna(), bins=bins, alpha=0.5, density=True, label="adjusted")
ax.hist(df["z_unadj"].dropna(), bins=bins, alpha=0.5, density=True, label="unadjusted")
ax.set_xlabel("Gnocchi z-score")
ax.set_ylabel("density")
ax.set_title("Genome-wide z-score distributions")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "genomewide_z_distributions.png"), dpi=150)
plt.close(fig)
print("  Saved genomewide_z_distributions.png")

print("\nDone.")

# Gene Desert Constraint Analysis

Diagnostic analyses of non-coding constraint (Gnocchi) scores across human gene deserts, with a focus on understanding how the regional-feature adjustment built into the Gnocchi pipeline affects per-window z-scores, and whether those effects are biologically interpretable or technical artefacts.

The central question: **do gene deserts that show anomalous constraint signatures (extreme z-scores, unusual spatial patterns) have properties — GC content, mappability, non-coding RNA content — that explain why Gnocchi behaves differently there?**

---

## Background

[Gnocchi](https://gnomad.broadinstitute.org) (gnomAD v3) quantifies non-coding constraint as a z-score comparing observed to expected variant counts in 1 kb windows across the genome. Expected counts are adjusted for 52 regional genomic features (GC content, recombination rate, repeat content, etc.) via a PCA-based model. This project investigates whether that adjustment introduces, removes, or mischaracterises constraint signals in gene-poor regions of the genome.

**Gene deserts** are defined here as large intergenic intervals (≥500 kb from the nearest gene) where Gnocchi behaviour may be driven more by technical covariates than genuine evolutionary constraint.

---

## Repository structure

```
gene_deserts/
├── utils/
│   └── desert_utils.py              # shared data-loading and labelling utilities
│
├── unadjusted_gnocchi_analysis.py   # entry point — builds the core merged table
├── distributions.py                 # Analysis B
├── spatial_acf.py                   # Analysis C
├── feature_profiling.py             # Analysis A
├── feature_score_correlations.py    # Analysis D
├── decomposition_approx.py          # Analysis E
├── lofo_approx.py                   # Analysis F
├── exemplar_investigations.py       # Analysis I
│
├── analysis_fleet_deserts.py        # fleet-scale (all 633 deserts)
├── mappability_gene_desert.py       # mapping quality / LCR / segdup diagnostics
├── desert_ncrna_landscape.py        # ncRNA annotation inside deserts
├── desert_gc_extrema.py             # GC content at z-score dips and spikes
├── extended_analyses/
│   ├── desert_gc_nonlinear.py       # non-linear GC-vs-z diagnostics (spline vs linear)
│   ├── desert_trinuc_context.py     # trinucleotide context composition bias diagnostics
│   ├── desert_background_selection.py # BGS proxy via edge-distance and recombination
│   ├── desert_replication_timing.py # replication timing overlay (GM12878)
│   ├── desert_lad_overlap.py        # LAD occupancy/concordance across 12 cell types
│   ├── desert_conserved_elements.py # phyloP conservation support for z spikes
│   └── desert_sv_overlap.py         # common large SV overlap (gnomAD v4.1)
│
├── data/                            # input data (not tracked in git — see below)
└── results/                         # all output figures and tables (tracked)
```

---

## Scripts

### `utils/desert_utils.py` — shared utilities

Central module imported by every analysis script. Defines:

- `DESERTS` — coordinates and notes for the 5 hand-curated exemplar gene deserts (GD513, GD198, GD588, GD158, GD167)
- `load_gnocchi()` — loads the merged adjusted/unadjusted Gnocchi table from `results/`
- `load_features()` — loads the 52-feature table with pickle caching (auto-rebuilds if cache is stale across pandas versions)
- `label_deserts()` — tags windows with exemplar desert IDs
- `load_all_deserts()` / `label_deserts_fleet()` — fleet-scale equivalents for all 633 deserts, using a per-chromosome `merge_asof` interval join for efficiency

---

### `unadjusted_gnocchi_analysis.py` — entry point

**Run this first.** Recomputes Gnocchi z-scores from unadjusted expected counts and merges them with the original (feature-adjusted) scores into a single table used by all downstream scripts.

Key outputs:
- `results/gnocchi_adj_vs_unadj.tsv.gz` — the merged table; every 1 kb window with `z_adj`, `z_unadj`, `delta_z = z_unadj − z_adj`
- `results/desert_summary.tsv` — per-exemplar summary statistics
- `results/desert_histograms.png`, `desert_scatter_adj_vs_unadj.png`, `desert_spatial_profiles.png`, `genomewide_z_distributions.png`

---

### Core exemplar analyses (A–F, I)

These operate on the 5 exemplar deserts and share data loaded via `desert_utils`.

| Script | Analysis | What it does |
|---|---|---|
| `feature_profiling.py` | A | Compares the 52 genomic feature values inside each exemplar to the genome-wide background; outputs a standardised deviation heatmap |
| `distributions.py` | B | Side-by-side z_adj vs z_unadj distribution statistics per exemplar; classifies whether the adjustment "fixes", "breaks", or is "neutral" on each desert's constraint anomaly |
| `spatial_acf.py` | C | Spatial autocorrelation of z_adj, z_unadj, and delta_z at lags 1–100 kb, compared against 200 same-length background stretches sampled genome-wide |
| `feature_score_correlations.py` | D | Pearson and Spearman correlation of all 52 features with z_adj, z_unadj, and delta_z, per exemplar and genome-wide; output as correlation heatmaps |
| `decomposition_approx.py` | E | Genome-wide ridge regression of `delta_z ~ 52 features`; per-window feature contributions reveal which covariates drive the z_adj − z_unadj gap in each desert |
| `lofo_approx.py` | F | Leave-one-feature-out (LOFO) analysis using the ridge from E; ranks features by the z-shift they produce when zeroed out per desert |
| `exemplar_investigations.py` | I | Targeted spatial deep-dives for GD198 (near 8p23 inversion), GD167 (recombination structure), and GD513 (highest constraint); overlays possible/observed/expected counts with recombination rate and cDNM density |

---

### Fleet-scale analyses (all 633 deserts)

#### `analysis_fleet_deserts.py`

Scales the adjusted vs unadjusted Gnocchi comparison from the 5 exemplars to all 633 deserts in `data/hg38_desert_nczscores.txt`.

Classifies each desert by how the adjustment changes its score:
- **Inflated by adjustment** — z_adj more positive than z_unadj (mean delta_z < −1)
- **Deflated by adjustment** — z_adj more negative than z_unadj (mean delta_z > +1)
- **Sign-flip** — adjustment reverses the direction of the constraint call
- **Neutral** — |mean delta_z| < 0.5

Key outputs: `results/fleet_summary.tsv`, `fleet_scatter_adj_vs_unadj.png`, `fleet_waterfall_delta_z.png`, `fleet_volcano_delta_z.png`, `fleet_chrom_delta_z.png`, `fleet_distributions.png`

---

#### `mappability_gene_desert.py`

Assesses whether anomalous Gnocchi scores could be explained by technical mapping issues — low mapping quality, high low-complexity region (LCR) fraction, or segmental duplication overlap.

For each 1 kb window: `MQ.mean`, `LCR` fraction, `Segdup` fraction are joined from the mappability stats file. Deserts are flagged against thresholds (default: MQ < 40, LCR > 30%, Segdup > 10%) and the flagged set is saved separately.

Key outputs: `results/mappability_desert_summary.tsv`, `mappability_flagged_deserts.tsv`, `mappability_exemplar_profiles.png`, `mappability_fleet_distributions.png`, `mappability_fleet_vs_zscore.png`

---

#### `desert_ncrna_landscape.py`

Maps non-coding RNA genes from GENCODE v39 onto all 633 deserts to ask whether unknown functional elements could explain localised constraint signals.

Target biotypes: `lncRNA`, `miRNA`, `snRNA`, `snoRNA`, `misc_RNA`, `Mt_tRNA`, `scaRNA`. For each exemplar, produces a spatial profile with a gene-track panel showing ncRNA positions as coloured rectangles, overlaid on the z_adj trace. Windows overlapping an ncRNA are compared to non-overlapping windows within the same desert.

Key outputs: `results/ncrna_desert_catalog.tsv`, `ncrna_desert_summary.tsv`, `ncrna_exemplar_{name}.png` (×5), `ncrna_fleet_summary.png`, `ncrna_zscore_context.png`

---

#### `desert_gc_extrema.py`

Targeted GC-content analysis at z-score dips and spikes, replacing the earlier GD588-only correlation panel from Analysis D.

Each 1 kb window is classified as `dip` / `spike` / `normal` using both absolute thresholds (z < -2 or z > +2) and within-desert relative thresholds (> 1.5 SD from the desert mean). Beyond the original per-desert effect-size test (Cohen's d + Welch t-test), the script now adds:

- spatial GC line profiles (multi-scale: 1k/10k/100k/1M) instead of point clouds
- GC deviation from each desert baseline (`GC_1k - mean_desert_GC_1k`)
- local-vs-regional contrast (`GC_1k - GC_100k`)
- GC-detrended z residual profiles (`z_adj` residual after linear GC fit)
- lagged cross-correlation peak lag between z and GC
- binned GC-vs-z trend plots (quantile bins with mean +/- SEM)
- fleet-level GC-effect volcano plot and pooled class-density summary

All thresholds are defined as named constants at the top of the file and can be tuned without touching any other code.

Key outputs: `results/gc_extrema_desert_flags.tsv` (all 633 deserts), `gc_extrema_comparison.tsv` (notable deserts only, includes CCF/residual stats), `gc_extrema_exemplar_{name}.png` (x5 enhanced panels), `gc_extrema_fleet_overview.png` (includes volcano + pooled density)

---

#### Extended anomaly diagnostics (new)

| Script | Focus | Primary outputs |
|---|---|---|
| `desert_gc_nonlinear.py` | Tests whether persistent GC-linked anomalies (e.g. GD588 class) are better explained by non-linear GC relationships than linear correction | `gc_nonlinear_desert_summary.tsv`, `gc_nonlinear_exemplar_{name}.png`, `gc_nonlinear_fleet_overview.png` |
| `desert_trinuc_context.py` | Quantifies context-spectrum shifts (desert vs genome) and flags trinucleotide-composition bias in unadjusted expectations | `trinuc_context_desert_summary.tsv`, `trinuc_context_exemplar_{name}.png`, `trinuc_context_fleet_overview.png` |
| `desert_background_selection.py` | B-value proxy analysis using edge-distance gradients + recombination interaction (no external B-map required) | `bgs_desert_summary.tsv`, `bgs_exemplar_{name}.png`, `bgs_fleet_overview.png` |
| `desert_replication_timing.py` | Joins GM12878 replication timing and tests RT associations with z_adj/z_unadj/delta_z (including partial corr vs GC) | `replication_timing_desert_summary.tsv`, `replication_timing_exemplar_{name}.png`, `replication_timing_fleet_overview.png` |
| `desert_lad_overlap.py` | Computes LAD occupancy across 12 LAD tracks and constitutive LAD effects on desert scores | `lad_desert_summary.tsv`, `lad_track_overlap_by_desert.tsv`, `lad_exemplar_{name}.png`, `lad_fleet_overview.png` |
| `desert_conserved_elements.py` | Uses phyloP447way primate conservation to ask whether desert z spikes are conservation-backed | `conserved_elements_desert_summary.tsv`, `conserved_elements_exemplar_{name}.png`, `conserved_elements_fleet_overview.png` |
| `desert_sv_overlap.py` | Tests overlap with common large SVs (INV/DEL/DUP/CPX/CNV) from gnomAD v4.1 | `sv_desert_summary.tsv`, `sv_flagged_deserts.tsv`, `sv_fleet_overview.png` |

---

## Running the analyses

Scripts must be run from the repository root (not from inside `data/` or `results/`):

```bash
cd gene_deserts/

# 1. Build the merged Gnocchi table — required before everything else
python unadjusted_gnocchi_analysis.py

# 2. Core exemplar analyses (order-independent after step 1)
python feature_profiling.py
python distributions.py
python spatial_acf.py
python feature_score_correlations.py
python decomposition_approx.py   # slow — fits genome-wide ridge
python lofo_approx.py            # requires decomposition_approx output
python exemplar_investigations.py

# 3. Fleet and diagnostic analyses (require step 1)
python analysis_fleet_deserts.py
python mappability_gene_desert.py
python desert_ncrna_landscape.py
python desert_gc_extrema.py      # requires feature cache (auto-built on first run)

# 4. Extended anomaly diagnostics (all require step 1; order-independent)
python extended_analyses/desert_gc_nonlinear.py
python extended_analyses/desert_trinuc_context.py
python extended_analyses/desert_background_selection.py
python extended_analyses/desert_replication_timing.py
python extended_analyses/desert_lad_overlap.py
python extended_analyses/desert_conserved_elements.py
python extended_analyses/desert_sv_overlap.py
```

The feature cache (`results/_features_cache.pkl.gz`) is built automatically on the first run of any script that needs it. If you switch Python/pandas environments it will be automatically detected as stale and rebuilt.

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
scikit-learn   # ridge regression in decomposition_approx.py and lofo_approx.py
statsmodels    # ACF in spatial_acf.py
```

All tested on Python 3.12. Install via:

```bash
pip install numpy pandas matplotlib scipy scikit-learn statsmodels
```

---

## Data

Data files live in `data/` and are not tracked in this repository. All files are publicly available or derived from public resources.

| File | Source | Description |
|---|---|---|
| `constraint_z_genome_1kb.qc.download.txt.gz` | [gnomAD v3 / Gnocchi](https://gnomad.broadinstitute.org/downloads) | Per-window observed, expected (adjusted), and z-scores for all 1 kb windows passing QC |
| `expected_unadj_sum_by_region.txt` | gnomAD v3 (Gnocchi pipeline) | Unadjusted (pre-PCA) expected variant counts per 1 kb window |
| `expected_counts_per_context_methyl_genome_1kb.txt.gz` | gnomAD v3 (Gnocchi pipeline) | Trinucleotide-context + methylation expected counts per window |
| `genomic_features13_genome_1kb.txt.gz` | gnomAD v3 (Gnocchi pipeline) | 52-column feature table (GC content, recombination, repeat content, cDNM density, etc.) at 4 spatial scales per 1 kb window |
| `gnocchi.windows.mq.lcr.segdup.stats.tsv.gz` | gnomAD v3 (Gnocchi pipeline) | Per-window mapping quality, LCR, and segmental duplication statistics |
| `hg38_desert_nczscores.txt` | This project | Coordinates and aggregate z-score summaries for all 633 gene deserts defined on GRCh38 |
| `gencode.v39.annotation.gtf.gz` | [GENCODE v39](https://www.gencodegenes.org/human/release_39.html) | Gene annotations for GRCh38; used to map ncRNA positions into deserts |
| `dnm01_10x_ft_logit_regularized_coef_z_3mer_context_flnk_1k-1M.txt` | gnomAD v3 (Gnocchi pipeline) | De novo mutation model coefficients |
| `desert.ncz.exemplars.apr2026.txt` | This project | Coordinates of the 5 hand-curated exemplar gene deserts |
| `GM12878_hg38_smoothed.txt` | [Koren Lab](https://www.thekorenlab.org/data) | Per-position replication timing values used in `desert_replication_timing.py` |
| `LADs/*.bed.gz` | [LAD atlas](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155244) (12 tissues/cell types) | Cell-type-specific LAD intervals used to compute occupancy and constitutive LAD labels |
| `gnomad.v4.1.sv.sites.bed.gz` | [gnomAD SV v4.1](https://gnomad.broadinstitute.org/downloads) | Structural variant catalog used for common large-SV desert-overlap analysis |
| `phyloP447wayPrimates.txt.gz` | UCSC / Zoonomia (447-way primate phyloP summary) | Conservation-score summary blocks aggregated to 1 kb windows in `desert_conserved_elements.py` |

For the Gnocchi-derived files, see the [gnomAD non-coding constraint preprint](https://www.biorxiv.org/content/10.1101/2022.03.20.485034) and the methods PDF `data/nc_constraint_gnomadv3_adj_r_methods.pdf` included in this repo.

---

## Results

All figures and tables are committed to `results/`. Interpretation of findings is documented separately in `gd_analyses.md`.

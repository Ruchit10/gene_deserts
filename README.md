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
├── desert_utils.py                  # shared data-loading and labelling utilities
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
│
├── data/                            # input data (not tracked in git — see below)
└── results/                         # all output figures and tables (tracked)
```

---

## Scripts

### `desert_utils.py` — shared utilities

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

Each 1 kb window is classified as `dip` / `spike` / `normal` using both absolute thresholds (z < −2 or z > +2) and within-desert relative thresholds (> 1.5 SD from the desert mean). The GC content difference between extreme and normal windows is tested per desert (Cohen's d effect size + Welch t-test), and Pearson r(z_adj ~ GC_content_1k) is computed for all 633 deserts to identify where the GC adjustment most strongly tracks the z-score profile.

All thresholds are defined as named constants at the top of the file and can be tuned without touching any other code.

Key outputs: `results/gc_extrema_desert_flags.tsv` (all 633 deserts), `gc_extrema_comparison.tsv` (notable deserts only), `gc_extrema_exemplar_{name}.png` (×5), `gc_extrema_fleet_overview.png`

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

Data files live in `data/` and are not tracked in this repository (~1.2 GB total). All files are either publicly available or part of the gnomAD v3 resource.

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

For the Gnocchi-derived files, see the [gnomAD non-coding constraint preprint](https://www.biorxiv.org/content/10.1101/2022.03.20.485034) and the methods PDF `data/nc_constraint_gnomadv3_adj_r_methods.pdf` included in this repo.

---

## Results

All figures and tables are committed to `results/`. Interpretation of findings is documented separately in `gd_analyses.md`.

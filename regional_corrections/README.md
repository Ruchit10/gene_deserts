# Regional feature adjustment for chrX PAR1/PAR2

gnomAD's released Gnocchi scores for chrX PAR were **never regionally adjusted**. The pipeline corrects
expected variant counts for 13 regional genomic features per trinucleotide context, but the published
feature matrix (`genomic_features13_genome_1kb.txt`) is chr1–22 only. Every chrX window therefore lost its
features to the per-context `.dropna()` and hit the `.fillna(1)` fallback, so the shipped `gnocchi` column
for PAR equals the *unadjusted* score.

Confirmed directly: `expected == expected_unadj` for all 2,497 PAR windows, whereas **zero** of 1,984,900
autosomal windows have `rr` exactly 1.0.

This directory regenerates the feature matrix for PAR and applies the released per-context models to it.

## Pipeline

```bash
# 1. verify the feature builder reproduces published values (all 13 features, autosomal)
python regional_corrections/build_par_features.py --validate

# 2. build the PAR feature matrix -> data/genomic_features13_chrX_par_1kb.txt.gz
python regional_corrections/build_par_features.py

# 3. verify the rr machinery reproduces published expected counts (unmasked, autosomal)
python regional_corrections/compute_par_rr.py --validate

# 4. compute rr + adjusted z for PAR -> results/par_regional_adjustment.tsv.gz
python regional_corrections/compute_par_rr.py

# 5. optional: measure what masking costs, on autosomes where truth exists
python regional_corrections/compute_par_rr.py --mask-ablation

# 6. regenerate the PAR figures/tables on adjusted scores
python par_gnocchi_analysis.py
```

Upstream sources (`run_nc_constraint_gnomad_v31_main.py`, `analyze_individual_feature_effects.py`,
`generate_context_models.py`, `constraint_basics.py`, `generic.py`, `nc_constraint_utils.py`) are kept here
for reference. `generate_context_models.py` is **not** used — it refits models, which turned out to be
unnecessary; note also that it came from another project and its `cocktail_DNMs_by_context_methyl.txt` input
is not the file this project would use.

## Feature reliability in PAR — verified, not assumed

Every source track in `data/genomic_features13/` was checked against PAR1 (chrX:10,001–2,781,479) and PAR2
(chrX:155,701,383–156,030,895).

| Feature | Verified PAR coverage | Verdict |
|---|---|---|
| `dist2telo` | chrX telomeres present; `normalize_dist2gaps_1kb.py` already includes chrX | **KEEP** |
| `dist2cent` | chrX centromere 58,605,579–62,412,542 | **KEEP** |
| `GC_content` | computed from reference sequence | **KEEP** |
| `LCR` | PAR1 19.80%, PAR2 2.05% of bp covered | **KEEP** |
| `SINE` | PAR1 25.07%, PAR2 7.91% | **KEEP** |
| `LINE` | PAR1 6.92%, PAR2 43.22% | **KEEP** |
| `CpG_island` | PAR1 2.71% (92 islands), PAR2 0.21% (2 islands) | **KEEP** |
| `recomb_male` | **0 chrX rows** in `genetic.map.final.pat.gor.bed` | **MASK** |
| `recomb_female` | chrX spans 3,532,526–154,781,072: begins 751 kb past PAR1's end, ends 920 kb before PAR2 — **0 PAR overlap** | **MASK** |
| `cDNM_maternal_05M` | **0 chrX rows** in `Goldmann_18_S5_cDNMs_F.lft38.bed` | **MASK** |
| `cDNM_paternal_05M` | **0 chrX rows** in `Goldmann_18_S5_cDNMs_M.lft38.bed` (also never selected) | **MASK** |
| `met_sperm` | source track not available (only GEO URLs in `Sperm_files.txt`) | **MASK** |
| `Nucleosome` | source track not available | **MASK** |

For recombination and cDNM the data is genuinely *absent* for PAR, so masking is forced rather than a
judgement call.

**Silent-zero trap.** A coverage computation against a track with no chrX intervals returns `0.0`, not
`NaN` — a plausible-looking low value that would bias `rr`. The masked columns are therefore written as
`NaN` in the feature matrix, and pinned to standardized 0 ("genome-average window") only at model-input
time.

## Conventions recovered by validation

The upstream shell recipe leaves several things implicit. These were pinned by reproducing published values
and are now bit-exact (`--validate`: **28/28 columns, r = 1.00000, max |diff| = 0.0000**):

- **Distance features are not scale-invariant.** Each scale is measured from the midpoint of *that scale's*
  flank window. Verified: `chr1-26000-27000` → `dist2telo_1k` = (26500−1)/248,956,422×100 = 0.010644 and
  `dist2telo_1M` = (500000−1)/248,956,422×100 = 0.200838, both matching published to all digits.
- **Coverage requires merging overlapping intervals.** Summing raw interval overlaps double-counts where
  repeat annotations overlap; without the merge, SINE/LINE drift up to ~3.4 percentage points high.
- **GC counts N in the denominator**: `100×(G+C)/(end−start)`, matching `hgGcPercent -doGaps`. A window with
  360 Ns reproduces the published value under this rule and not under N-exclusion.
- **Flank windows slide at chromosome edges** to preserve full width rather than truncating
  (`chrX-156031000-156032000` → 1M flank `[155040895, 156040895)`). The provided flank BEDs already cover
  chrX, so this needed no reimplementation.
- **`ft_mean_std.txt` is headerless** and its row order matches each context's `ft_sel` exactly (checked for
  all 32 contexts) — it is the authoritative column order for the PCA.
- The released pickles need a **pandas <2 unpickling shim** (`pandas.core.indexes.numeric`).

## Validation results

| Gate | Result |
|---|---|
| Feature builder vs published matrix | **28/28 columns**, r = 1.00000, max \|diff\| = 0.0000 |
| `rr` machinery vs published `expected` (unmasked, autosomal) | corr **0.99992905**, median rel. err **4.2e-09**, 99.997% within 0.1% |
| `pass_qc` recomputed from `pct_pass`/coverage/`possible` | 2497/2497 agree (1,146 pass) |
| PAR values outside autosomal training range | 0.443% — only `LINE_1M` in PAR2, marginally above max (50.85 vs 48.13) |

The rr gate is a **unit test of this code**, run unmasked with all 13 features on autosomal windows where
ground truth exists (`true rr = expected / expected_unadj`). It produces no autosomal results. PAR offers no
ground truth, so without it a bug and the masking effect would be indistinguishable.

### What masking costs (`--mask-ablation`, measured on autosomes)

| Configuration | rr_eff mean | rr_eff sd | median shift |
|---|---|---|---|
| all 13 features | 0.9810 | 0.0936 | −4.65% |
| 6 masked (PAR config) | 0.9851 | 0.0700 | −3.11% |

corr(full, masked) = **0.85**, SD retained **75%**, masking bias **+1.7%** on expected. So of PAR1's
+14.8% median shift, roughly +1.7 pp is attributable to masking and the remainder is genuine signal from
PAR1's sequence composition (GC 47.8% vs 41.0% autosomal, CpG islands 2.9% vs 0.8%, SINE 29.8% vs 13.9%).

## Result

All 2,497 PAR windows now have `rr ≠ 1` (previously all exactly 1).

| Region | rr mean (sd) | median expected shift | z_unadj mean | z_adj mean |
|---|---|---|---|---|
| PAR1 (n=2,187) | 1.166 (0.146) | +14.79% | −0.977 | **+0.634** |
| PAR2 (n=310) | 1.075 (0.080) | +6.60% | −1.237 | **−0.357** |

For context, the autosomal adjustment moves expected by a median of 4.5%, so PAR1's shift is unusually
large — consistent with its extreme sequence composition.

## Limitation that belongs with every number here

`recomb_male` is selected in **31 of 32 contexts** — the most-used feature in the model — and has no PAR
coverage at all, while obligate male recombination (~20× genome average) is PAR1's defining feature. This
correction accounts for sequence and annotation composition, **not** recombination. PAR1's high GC is itself
partly a consequence of that recombination via GC-biased gene conversion, so `GC_content` absorbs some of it
incidentally, but that is not a substitute.

Separately, masking leaves some contexts thin: `AAA`, `AAG`, `ACG`, `GAC` and `GAT` retain only one
informative feature each. `ACG` is a CpG context, and CpG transitions dominate expected counts.

"""Shared utilities for Gnocchi/Roulette constraint-benchmark analyses.

Houses:
  - GENCODE-derived genome-wide non-coding region categories (Promoter,
    5'/3' UTR, Intron, Distal intergenic) used by roulette_comparison.py's
    --region-scope noncoding path. ENCODE cCRE/enhancer and CTCF/TF
    ChIP-seq categories are deferred until those BED files are added to
    data/. Categories are non-exclusive: a window may belong to more than
    one at once (e.g. is_promoter and is_intron both True).
  - The shared Roulette-vs-Gnocchi z-score computation (load_roulette_z_table),
    used by both roulette_comparison.py and gnocchi_paper_benchmarks.py so
    the load/merge/scale logic lives in exactly one place.
  - Loaders (+ a missing-file-tolerant manifest) for the Gnocchi-paper
    benchmark tables staged in data/gnocchi_benchmark_data/, plus the
    enrichment/ROC/enhancer-rollup statistics and plotting-style helpers
    used by gnocchi_paper_benchmarks.py, ported from the style of the
    paper's own fig_utils.py/efig_utils.py (github.com/atgu/gnomad_nc_constraint).
"""

from __future__ import annotations

import gzip
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from utils.desert_utils import (
    DATA_DIR,
    RESULTS_DIR,
    aggregate_windows_over_regions,
    label_deserts_fleet,
    load_all_deserts,
    load_gnocchi,
    merge_intervals,
    window_interval_overlap,
)

# ── Roulette z-score computation (shared by roulette_comparison.py and ─────
# gnocchi_paper_benchmarks.py) ───────────────────────────────────────────────

# Diploid-calibrated Roulette expected counts (default, desert-only); needs a
# factor of 2 to reach the haploid gnomAD basis.
ROULETTE_PATH_DIPLOID = os.path.join(DATA_DIR, "roulette_gd_relative_mu_agg_1kb.tsv.bgz")
# Haploid-calibrated Roulette expected counts (desert-only); already on the
# haploid basis, so no factor of 2 is applied.
ROULETTE_PATH_HAPLOID = os.path.join(DATA_DIR, "roulette_v3_hap_gd_relative_mu_agg_1kb.tsv.bgz")
# Genome-wide, non-desert-restricted Roulette expected counts. Despite sharing
# the diploid file's 4-column schema (element_id, mu, exp, n_variants), this
# file is calibrated on the HAPLOID model, so it always pairs with
# HAPLOID_FACTOR -- there is no genome-wide diploid-calibrated file.
ROULETTE_PATH_NONCODING = os.path.join(DATA_DIR, "roulette_nc_relative_mu_agg_1kb.tsv.bgz")

# Roulette enumerates 3 alternate alleles per base, so a fully-covered 1kb
# window has 1000 * 3 = 3000 possible substitutions.
MAX_COVERAGE = 3000

# Roulette rates in the default (diploid) file need x2 to reach gnomAD's
# haploid basis; the haploid-calibrated files already use a factor of 1.
DIPLOID_FACTOR = 2.0
HAPLOID_FACTOR = 1.0


def compute_gnocchi_z(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Signed chi deviation, identical to the Gnocchi z definition."""
    obs = np.asarray(obs, dtype=float)
    exp = np.asarray(exp, dtype=float)
    exp = np.clip(exp, 1e-12, None)
    chi2 = (obs - exp) ** 2 / exp
    return np.where(obs < exp, np.sqrt(chi2), -np.sqrt(chi2))


def load_roulette_z_table(
    roulette_path: str,
    scale_factor: float,
    gnocchi_usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load Roulette aggregated expected counts, merge onto the Gnocchi
    table, and compute z_roulette/oe_roulette (+ oe_adj/oe_unadj/delta_z
    when the relevant Gnocchi columns are present).

    Shared by roulette_comparison.py's --region-scope paths and
    gnocchi_paper_benchmarks.py, so this load/merge/scale logic (previously
    duplicated) lives in exactly one place.
    """
    print("Loading Roulette aggregated expected counts ...")
    roulette = pd.read_csv(roulette_path, sep="\t", compression="gzip")
    numeric_cols = ["exp", "n_variants"]
    if "mu" in roulette.columns:
        numeric_cols.append("mu")
    for col in numeric_cols:
        roulette[col] = pd.to_numeric(roulette[col], errors="coerce")
    footer = roulette["element_id"].isna() | (roulette["element_id"].astype(str) == "NA")
    if footer.any():
        print(f"  filtering {int(footer.sum())} footer/total row(s)")
    roulette = roulette[~footer].copy()
    roulette = roulette.rename(columns={"exp": "exp_roulette_raw"})

    print("Loading merged Gnocchi table ...")
    usecols = gnocchi_usecols or [
        "chrom", "start", "end", "element_id",
        "possible", "observed", "expected", "expected_unadj",
        "z_adj", "z_unadj",
    ]
    gn = load_gnocchi(usecols=usecols)
    print(f"  Gnocchi rows: {len(gn):,}")
    print(f"  Roulette rows (after footer filter): {len(roulette):,}")

    print("Merging on element_id ...")
    merge_cols = ["element_id", "exp_roulette_raw", "n_variants"]
    if "mu" in roulette.columns:
        merge_cols.append("mu")
    df = gn.merge(roulette[merge_cols], on="element_id", how="inner")
    if "mu" in df.columns:
        df = df.drop(columns=["mu"])
    print(f"  merged rows: {len(df):,}")

    valid = (
        np.isfinite(df["exp_roulette_raw"]) & (df["exp_roulette_raw"] > 0)
        & np.isfinite(df["n_variants"]) & (df["n_variants"] > 0)
        & np.isfinite(df["possible"]) & (df["possible"] > 0)
        & np.isfinite(df["observed"])
        & np.isfinite(df["expected_unadj"]) & (df["expected_unadj"] > 0)
    )
    before = len(df)
    df = df[valid].copy()
    print(f"  retained valid rows: {len(df):,} / {before:,}")

    n_low_cov = int((df["n_variants"] < MAX_COVERAGE).sum())
    print(f"  windows with incomplete Roulette coverage (n_variants<{MAX_COVERAGE}): {n_low_cov:,}")

    print(f"Building comparable Roulette expected (scale_factor={scale_factor:.0f} + accessibility) ...")
    df["coverage"] = df["n_variants"] / MAX_COVERAGE
    exp_per_site = df["exp_roulette_raw"] / df["n_variants"]
    df["exp_roulette"] = scale_factor * exp_per_site * df["possible"]
    df["oe_roulette"] = df["observed"] / df["exp_roulette"]
    df["z_roulette"] = compute_gnocchi_z(df["observed"].values, df["exp_roulette"].values)

    if "expected" in df.columns:
        df["oe_adj"] = df["observed"] / df["expected"]
    if "expected_unadj" in df.columns:
        df["oe_unadj"] = df["observed"] / df["expected_unadj"]
    if "z_adj" in df.columns and "z_unadj" in df.columns:
        df["delta_z"] = df["z_unadj"] - df["z_adj"]

    return df


# ── GENCODE-derived genome-wide non-coding region categories ───────────────

GTF_FILE = os.path.join(DATA_DIR, "gencode.v39.annotation.gtf.gz")
PROMOTER_FLANK_BP = 2000
REGION_CACHE = os.path.join(RESULTS_DIR, "_gencode_region_cache.pkl.gz")
REGION_CATEGORIES = ["gene_body", "promoter", "five_utr", "three_utr", "intron"]

_RELEVANT_FEATURES = {"gene", "transcript", "exon", "CDS", "UTR"}
_ATTR_RE = re.compile(r'(\w+) "([^"]+)"')


def _parse_gtf_records() -> dict[str, pd.DataFrame]:
    """Single streaming pass over the GENCODE GTF, keeping the record types
    needed to derive gene bodies, promoters, UTR 5'/3' split, and introns.

    No gene_type filtering: distal-intergenic must exclude any annotated
    gene, not just protein-coding ones.
    """
    gene_rows: list[dict] = []
    transcript_rows: list[dict] = []
    exon_rows: list[dict] = []
    cds_rows: list[dict] = []
    utr_rows: list[dict] = []

    with gzip.open(GTF_FILE, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in _RELEVANT_FEATURES:
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1  # 1-based closed -> 0-based half-open
            end = int(fields[4])
            strand = fields[6]
            attrs = dict(_ATTR_RE.findall(fields[8]))
            transcript_id = attrs.get("transcript_id", "")

            if feature == "gene":
                gene_rows.append({"chrom": chrom, "start": start, "end": end})
            elif feature == "transcript":
                transcript_rows.append({
                    "chrom": chrom, "start": start, "end": end,
                    "strand": strand, "transcript_id": transcript_id,
                })
            elif feature == "exon":
                exon_rows.append({
                    "chrom": chrom, "start": start, "end": end,
                    "transcript_id": transcript_id,
                })
            elif feature == "CDS":
                cds_rows.append({
                    "chrom": chrom, "start": start, "end": end,
                    "transcript_id": transcript_id,
                })
            elif feature == "UTR":
                utr_rows.append({
                    "chrom": chrom, "start": start, "end": end,
                    "strand": strand, "transcript_id": transcript_id,
                })

    return {
        "gene": pd.DataFrame(gene_rows, columns=["chrom", "start", "end"]),
        "transcript": pd.DataFrame(
            transcript_rows, columns=["chrom", "start", "end", "strand", "transcript_id"]),
        "exon": pd.DataFrame(exon_rows, columns=["chrom", "start", "end", "transcript_id"]),
        "cds": pd.DataFrame(cds_rows, columns=["chrom", "start", "end", "transcript_id"]),
        "utr": pd.DataFrame(
            utr_rows, columns=["chrom", "start", "end", "strand", "transcript_id"]),
    }


def _derive_promoters(transcripts: pd.DataFrame) -> pd.DataFrame:
    """TSS +/- PROMOTER_FLANK_BP per transcript (TSS = start on '+' strand,
    end on '-' strand)."""
    if transcripts.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    tss = np.where(transcripts["strand"].to_numpy() == "+",
                    transcripts["start"].to_numpy(dtype=np.int64),
                    transcripts["end"].to_numpy(dtype=np.int64))
    starts = np.clip(tss - PROMOTER_FLANK_BP, 0, None)
    ends = tss + PROMOTER_FLANK_BP
    return pd.DataFrame({
        "chrom": transcripts["chrom"].to_numpy(),
        "start": starts.astype(np.int64),
        "end": ends.astype(np.int64),
    })


def _derive_utr_split(
    cds: pd.DataFrame, utr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw UTR records into 5'/3' using each transcript's CDS extent
    (min CDS start / max CDS end) and strand, since GENCODE tags all UTRs
    simply 'UTR' without a 5'/3' distinction."""
    empty = pd.DataFrame(columns=["chrom", "start", "end"])
    if cds.empty or utr.empty:
        return empty, empty

    cds_extent = cds.groupby("transcript_id").agg(
        cds_min_start=("start", "min"), cds_max_end=("end", "max"))
    merged = utr.merge(cds_extent, on="transcript_id", how="inner")
    if merged.empty:
        return empty, empty

    is_plus = merged["strand"].to_numpy() == "+"
    starts = merged["start"].to_numpy(dtype=np.int64)
    ends = merged["end"].to_numpy(dtype=np.int64)
    cds_min = merged["cds_min_start"].to_numpy(dtype=np.int64)
    cds_max = merged["cds_max_end"].to_numpy(dtype=np.int64)

    is_five = (is_plus & (ends <= cds_min)) | (~is_plus & (starts >= cds_max))
    is_three = (is_plus & (starts >= cds_max)) | (~is_plus & (ends <= cds_min))

    n_dropped = int(len(merged) - is_five.sum() - is_three.sum())
    if n_dropped:
        print(f"[region_annotation] dropped {n_dropped:,} UTR records that didn't "
              "cleanly resolve to 5'/3' relative to their transcript's CDS")

    cols = ["chrom", "start", "end"]
    return merged.loc[is_five, cols].copy(), merged.loc[is_three, cols].copy()


def _derive_introns(exon: pd.DataFrame) -> pd.DataFrame:
    """Per-transcript gaps between consecutive sorted exons."""
    if exon.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    rows: list[dict] = []
    for _, sub in exon.sort_values(["transcript_id", "start"]).groupby("transcript_id", sort=False):
        if len(sub) < 2:
            continue
        chrom = sub["chrom"].iloc[0]
        starts = sub["start"].to_numpy(dtype=np.int64)
        ends = sub["end"].to_numpy(dtype=np.int64)
        for i in range(len(starts) - 1):
            gap_start, gap_end = int(ends[i]), int(starts[i + 1])
            if gap_end > gap_start:
                rows.append({"chrom": chrom, "start": gap_start, "end": gap_end})
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def _build_region_cache() -> None:
    print("[region_annotation] parsing GENCODE GTF for region categories (one-time) ...")
    records = _parse_gtf_records()
    five_raw, three_raw = _derive_utr_split(records["cds"], records["utr"])

    raw = {
        "gene_body": records["gene"][["chrom", "start", "end"]],
        "promoter": _derive_promoters(records["transcript"]),
        "five_utr": five_raw,
        "three_utr": three_raw,
        "intron": _derive_introns(records["exon"]),
    }
    regions = {name: merge_intervals(df) for name, df in raw.items()}
    for name, region_df in regions.items():
        print(f"  {name}: {len(region_df):,} merged intervals")
    pd.to_pickle(regions, REGION_CACHE, compression="gzip")


def load_gencode_regions() -> dict[str, pd.DataFrame]:
    """Build cache if missing, read pickle, rebuild-on-exception -- mirrors
    utils/desert_utils.py's load_features()."""
    if not os.path.exists(REGION_CACHE):
        print(f"[region_annotation] building region cache at {REGION_CACHE} (one-time) ...")
        _build_region_cache()
    try:
        regions = pd.read_pickle(REGION_CACHE, compression="gzip")
    except Exception as exc:
        print(
            "[region_annotation] region cache unreadable; rebuilding cache "
            f"at {REGION_CACHE} ({exc.__class__.__name__}: {exc})"
        )
        try:
            os.remove(REGION_CACHE)
        except OSError:
            pass
        _build_region_cache()
        regions = pd.read_pickle(REGION_CACHE, compression="gzip")
    return regions


def label_region_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Add boolean region-category columns to a copy of `frame`.

    `frame` must contain chrom/start/end. is_distal_intergenic requires NOT
    overlapping any gene body or promoter window, AND NOT falling inside one
    of the 633 curated gene deserts (reusing label_deserts_fleet verbatim,
    since desert boundaries aren't 1kb-grid-aligned and need real
    containment logic, not a fresh ad hoc overlap check).
    """
    required = {"chrom", "start", "end"}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame must contain columns {sorted(required)}")

    regions = load_gencode_regions()
    out = frame.copy()
    out["is_promoter"] = window_interval_overlap(out, regions["promoter"], return_fraction=False)
    out["is_5utr"] = window_interval_overlap(out, regions["five_utr"], return_fraction=False)
    out["is_3utr"] = window_interval_overlap(out, regions["three_utr"], return_fraction=False)
    out["is_intron"] = window_interval_overlap(out, regions["intron"], return_fraction=False)

    gene_or_promoter = merge_intervals(
        pd.concat([regions["gene_body"], regions["promoter"]], ignore_index=True))
    in_gene_or_promoter = window_interval_overlap(out, gene_or_promoter, return_fraction=False)

    all_deserts = load_all_deserts()
    desert_labeled = label_deserts_fleet(out[["chrom", "start", "end"]], all_deserts)
    in_desert = desert_labeled["desert"].notna()

    out["is_distal_intergenic"] = (~in_gene_or_promoter) & (~in_desert)
    return out


# ── Gnocchi-paper benchmark data: loaders + missing-file-tolerant manifest ──

BENCHMARK_DATA_DIR = os.path.join(DATA_DIR, "gnocchi_benchmark_data")

_COMPARISONS_FILES = {
    "gwas_catalog_repl": "comparisons_gwas_catalog_repl.txt",
    "gwas_fine-mapping_pip09": "comparisons_gwas_fine-mapping_pip09.txt",
    "gwas_fine-mapping_pip09_hc": "comparisons_gwas_fine-mapping_pip09_hc.txt",
    "clinvar_hgmd": "comparisons_likely_pathogenic_clinvar_hgmd.txt",
    "topmed_mac1": "comparisons_topmed_mac1.sampled.cov.txt",
    "topmed_maf001": "comparisons_topmed_maf001.sampled.cov.txt",
    "topmed_maf001_01": "comparisons_topmed_maf001_01.sampled.cov.txt",
    "topmed_maf01_1": "comparisons_topmed_maf01_1.sampled.cov.txt",
    "topmed_maf1_5": "comparisons_topmed_maf1_5.sampled.cov.txt",
    "topmed_maf5": "comparisons_topmed_maf5.sampled.cov.txt",
}
POSITIVE_SETS = [
    "gwas_catalog_repl", "gwas_fine-mapping_pip09",
    "gwas_fine-mapping_pip09_hc", "clinvar_hgmd",
]
NEGATIVE_SETS = [
    "topmed_mac1", "topmed_maf001", "topmed_maf001_01",
    "topmed_maf01_1", "topmed_maf1_5", "topmed_maf5",
]
COMPARATOR_SCORES = ["Orion", "CDTS", "gwRVIS", "DR", "phastCons", "phyloP", "GERP"]
# Human lineage-specific constraint metrics vs. interspecies conservation
# metrics -- the two groups the paper's dominance analysis contrasts.
HUMAN_CONSTRAINT_SCORES = ["z_adj", "z_unadj", "z_roulette", "Orion", "CDTS", "gwRVIS", "DR"]
CONSERVATION_SCORES = ["phastCons", "phyloP", "GERP"]

ANNOT_TABLE = os.path.join(BENCHMARK_DATA_DIR, "constraint_z_genome_1kb.annot.txt.gz")
ENH_GENE_ROADMAPLINKS_TABLE = os.path.join(BENCHMARK_DATA_DIR, "enh_gene_roadmaplinks.txt")
ENHZ_LOEUF_PRED_TABLE = os.path.join(BENCHMARK_DATA_DIR, "enhz_loeuf_pred.txt")

BENCHMARK_MANIFEST: dict[str, list[str]] = {
    "score_distributions": [ANNOT_TABLE],
    "oe_scatter": [ANNOT_TABLE],
    "prop_constrained_by_coding": [ANNOT_TABLE],
    "enrichment_regulatory_elements": [ANNOT_TABLE],
    "enrichment_gwas": [ANNOT_TABLE],
    "enrichment_gwas_vs_ccre": [ANNOT_TABLE],
    "prop_roadmaplinks": [ANNOT_TABLE],
    "roc_auc": [
        os.path.join(BENCHMARK_DATA_DIR, f) for f in _COMPARISONS_FILES.values()
    ],
    "auc_vs_af": [
        os.path.join(BENCHMARK_DATA_DIR, _COMPARISONS_FILES[n])
        for n in ["gwas_fine-mapping_pip09"] + NEGATIVE_SETS
    ],
    "gnocchi_self_comparison": [
        os.path.join(BENCHMARK_DATA_DIR, f) for f in _COMPARISONS_FILES.values()
    ],
    "dominance_analysis": [
        os.path.join(BENCHMARK_DATA_DIR, _COMPARISONS_FILES[n])
        for n in ["gwas_catalog_repl", "clinvar_hgmd"] + NEGATIVE_SETS
    ],
    "enhancer_geneset": [ANNOT_TABLE, ENH_GENE_ROADMAPLINKS_TABLE],
    "enhancer_loeuf_roc": [ANNOT_TABLE, ENH_GENE_ROADMAPLINKS_TABLE, ENHZ_LOEUF_PRED_TABLE],
}


def check_benchmark_data(name: str) -> list[str]:
    """Return the missing files required for benchmark `name` (empty = all present)."""
    return [p for p in BENCHMARK_MANIFEST.get(name, []) if not os.path.exists(p)]


def load_comparisons_table(name: str) -> pd.DataFrame | None:
    """Load one comparisons_*.txt positive/negative variant-score table by
    short name (see _COMPARISONS_FILES); None + printed notice if missing."""
    path = os.path.join(BENCHMARK_DATA_DIR, _COMPARISONS_FILES[name])
    if not os.path.exists(path):
        print(f"[benchmark_utils] skipping '{name}': missing {path}")
        return None
    return pd.read_csv(path, sep="\t")


def load_annot_table() -> pd.DataFrame | None:
    """Load the genome-wide 1kb-window annotated constraint track."""
    if not os.path.exists(ANNOT_TABLE):
        print(f"[benchmark_utils] skipping annot table: missing {ANNOT_TABLE}")
        return None
    return pd.read_csv(ANNOT_TABLE, sep="\t", compression="gzip")


def load_enh_gene_roadmaplinks() -> pd.DataFrame | None:
    """Load enhancer->gene RoadmapLinks with Gnocchi enhancer_constraint_Z
    and gene essentiality/LOEUF-category flags."""
    if not os.path.exists(ENH_GENE_ROADMAPLINKS_TABLE):
        print(f"[benchmark_utils] skipping enh_gene_roadmaplinks: missing {ENH_GENE_ROADMAPLINKS_TABLE}")
        return None
    return pd.read_csv(ENH_GENE_ROADMAPLINKS_TABLE, sep="\t")


def load_enhz_loeuf_pred() -> pd.DataFrame | None:
    """Load the gene-level enhancer-Z / LOEUF train-test prediction table."""
    if not os.path.exists(ENHZ_LOEUF_PRED_TABLE):
        print(f"[benchmark_utils] skipping enhz_loeuf_pred: missing {ENHZ_LOEUF_PRED_TABLE}")
        return None
    return pd.read_csv(ENHZ_LOEUF_PRED_TABLE, sep="\t")


# ── Locus / element_id <-> 1kb-window helpers ───────────────────────────────

def parse_element_id(df: pd.DataFrame, element_id_col: str = "element_id") -> pd.DataFrame:
    """Parse a packed 'chrom-start-end' column (Gnocchi element_id, or the
    similarly-packed 'enhancer' column in enh_gene_roadmaplinks.txt) into
    chrom/start/end columns."""
    out = df.copy()
    parts = out[element_id_col].str.rsplit("-", n=2, expand=True)
    out["chrom"] = parts[0]
    out["start"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    out["end"] = pd.to_numeric(parts[2], errors="coerce").astype("Int64")
    return out


def attach_window_id(df: pd.DataFrame, locus_col: str = "locus") -> pd.DataFrame:
    """Parse a 'chr:pos' locus column (comparisons_*.txt) into the
    containing 1kb window's chrom/start/end/element_id, for joining
    window-level scores (e.g. z_roulette) onto per-variant tables."""
    out = df.copy()
    parts = out[locus_col].str.split(":", n=1, expand=True)
    chrom = parts[0]
    pos = pd.to_numeric(parts[1], errors="coerce")
    start = ((pos // 1000) * 1000).astype("Int64")
    out["chrom"] = chrom
    out["start"] = start
    out["end"] = out["start"] + 1000
    out["element_id"] = (
        chrom + "-" + out["start"].astype(str) + "-" + out["end"].astype(str)
    )
    return out


def build_roulette_windows() -> pd.DataFrame:
    """Genome-wide window table (chrom/start/end/element_id/observed/
    expected/expected_unadj/z_adj/z_unadj/exp_roulette/z_roulette/...),
    computed once and reused for every benchmark that needs to join
    z_roulette onto element_id- or locus-keyed tables, or aggregate window
    counts up to arbitrary regions (e.g. enhancers)."""
    return load_roulette_z_table(ROULETTE_PATH_NONCODING, HAPLOID_FACTOR)


_ROULETTE_JOIN_COLS = ["element_id", "z_roulette", "z_unadj", "exp_roulette", "oe_roulette"]


def attach_roulette_score(
    df: pd.DataFrame, roulette_windows: pd.DataFrame, locus_col: str = "locus",
) -> pd.DataFrame:
    """Join z_roulette (+ z_unadj/exp_roulette/oe_roulette) onto a
    locus-keyed (chr:pos) table, renaming its native 'z' column (Gnocchi
    adjusted) to 'z_adj' for consistency with the rest of this codebase."""
    out = attach_window_id(df, locus_col=locus_col)
    if "z" in out.columns:
        out = out.rename(columns={"z": "z_adj"})
    cols = [c for c in _ROULETTE_JOIN_COLS if c in roulette_windows.columns]
    return out.merge(roulette_windows[cols], on="element_id", how="left")


def attach_roulette_to_annot(
    annot_df: pd.DataFrame, roulette_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Join z_roulette (+ z_unadj/exp_roulette/oe_roulette) onto the
    element_id-keyed annot table, and rename its Gnocchi 'z' column to
    'z_adj' for consistency with the rest of this codebase's naming."""
    out = annot_df.rename(columns={"z": "z_adj"})
    cols = [c for c in _ROULETTE_JOIN_COLS if c in roulette_windows.columns]
    return out.merge(roulette_windows[cols], on="element_id", how="left")


# ── Z-bin / enrichment statistics (ported from fig_utils.py's style) ───────

Z_BIN_EDGES = [-10, -4, -3, -2, -1, 0, 1, 2, 3, 4, 10]
Z_BIN_LABELS = ["<-4", "-4..-3", "-3..-2", "-2..-1", "-1..0",
                "0..1", "1..2", "2..3", "3..4", ">4"]


def sem(x: float, n: float) -> float:
    """Standard error of a proportion x/n."""
    if n <= 0:
        return float("nan")
    p = x / n
    return float(np.sqrt(p * (1 - p) / n))


def ci_of_odds(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float, float]:
    """Odds ratio of (x1/n1) vs (x2/n2) + 95% CI, via Haldane-Anscombe
    correction (+0.5 to every cell) and the log-odds normal approximation."""
    a, b, c, d = x1 + 0.5, (n1 - x1) + 0.5, x2 + 0.5, (n2 - x2) + 0.5
    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = np.exp(log_or - 1.96 * se_log_or)
    hi = np.exp(log_or + 1.96 * se_log_or)
    return float(odds_ratio), float(lo), float(hi)


def coerce_bool(s: pd.Series) -> pd.Series:
    """The annot table stores some flag columns as 'True'/'False' strings
    and others as 0.0/1.0 floats -- normalize either to a bool Series."""
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(bool)
    return (
        s.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def filter_noncoding_qc(annot: pd.DataFrame) -> pd.DataFrame:
    """Restrict to non-coding, QC-passing windows, matching the filter
    fig_utils.py/efig_utils.py apply before every enrichment-by-Z-bin figure
    (plt_enrichment_re/plt_enrichment_gwas/plt_enrichment_gwas_vs_ccre/
    plt_prop_roadmaplinks all start from `df_z[(df_z['pass_qc']) &
    (df_z['coding_prop']==0)]`) -- without it, ~7% of windows here overlap
    coding sequence and skew the enrichment estimates relative to the paper."""
    mask = annot["coding_prop"] == 0
    if "pass_qc" in annot.columns:
        mask &= coerce_bool(annot["pass_qc"])
    return annot[mask]


def enrichment_by_zbin(
    df: pd.DataFrame, z_col: str, annot_col: str, bin_edges: list[float] = Z_BIN_EDGES,
) -> pd.DataFrame:
    """Per Z-bin: odds ratio (+95% CI) of carrying `annot_col` vs. the rest
    of the genome, matching fig_utils.py's plt_enrichment_re/plt_enrichment_gwas
    scheme (Fisher-style binned enrichment, not a smooth regression)."""
    sub = df[[z_col, annot_col]].dropna(subset=[z_col]).copy()
    sub[annot_col] = coerce_bool(sub[annot_col])
    sub["_bin"] = pd.cut(sub[z_col], bins=bin_edges, labels=False, include_lowest=True)

    n_total = len(sub)
    x_total = int(sub[annot_col].sum())

    rows = []
    for bin_idx in range(len(bin_edges) - 1):
        bsub = sub[sub["_bin"] == bin_idx]
        n_bin = len(bsub)
        if n_bin == 0:
            rows.append({"bin_idx": bin_idx, "n": 0, "n_annot": 0, "frac_annot": np.nan,
                         "odds_ratio": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        x_bin = int(bsub[annot_col].sum())
        x_rest, n_rest = x_total - x_bin, n_total - n_bin
        odds_ratio, lo, hi = ci_of_odds(x_bin, n_bin, x_rest, n_rest)
        rows.append({"bin_idx": bin_idx, "n": n_bin, "n_annot": x_bin,
                     "frac_annot": x_bin / n_bin,
                     "odds_ratio": odds_ratio, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ── ROC/AUC helper (ported from fig_utils.py's plt_comparison_roc scheme) ──

def roc_auc_by_score(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    score_cols: list[str],
    n_boot: int = 10,
    seed: int = 714,
) -> dict[str, dict[str, object]]:
    """Per score column: mean ROC/AUC classifying pos_df vs. a
    10x-subsampled draw from neg_df, averaged over n_boot subsamples
    (mirrors plt_comparison_roc's bootstrap-average scheme; uses all of
    neg_df when it's smaller than 10x the positive set).

    Note on direction: sklearn's roc_auc_score assumes higher score =>
    more likely positive. Gnocchi/conservation scores follow that
    convention here (higher z / higher phyloP/phastCons/GERP = more
    constrained), but if a comparator's AUC comes out well below 0.5, its
    native convention is likely inverted -- that's diagnostic information,
    not silently corrected.
    """
    rng = np.random.default_rng(seed)
    n_pos = len(pos_df)
    neg_size = min(len(neg_df), 10 * n_pos)

    results: dict[str, dict[str, object]] = {}
    for col in score_cols:
        if col not in pos_df.columns or col not in neg_df.columns:
            continue
        aucs = []
        fpr_last, tpr_last = None, None
        for _ in range(n_boot):
            neg_sample = neg_df.sample(n=neg_size, random_state=int(rng.integers(0, 2**32 - 1)))
            y = np.concatenate([np.ones(n_pos), np.zeros(len(neg_sample))])
            scores = pd.concat([pos_df[col], neg_sample[col]], ignore_index=True).to_numpy(dtype=float)
            mask = np.isfinite(scores)
            if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
                continue
            aucs.append(roc_auc_score(y[mask], scores[mask]))
            fpr_last, tpr_last, _ = roc_curve(y[mask], scores[mask])
        if not aucs:
            continue
        results[col] = {
            "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "fpr": fpr_last, "tpr": tpr_last,
        }
    return results


# ── Shared plotting style (matches fig_utils.py/efig_utils.py's visual ─────
# scheme; output stays PNG@150dpi into results/, this repo's own convention) ─

GNOCCHI_COLORS = {
    "z_adj": "#4c4173",       # paper's primary "Gnocchi" cubehelix accent
    "z_unadj": "#81b4bf",     # paper's secondary/"non-coding" cubehelix accent
    "z_roulette": "#c0392b",  # new: distinct warm red for the Roulette comparator
}
COMPARATOR_COLORS = {
    "Orion": "#33a02c", "CDTS": "#fb9a99", "DR": "#993404", "gwRVIS": "#6baed6",
    "phastCons": "#969696", "phyloP": "#737373", "GERP": "#bdbdbd",
}
REFERENCE_LINE_COLOR = "#969696"

# Per-annotation colors for the Fig. 2a/2b-style overlay layout, using the
# exact same sns.cubehelix_palette() calls + indices as fig_utils.py's
# plt_enrichment_re/plt_enrichment_gwas (ann_color dicts).
_CMAP_DEFAULT = sns.cubehelix_palette()
_CMAP_ROTATED = sns.cubehelix_palette(start=0.5, rot=-0.5)
REGULATORY_ANN_COLORS = {
    "ENCODE cCRE-PLS": _CMAP_DEFAULT[-2],
    "ENCODE cCRE-pELS": _CMAP_DEFAULT[-3],
    "ENCODE cCRE-dELS": _CMAP_DEFAULT[-4],
    "ENCODE CTCF-only": "#969696",
    "Super enhancers": _CMAP_ROTATED[-2],
    "FANTOM enhancers": _CMAP_ROTATED[-4],
}
GWAS_ANN_COLORS = {
    "GWAS Catalog": _CMAP_ROTATED[-5],
    "GWAS Catalog repl (ext)": _CMAP_ROTATED[-4],
    "GWAS fine-mapping": _CMAP_ROTATED[-2],
}


def style_axes(ax) -> None:
    """The one near-universal styling call in fig_utils.py/efig_utils.py."""
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(axis="both", top=False, right=False)


def score_color(name: str, fallback_index: int = 0) -> str:
    """Color for any score column: Gnocchi-family and paper-comparator
    colors are fixed; anything else cycles through tab10."""
    if name in GNOCCHI_COLORS:
        return GNOCCHI_COLORS[name]
    if name in COMPARATOR_COLORS:
        return COMPARATOR_COLORS[name]
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab10").colors[fallback_index % 10]


# ── Enhancer-level rollup (sum obs/exp over overlapping windows, then one ──
# chi-sq z on the totals -- the same convention the pipeline uses for every
# other multi-window aggregation; see run_nc_constraint_gnomad_v31_main.py's
# per-element_id groupby-sum before its chi-sq step) ────────────────────────

def compute_enhancer_z(
    enhancer_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    obs_col: str = "observed",
    exp_col: str = "expected",
) -> pd.Series:
    """Roll up 1kb-window obs/exp counts to enhancer-level Z: sum obs_col
    and exp_col across every window overlapping each enhancer, then apply
    the chi-sq deviation once on the totals."""
    agg = aggregate_windows_over_regions(enhancer_df, windows_df, [obs_col, exp_col])
    return pd.Series(
        compute_gnocchi_z(agg[obs_col].to_numpy(), agg[exp_col].to_numpy()),
        index=enhancer_df.index,
    )


def validate_enhancer_rollup(
    enh_gene_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    known_z_col: str = "enhancer_constraint_Z",
) -> float:
    """Recompute Gnocchi enhancer Z from raw windows via compute_enhancer_z
    and return its Pearson correlation with the file's existing (known)
    Gnocchi enhancer_constraint_Z -- a sanity check before trusting the
    same rollup rule for a Roulette version."""
    enh = parse_element_id(enh_gene_df, "enhancer")
    recomputed = compute_enhancer_z(enh, windows_df).to_numpy()
    known = enh_gene_df[known_z_col].to_numpy(dtype=float)
    mask = np.isfinite(recomputed) & np.isfinite(known)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(recomputed[mask], known[mask])[0, 1])


def validate_gene_rollup_rule(
    enh_gene_df: pd.DataFrame,
    enhz_pred_df: pd.DataFrame,
    enh_z_col: str = "enhancer_constraint_Z",
) -> dict[str, float]:
    """Empirically determine how enhz_loeuf_pred.txt's gene-level Z rolls
    up from enh_gene_roadmaplinks.txt's per-enhancer Z (mean/max/min over a
    gene's linked enhancers), by correlating each candidate rule against
    the known gene-level values -- rather than guessing -- so the same,
    now-confirmed rule can be applied to a Roulette gene-level rollup."""
    per_gene = enh_gene_df.groupby("gene")[enh_z_col].agg(["mean", "max", "min"])
    merged = enhz_pred_df[["gene", enh_z_col]].merge(
        per_gene, on="gene", how="inner", suffixes=("", "_agg"))
    results = {}
    for rule in ["mean", "max", "min"]:
        known = merged[enh_z_col].to_numpy(dtype=float)
        candidate = merged[rule].to_numpy(dtype=float)
        mask = np.isfinite(known) & np.isfinite(candidate)
        results[rule] = float(np.corrcoef(known[mask], candidate[mask])[0, 1]) if mask.sum() >= 3 else float("nan")
    return results

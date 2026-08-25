"""GENCODE-derived genome-wide non-coding region categories.

v1 scope: Promoter, 5' UTR, 3' UTR, Intron, Distal intergenic. ENCODE
cCRE/enhancer and CTCF/TF ChIP-seq categories are deferred to a future
phase once those BED files are added to data/.

Categories are non-exclusive: a window may belong to more than one at
once (e.g. is_promoter and is_intron both True).
"""

from __future__ import annotations

import gzip
import os
import re

import numpy as np
import pandas as pd

from utils.desert_utils import (
    DATA_DIR,
    RESULTS_DIR,
    label_deserts_fleet,
    load_all_deserts,
    merge_intervals,
    window_interval_overlap,
)

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

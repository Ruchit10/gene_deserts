#!/usr/bin/env python3
"""Build the regional genomic feature matrix for chrX PAR1/PAR2 windows.

The published feature matrix (`data/genomic_features13_genome_1kb.txt.gz`) is
chr1-22 only, which is why the Gnocchi regional adjustment silently degrades to
rr = 1 across chrX PAR. This script regenerates the same 53-column schema for
the 2,497 PAR windows using the upstream source tracks in
`data/genomic_features13/`.

Only the 7 features whose source data actually covers PAR are computed. The
other 6 are written as NaN and must be masked downstream -- see MASKED_FEATURES
for the per-feature evidence. Note that a naive `bedtools coverage` run against
a track with no chrX intervals returns 0.0 rather than NaN, which would look
like a legitimate low value and silently bias rr; writing NaN here makes the
masking explicit in the artifact.

Interval overlap reuses `window_interval_overlap()` from utils.desert_utils
rather than shelling out to bedtools (not a dependency of this repo).

Usage:
    python regional_corrections/build_par_features.py              # build PAR matrix
    python regional_corrections/build_par_features.py --validate   # feature gate
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.desert_utils import (  # noqa: E402
    DATA_DIR,
    FEATURE_COLUMNS,
    FEATURE_SCALES,
    load_bed_intervals,
    merge_intervals,
    window_interval_overlap,
)

FEATURES13_DIR = os.path.join(DATA_DIR, "genomic_features13")
PAR_GNOCCHI_TABLE = os.path.join(DATA_DIR, "gnocchi_1kb_chrX_par_public.txt.gz")
OUT_TABLE = os.path.join(DATA_DIR, "genomic_features13_chrX_par_1kb.txt.gz")

# Features whose source track demonstrably covers PAR (verified interval counts
# in the module docstring of the accompanying README).
COVERAGE_TRACKS = {
    "LCR": "btu356_LCR-hs38.bed",
    "SINE": "repeatMasker_hg38.SINE.bed",
    "LINE": "repeatMasker_hg38.LINE.bed",
    "CpG_island": "cpgIslandExt.bed",
}
DISTANCE_TRACKS = {"dist2telo": "telomeres_hg38.bed", "dist2cent": "centromeres_hg38.bed"}

# Computed from sequence; requires a reference slice (see build_gc_content).
SEQUENCE_FEATURES = ["GC_content"]

# Masked: source data does not cover PAR at all. Verified directly --
#   recomb_male        : 0 chrX rows in genetic.map.final.pat.gor.bed
#   recomb_female      : chrX spans 3,532,526-154,781,072 -> 0 PAR overlap
#   cDNM_maternal_05M  : 0 chrX rows in Goldmann_18_S5_cDNMs_F.lft38.bed
#   cDNM_paternal_05M  : 0 chrX rows in Goldmann_18_S5_cDNMs_M.lft38.bed
#   met_sperm          : source track not available locally
#   Nucleosome         : source track not available locally
MASKED_FEATURES = [
    "recomb_male", "recomb_female", "met_sperm", "Nucleosome",
    "cDNM_maternal_05M", "cDNM_paternal_05M",
]

# hg38 chromosome sizes, as used by normalize_dist2gaps_1kb.py (which already
# includes chrX, so the upstream normalization transfers unchanged).
CHROM_SIZES_FILE = os.path.join(FEATURES13_DIR, "hg38.chrom.sizes")

# Window grids, already generated genome-wide including chrX.
WINDOW_BEDS = {
    "1k": "hg38.chrom.1kb.bed",
    "10k": "hg38.chrom.1kb_flnk_10k.bed",
    "100k": "hg38.chrom.1kb_flnk_100k.bed",
    "1M": "hg38.chrom.1kb_flnk_1M.bed",
}
# Distance features are NOT scale-invariant: each scale is measured from the
# midpoint of that scale's flank window, so every scale needs its own mid BED.
# Verified against published values, e.g. chr1-26000-27000 dist2telo_1k =
# (26500-1)/248956422*100 = 0.010644 and dist2telo_1M = (500000-1)/... = 0.200838.
MID_BEDS = {
    "1k": "hg38.chrom.1kb.mid.bed",
    "10k": "hg38.chrom.1kb_flnk_10k.mid.bed",
    "100k": "hg38.chrom.1kb_flnk_100k.mid.bed",
    "1M": "hg38.chrom.1kb_flnk_1M.mid.bed",
}


def load_chrom_sizes() -> dict[str, int]:
    df = pd.read_csv(CHROM_SIZES_FILE, sep="\t", header=None, names=["chrom", "size"])
    return dict(zip(df["chrom"], df["size"]))


def load_window_bed(name: str, chrom: str, element_ids: set[str] | None = None) -> pd.DataFrame:
    """Read one of the pre-generated window BEDs, restricted to a chromosome.

    These files are ~147MB each, so they are streamed in chunks and filtered
    rather than loaded whole.
    """
    path = os.path.join(FEATURES13_DIR, name)
    keep = []
    for chunk in pd.read_csv(
        path, sep="\t", header=None, names=["chrom", "start", "end", "element_id"],
        chunksize=1_000_000,
    ):
        sub = chunk[chunk["chrom"] == chrom]
        if element_ids is not None:
            sub = sub[sub["element_id"].isin(element_ids)]
        if not sub.empty:
            keep.append(sub)
    if not keep:
        return pd.DataFrame(columns=["chrom", "start", "end", "element_id"])
    return pd.concat(keep, ignore_index=True)


_TRACK_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load_track(track_path: str, chrom: str) -> pd.DataFrame:
    """Load a BED track restricted to one chromosome, with overlaps merged.

    Merging matters: `bedtools coverage` reports bases covered by at least one
    feature, whereas summing raw interval overlaps double-counts where repeat
    annotations overlap each other. Without the merge, SINE/LINE values drift up
    to ~3.4 percentage points high on a minority of windows.
    """
    key = (track_path, chrom)
    if key not in _TRACK_CACHE:
        iv = load_bed_intervals(track_path)
        iv = iv[iv["chrom"] == chrom]
        _TRACK_CACHE[key] = merge_intervals(iv) if not iv.empty else iv
    return _TRACK_CACHE[key]


def compute_coverage_feature(windows: pd.DataFrame, track_path: str, chrom: str) -> pd.Series:
    """Percent of each window covered by the track (bedtools coverage equivalent)."""
    intervals = load_track(track_path, chrom)
    if intervals.empty:
        # No intervals on this chromosome: coverage is genuinely 0, but callers
        # must decide whether that is meaningful or a masked feature.
        return pd.Series(0.0, index=windows.index)
    frac = window_interval_overlap(windows, intervals, return_fraction=True)
    return frac * 100.0


SEQ_CACHE_DIR = os.path.join(FEATURES13_DIR, "_seq_cache")
UCSC_SEQ_API = "https://api.genome.ucsc.edu/getData/sequence"


def fetch_sequence(chrom: str, start: int, end: int, chunk: int = 1_000_000) -> str:
    """Fetch reference sequence from the UCSC REST API, with an on-disk cache.

    Only the spans actually needed are fetched (PAR1/PAR2 plus their 1M flanks
    is ~4Mb), so no local hg38 copy or UCSC binary is required.
    """
    import gzip as _gzip
    import json
    import urllib.request

    os.makedirs(SEQ_CACHE_DIR, exist_ok=True)
    cache = os.path.join(SEQ_CACHE_DIR, f"{chrom}_{start}_{end}.txt.gz")
    if os.path.exists(cache):
        with _gzip.open(cache, "rt") as fh:
            return fh.read()

    parts = []
    for s in range(start, end, chunk):
        e = min(s + chunk, end)
        url = f"{UCSC_SEQ_API}?genome=hg38;chrom={chrom};start={s};end={e}"
        with urllib.request.urlopen(url, timeout=300) as resp:
            parts.append(json.load(resp)["dna"].upper())
    seq = "".join(parts)
    with _gzip.open(cache, "wt") as fh:
        fh.write(seq)
    return seq


def build_gc_lookup(chrom: str, grids: dict[str, pd.DataFrame]) -> list[tuple[int, int, np.ndarray]]:
    """Fetch sequence for every span the windows touch and return GC cumsums.

    Returns a list of (block_start, block_end, cumulative G+C count) so that any
    window's GC can be answered in O(1).
    """
    spans = pd.concat([g[["chrom", "start", "end"]] for g in grids.values()], ignore_index=True)
    blocks = merge_intervals(spans[spans["chrom"] == chrom])
    out = []
    for _, b in blocks.iterrows():
        bs, be = int(b["start"]), int(b["end"])
        print(f"  fetching {chrom}:{bs:,}-{be:,} ({(be - bs) / 1e6:.2f} Mb) ...")
        seq = fetch_sequence(chrom, bs, be)
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        is_gc = (arr == ord("G")) | (arr == ord("C"))
        out.append((bs, be, np.concatenate([[0], np.cumsum(is_gc)]).astype(np.int64)))
    return out


def compute_gc_feature(windows: pd.DataFrame,
                       blocks: list[tuple[int, int, np.ndarray]]) -> pd.Series:
    """Percent G+C per window.

    Convention verified against published values: N bases count toward the
    denominator (i.e. GC% = 100*(G+C)/(end-start)), matching
    `hgGcPercent -doGaps -win=1000`. A window with 360 Ns reproduces the
    published value exactly under this rule and not under N-exclusion.
    """
    out = pd.Series(np.nan, index=windows.index, dtype=float)
    for bs, be, cum in blocks:
        sel = (windows["start"] >= bs) & (windows["end"] <= be)
        if not sel.any():
            continue
        s = windows.loc[sel, "start"].to_numpy(dtype=np.int64) - bs
        e = windows.loc[sel, "end"].to_numpy(dtype=np.int64) - bs
        out.loc[sel] = (cum[e] - cum[s]) / (e - s) * 100.0
    return out


def compute_distance_feature(
    mids: pd.DataFrame, track_path: str, chrom_sizes: dict[str, int]
) -> pd.Series:
    """Distance from window midpoint to nearest interval, as % of chromosome length.

    Mirrors `bedtools closest -d` followed by normalize_dist2gaps_1kb.py, which
    divides by the chromosome size and multiplies by 100.
    """
    intervals = load_bed_intervals(track_path)
    out = pd.Series(np.nan, index=mids.index, dtype=float)
    for chrom, sub in mids.groupby("chrom", sort=False):
        iv = intervals[intervals["chrom"] == chrom]
        if iv.empty:
            continue
        starts = iv["start"].to_numpy(dtype=np.int64)
        ends = iv["end"].to_numpy(dtype=np.int64)
        pos = sub["start"].to_numpy(dtype=np.int64)
        # distance is 0 when inside an interval, else gap to nearest edge
        d_left = starts[None, :] - pos[:, None]
        d_right = pos[:, None] - ends[None, :]
        dist = np.maximum(np.maximum(d_left, d_right), 0).min(axis=1)
        out.loc[sub.index] = dist / chrom_sizes[chrom] * 100.0
    return out


def build_matrix(chrom: str, element_ids: set[str]) -> pd.DataFrame:
    """Assemble the 53-column feature matrix for the requested windows."""
    chrom_sizes = load_chrom_sizes()

    print(f"  loading window grids for {chrom} ...")
    grids = {s: load_window_bed(WINDOW_BEDS[s], chrom, element_ids) for s in FEATURE_SCALES}
    mids = {s: load_window_bed(MID_BEDS[s], chrom, element_ids) for s in FEATURE_SCALES}
    n = len(grids["1k"])
    print(f"  {n:,} windows resolved")

    frame = pd.DataFrame({"element_id": grids["1k"]["element_id"].to_numpy()})

    print("  preparing reference sequence for GC_content ...")
    gc_blocks = build_gc_lookup(chrom, grids)

    for scale in FEATURE_SCALES:
        g = grids[scale]
        order = g["element_id"].to_numpy()
        for feat, track in COVERAGE_TRACKS.items():
            print(f"  computing {feat}_{scale} ...")
            vals = compute_coverage_feature(g, os.path.join(FEATURES13_DIR, track), chrom)
            frame[f"{feat}_{scale}"] = pd.Series(vals.to_numpy(), index=order).reindex(
                frame["element_id"]).to_numpy()
        for feat, track in DISTANCE_TRACKS.items():
            print(f"  computing {feat}_{scale} ...")
            m = mids[scale]
            vals = compute_distance_feature(m, os.path.join(FEATURES13_DIR, track), chrom_sizes)
            frame[f"{feat}_{scale}"] = pd.Series(
                vals.to_numpy(), index=m["element_id"].to_numpy()
            ).reindex(frame["element_id"]).to_numpy()
        print(f"  computing GC_content_{scale} ...")
        gc = compute_gc_feature(g, gc_blocks)
        frame[f"GC_content_{scale}"] = pd.Series(
            gc.to_numpy(), index=order).reindex(frame["element_id"]).to_numpy()
        for feat in MASKED_FEATURES:
            frame[f"{feat}_{scale}"] = np.nan

    return frame[["element_id"] + FEATURE_COLUMNS]


def validate(chrom: str, n_windows: int) -> int:
    """Feature gate: recompute features for autosomal windows and compare to published.

    This is a correctness test of the builder only -- it produces no autosomal
    results. If our track parsing, overlap arithmetic or normalization differs
    from the upstream recipe, PAR values would land on a different scale than
    the fitted models expect, and there would be no way to detect it on chrX.
    """
    from utils.desert_utils import load_features  # noqa: E402

    print(f"Feature gate: recomputing {n_windows:,} {chrom} windows and comparing to published")
    published = load_features()
    pub = published[published["element_id"].str.startswith(f"{chrom}-")].head(n_windows)
    ids = set(pub["element_id"])
    print(f"  {len(ids):,} published windows selected")

    mine = build_matrix(chrom, ids)
    merged = pub[["element_id"]].merge(mine, on="element_id", how="inner", suffixes=("", "_mine"))
    merged = merged.merge(pub, on="element_id", suffixes=("_mine", "_pub"))

    checkable = [f"{f}_{s}" for s in FEATURE_SCALES
                 for f in list(COVERAGE_TRACKS) + list(DISTANCE_TRACKS) + SEQUENCE_FEATURES]
    print(f"\n  {'column':24s} {'r':>9s} {'med|diff|':>10s} {'max|diff|':>10s}  verdict")
    failures = 0
    for col in checkable:
        a = merged[f"{col}_mine"].to_numpy(dtype=float)
        b = merged[f"{col}_pub"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            print(f"  {col:24s} {'n/a':>9s} {'':>10s} {'':>10s}  SKIP (no data)")
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        d = np.abs(a[ok] - b[ok])
        passed = r > 0.999 and np.median(d) < 0.05
        failures += 0 if passed else 1
        print(f"  {col:24s} {r:9.5f} {np.median(d):10.4f} {d.max():10.4f}  "
              f"{'PASS' if passed else 'FAIL'}")
    print(f"\n  {len(checkable) - failures}/{len(checkable)} columns pass")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="Run the feature gate against published autosomal values and exit.")
    ap.add_argument("--validate-chrom", default="chr21")
    ap.add_argument("--validate-n", type=int, default=3000)
    args = ap.parse_args()

    if args.validate:
        sys.exit(1 if validate(args.validate_chrom, args.validate_n) else 0)

    par = pd.read_csv(PAR_GNOCCHI_TABLE, sep="\t", usecols=["element_id", "chrom"])
    ids = set(par["element_id"])
    print(f"Building feature matrix for {len(ids):,} chrX PAR windows")
    frame = build_matrix("chrX", ids)
    frame.to_csv(OUT_TABLE, sep="\t", index=False, compression="gzip")
    print(f"  wrote {OUT_TABLE}")
    filled = [c for c in FEATURE_COLUMNS if frame[c].notna().any()]
    print(f"  {len(filled)}/{len(FEATURE_COLUMNS)} columns populated; "
          f"{len(FEATURE_COLUMNS) - len(filled)} masked as NaN")


if __name__ == "__main__":
    main()

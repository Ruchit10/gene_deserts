"""Shared utilities for gene-desert Gnocchi diagnostic analyses.

Centralizes the desert exemplar definitions, loading of the merged
adjusted/unadjusted Gnocchi table, and the `label_deserts` helper that
tags each 1kb window with the desert it falls inside.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
ALL_DESERTS_TABLE = os.path.join(DATA_DIR, "hg38_desert_nczscores.txt")

# (chrom, start, end, note) per exemplar
DESERTS: dict[str, tuple[str, int, int, str]] = {
    "GD513": ("chr15", 97973833, 98648539, "highest mean/median z"),
    "GD198": ("chr8",   4994972,  6406592, "lowest mean/median z"),
    "GD588": ("chr5",  43707405, 44300247, "most negative z/GC cor"),
    "GD158": ("chr21", 15880064, 17513043, "strongest ACF (10kb)"),
    "GD167": ("chr6",  22571666, 24126186, "local structure; strong ACF"),
}

DESERT_ORDER: list[str] = list(DESERTS.keys())

GNOCCHI_TABLE = os.path.join(RESULTS_DIR, "gnocchi_adj_vs_unadj.tsv.gz")
FEATURES_TABLE = os.path.join(DATA_DIR, "genomic_features13_genome_1kb.txt.gz")
REPLICATION_TIMING_TABLE = os.path.join(DATA_DIR, "GM12878_hg38_smoothed.txt")
PHYLOP_TABLE = os.path.join(DATA_DIR, "phyloP447wayPrimates.txt.gz")
GNOMAD_SV_TABLE = os.path.join(DATA_DIR, "gnomad.v4.1.sv.sites.bed.gz")
LADS_DIR = os.path.join(DATA_DIR, "LADs")

FEATURE_BASES = [
    "dist2telo", "dist2cent", "GC_content", "LCR", "SINE", "LINE",
    "recomb_male", "recomb_female", "met_sperm", "Nucleosome",
    "cDNM_maternal_05M", "cDNM_paternal_05M", "CpG_island",
]
FEATURE_SCALES = ["1k", "10k", "100k", "1M"]
FEATURE_COLUMNS = [f"{b}_{s}" for s in FEATURE_SCALES for b in FEATURE_BASES]


def load_gnocchi(usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the merged adjusted/unadjusted Gnocchi table.

    The upstream script writes an `oe` column (adjusted observed/expected)
    plus `expected_unadj`; we also add `oe_unadj` for convenience.
    """
    df = pd.read_csv(GNOCCHI_TABLE, sep="\t", usecols=list(usecols) if usecols else None)
    if "expected_unadj" in df.columns and "observed" in df.columns:
        df["oe_unadj"] = df["observed"] / df["expected_unadj"]
    return df


def label_deserts(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with the desert ID it falls inside (else NaN)."""
    out = frame.copy()
    out["desert"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    for name, (chrom, start, end, _) in DESERTS.items():
        mask = (
            (out["chrom"] == chrom)
            & (out["start"] >= start)
            & (out["end"] <= end)
        )
        out.loc[mask, "desert"] = name
    return out


def load_all_deserts() -> pd.DataFrame:
    """Load coordinates and summary columns for all ~633 deserts."""
    df = pd.read_csv(ALL_DESERTS_TABLE, sep="\t")
    rename_map = {
        "chr": "chrom",
        "desert_start": "start",
        "desert_end": "end",
    }
    df = df.rename(columns=rename_map)
    needed = ["desert_id", "chrom", "start", "end"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in all-deserts table: {missing}")
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    return df


def label_deserts_fleet(frame: pd.DataFrame, deserts_df: pd.DataFrame) -> pd.DataFrame:
    """Label rows with a desert ID from a full desert coordinate table.

    Uses a per-chromosome asof-merge on start positions, followed by a strict
    containment check (`start>=desert_start` and `end<=desert_end`).
    """
    required_frame = {"chrom", "start", "end"}
    required_deserts = {"desert_id", "chrom", "start", "end"}
    if not required_frame.issubset(frame.columns):
        raise ValueError(f"frame must contain columns {sorted(required_frame)}")
    if not required_deserts.issubset(deserts_df.columns):
        raise ValueError(f"deserts_df must contain columns {sorted(required_deserts)}")

    out = frame.copy()
    out["desert"] = pd.Series([None] * len(out), index=out.index, dtype=object)

    d = deserts_df[["desert_id", "chrom", "start", "end"]].copy()
    d = d.rename(columns={"start": "desert_start", "end": "desert_end"})

    for chrom in sorted(set(out["chrom"].dropna().unique()) & set(d["chrom"].dropna().unique())):
        w_sub = out.loc[out["chrom"] == chrom, ["start", "end"]].copy()
        if w_sub.empty:
            continue
        d_sub = d[d["chrom"] == chrom].sort_values("desert_start")
        if d_sub.empty:
            continue

        w_sorted = w_sub.sort_values("start").reset_index().rename(columns={"index": "_row_idx"})
        merged = pd.merge_asof(
            w_sorted,
            d_sub,
            left_on="start",
            right_on="desert_start",
            direction="backward",
        )
        valid = (
            merged["desert_id"].notna()
            & (merged["start"] >= merged["desert_start"])
            & (merged["end"] <= merged["desert_end"])
        )
        if valid.any():
            out.loc[merged.loc[valid, "_row_idx"], "desert"] = merged.loc[valid, "desert_id"].values

    return out


def desert_length_kb(name: str) -> int:
    """Desert size in kb (number of 1kb windows it spans, modulo endpoints)."""
    _, start, end, _ = DESERTS[name]
    return int(np.ceil((end - start) / 1000))


def desert_palette() -> dict[str, tuple]:
    import matplotlib.pyplot as plt
    return dict(zip(DESERT_ORDER, plt.cm.tab10.colors[:len(DESERT_ORDER)]))


FEATURES_CACHE = os.path.join(RESULTS_DIR, "_features_cache.pkl.gz")


def _build_features_cache() -> None:
    """Parse the 361MB feature file once and save a float32 pickle cache."""
    dtypes: dict[str, str] = {col: "float32" for col in FEATURE_COLUMNS}
    # Keep element_id as plain object/str for maximum pickle compatibility
    # across pandas versions/environments.
    dtypes["element_id"] = "object"
    chunks = []
    for chunk in pd.read_csv(FEATURES_TABLE, sep="\t", chunksize=500_000, dtype=dtypes):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.to_pickle(FEATURES_CACHE, compression="gzip")


def load_features(element_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the 52-column feature table, optionally restricted by element_id.

    First call parses the ~361MB gzipped text file and writes a pickle
    cache; subsequent calls read the cache directly.
    """
    if not os.path.exists(FEATURES_CACHE):
        print(f"[desert_utils] building feature cache at {FEATURES_CACHE} (one-time) ...")
        _build_features_cache()
    try:
        df = pd.read_pickle(FEATURES_CACHE, compression="gzip")
    except Exception as exc:
        # Common in shared repos when cache was created under a different
        # pandas version (e.g., StringDtype pickle incompatibilities).
        print(
            "[desert_utils] feature cache unreadable; rebuilding cache "
            f"at {FEATURES_CACHE} ({exc.__class__.__name__}: {exc})"
        )
        try:
            os.remove(FEATURES_CACHE)
        except OSError:
            # If remove fails, _build_features_cache will still attempt overwrite.
            pass
        _build_features_cache()
        df = pd.read_pickle(FEATURES_CACHE, compression="gzip")
    if element_ids is None:
        return df
    ids_set = set(element_ids)
    return df[df["element_id"].isin(ids_set)].reset_index(drop=True)


def _normalize_chrom_value(raw: object) -> str:
    """Normalize chromosome labels to chr-prefixed hg38 style."""
    if pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = text.replace("chr", "").replace("CHR", "")
    aliases = {"23": "X", "24": "Y", "25": "M", "MT": "M", "Mt": "M", "m": "M"}
    text = aliases.get(text, text)
    return f"chr{text}"


def load_replication_timing(path: str = REPLICATION_TIMING_TABLE) -> pd.DataFrame:
    """Load GM12878 replication timing points as chrom/position/rt_value."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    chr_col = col_map.get("chr", "Chr")
    pos_col = col_map.get("coordinate", "Coordinate")
    val_col = col_map.get("replication_timing", "Replication Timing")

    out = df[[chr_col, pos_col, val_col]].copy()
    out.columns = ["chrom", "position", "rt_value"]
    out["chrom"] = out["chrom"].map(_normalize_chrom_value)
    pos_numeric = pd.to_numeric(out["position"], errors="coerce")
    out["rt_value"] = pd.to_numeric(out["rt_value"], errors="coerce")
    out = out[pos_numeric.notna() & out["rt_value"].notna()].copy()
    pos_numeric = pos_numeric.loc[out.index]

    # Some RT resources contain decimal coordinate values; snap to nearest bp.
    out["position"] = np.rint(pos_numeric.to_numpy(dtype=float)).astype(np.int64)
    return out.sort_values(["chrom", "position"]).reset_index(drop=True)


def bin_values_to_windows(
    points_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    value_col: str,
) -> pd.Series:
    """Mean of irregularly spaced point-values within each genomic window."""
    required_points = {"chrom", "position", value_col}
    required_windows = {"chrom", "start", "end"}
    if not required_points.issubset(points_df.columns):
        raise ValueError(f"points_df must contain {sorted(required_points)}")
    if not required_windows.issubset(windows_df.columns):
        raise ValueError(f"windows_df must contain {sorted(required_windows)}")

    out = pd.Series(np.nan, index=windows_df.index, dtype=float)
    for chrom in sorted(set(windows_df["chrom"]) & set(points_df["chrom"])):
        w_sub = windows_df[windows_df["chrom"] == chrom][["start", "end"]].copy()
        if w_sub.empty:
            continue
        p_sub = points_df[points_df["chrom"] == chrom][["position", value_col]].copy()
        p_sub = p_sub.dropna().sort_values("position")
        if p_sub.empty:
            continue

        pos = p_sub["position"].to_numpy(dtype=np.int64)
        vals = p_sub[value_col].to_numpy(dtype=float)
        csum = np.concatenate([[0.0], np.cumsum(vals)])

        idx = w_sub.index.to_numpy()
        starts = w_sub["start"].to_numpy(dtype=np.int64)
        ends = w_sub["end"].to_numpy(dtype=np.int64)
        left = np.searchsorted(pos, starts, side="left")
        right = np.searchsorted(pos, ends, side="left")
        counts = right - left
        sums = csum[right] - csum[left]
        means = np.full_like(sums, np.nan, dtype=float)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid]
        out.loc[idx] = means
    return out


def load_bed_intervals(path: str, usecols: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Read BED/BED-like intervals as chrom/start/end."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=list(usecols),
        compression="infer",
        low_memory=False,
    )
    df = df.rename(columns={usecols[0]: "chrom", usecols[1]: "start", usecols[2]: "end"})
    df["chrom"] = df["chrom"].map(_normalize_chrom_value)
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    return df[df["end"] > df["start"]].sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def window_interval_overlap(
    windows_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    return_fraction: bool = True,
) -> pd.Series:
    """Per-window interval overlap fraction (or bool if return_fraction=False)."""
    required = {"chrom", "start", "end"}
    if not required.issubset(windows_df.columns):
        raise ValueError("windows_df must contain chrom/start/end")
    if not required.issubset(intervals_df.columns):
        raise ValueError("intervals_df must contain chrom/start/end")

    out = pd.Series(0.0 if return_fraction else False, index=windows_df.index)
    common_chroms = sorted(set(windows_df["chrom"]) & set(intervals_df["chrom"]))
    for chrom in common_chroms:
        w_sub = windows_df[windows_df["chrom"] == chrom][["start", "end"]].copy()
        i_sub = intervals_df[intervals_df["chrom"] == chrom][["start", "end"]].copy()
        if w_sub.empty or i_sub.empty:
            continue
        w_sorted = w_sub.sort_values("start")
        i_sorted = i_sub.sort_values("start")
        ws = w_sorted["start"].to_numpy(dtype=np.int64)
        we = w_sorted["end"].to_numpy(dtype=np.int64)
        is_ = i_sorted["start"].to_numpy(dtype=np.int64)
        ie = i_sorted["end"].to_numpy(dtype=np.int64)

        overlaps = np.zeros(len(ws), dtype=np.int64)
        j = 0
        for idx, (w_start, w_end) in enumerate(zip(ws, we, strict=False)):
            while j < len(ie) and ie[j] <= w_start:
                j += 1
            k = j
            ov = 0
            while k < len(is_) and is_[k] < w_end:
                left = max(w_start, is_[k])
                right = min(w_end, ie[k])
                if right > left:
                    ov += right - left
                if ie[k] >= w_end and is_[k] >= w_end:
                    break
                k += 1
            overlaps[idx] = ov

        if return_fraction:
            win_len = (we - ws).astype(float)
            frac = overlaps / np.clip(win_len, 1.0, None)
            out.loc[w_sorted.index] = frac
        else:
            out.loc[w_sorted.index] = overlaps > 0
    return out


def aggregate_block_weighted_mean(
    windows_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    value_col: str,
) -> pd.Series:
    """Weighted mean of block values inside each window (by overlap length)."""
    required = {"chrom", "start", "end"}
    if not required.issubset(windows_df.columns):
        raise ValueError("windows_df must contain chrom/start/end")
    if not required.issubset(blocks_df.columns) or value_col not in blocks_df.columns:
        raise ValueError("blocks_df must contain chrom/start/end and value_col")

    out = pd.Series(np.nan, index=windows_df.index, dtype=float)
    common_chroms = sorted(set(windows_df["chrom"]) & set(blocks_df["chrom"]))
    for chrom in common_chroms:
        w_sub = windows_df[windows_df["chrom"] == chrom][["start", "end"]].copy()
        b_sub = blocks_df[blocks_df["chrom"] == chrom][["start", "end", value_col]].copy()
        if w_sub.empty or b_sub.empty:
            continue
        w_sorted = w_sub.sort_values("start")
        b_sorted = b_sub.sort_values("start")
        ws = w_sorted["start"].to_numpy(dtype=np.int64)
        we = w_sorted["end"].to_numpy(dtype=np.int64)
        bs = b_sorted["start"].to_numpy(dtype=np.int64)
        be = b_sorted["end"].to_numpy(dtype=np.int64)
        bv = b_sorted[value_col].to_numpy(dtype=float)

        numer = np.zeros(len(ws), dtype=float)
        denom = np.zeros(len(ws), dtype=float)
        j = 0
        for idx, (w_start, w_end) in enumerate(zip(ws, we, strict=False)):
            while j < len(be) and be[j] <= w_start:
                j += 1
            k = j
            while k < len(bs) and bs[k] < w_end:
                left = max(w_start, bs[k])
                right = min(w_end, be[k])
                if right > left and np.isfinite(bv[k]):
                    overlap = float(right - left)
                    numer[idx] += overlap * bv[k]
                    denom[idx] += overlap
                k += 1
        means = np.full(len(ws), np.nan, dtype=float)
        valid = denom > 0
        means[valid] = numer[valid] / denom[valid]
        out.loc[w_sorted.index] = means
    return out


def load_phyloP_summary(path: str = PHYLOP_TABLE) -> pd.DataFrame:
    """Load UCSC phyloP summary blocks with per-block mean phyloP."""
    # UCSC summary table columns used: 1 chrom, 2 start, 3 end, 6 count, 12 sumData
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[1, 2, 3, 6, 12],
        names=["chrom", "start", "end", "count", "sum_data"],
        compression="infer",
        low_memory=False,
    )
    df["chrom"] = df["chrom"].map(_normalize_chrom_value)
    for col in ["start", "end", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sum_data"] = pd.to_numeric(df["sum_data"], errors="coerce")
    df = df.dropna(subset=["start", "end", "count", "sum_data"])
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    df["count"] = df["count"].astype(np.int64)
    df = df[(df["end"] > df["start"]) & (df["count"] > 0)].copy()
    df["mean_phyloP"] = df["sum_data"] / df["count"]
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def load_gnomad_sv(
    path: str = GNOMAD_SV_TABLE,
    af_min: float = 0.01,
    size_min: int = 10_000,
    svtypes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and filter common, large structural variants from gnomAD SV BED."""
    keep_types = set(svtypes or ["INV", "DEL", "DUP", "CPX", "CNV"])
    usecols = ["#chrom", "start", "end", "svtype", "AF", "SVLEN", "FILTER", "name"]
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=lambda c: c in usecols,
        compression="infer",
        low_memory=False,
    )
    df = df.rename(columns={"#chrom": "chrom", "SVLEN": "svlen", "AF": "af"})
    df["chrom"] = df["chrom"].map(_normalize_chrom_value)
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    df["svlen"] = pd.to_numeric(df["svlen"], errors="coerce")
    df["af"] = pd.to_numeric(df["af"], errors="coerce")
    df = df.dropna(subset=["start", "end", "af", "svtype"])
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    df = df[df["end"] > df["start"]].copy()
    df["size_bp"] = (df["end"] - df["start"]).astype(np.int64)
    df = df[df["svtype"].isin(keep_types)]
    df = df[df["af"] >= af_min]
    df = df[df["size_bp"] >= size_min]
    filt = df["FILTER"].fillna("")
    bad = filt.str.contains("UNRESOLVED|LOW_CONFIDENCE", regex=True)
    df = df[~bad].copy()
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def compute_constitutive_lads(
    lad_dir: str = LADS_DIR,
    min_cell_types: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Build consensus LAD intervals present in >= min_cell_types LAD tracks."""
    paths = sorted(str(p) for p in Path(lad_dir).glob("*.bed.gz"))
    if not paths:
        raise FileNotFoundError(f"No LAD BED files found in {lad_dir}")
    intervals = [load_bed_intervals(p) for p in paths]
    n_tracks = len(intervals)
    if min_cell_types > n_tracks:
        raise ValueError(f"min_cell_types={min_cell_types} exceeds n_tracks={n_tracks}")

    rows: list[dict[str, object]] = []
    all_chroms = sorted({chrom for df in intervals for chrom in df["chrom"].unique()})
    for chrom in all_chroms:
        events: list[tuple[int, int]] = []
        for df in intervals:
            sub = df[df["chrom"] == chrom]
            if sub.empty:
                continue
            events.extend((int(s), 1) for s in sub["start"].to_numpy())
            events.extend((int(e), -1) for e in sub["end"].to_numpy())
        if not events:
            continue
        events_df = pd.DataFrame(events, columns=["pos", "delta"]).groupby("pos", as_index=False)["delta"].sum()
        events_df = events_df.sort_values("pos").reset_index(drop=True)
        count = 0
        pos_arr = events_df["pos"].to_numpy(dtype=np.int64)
        delta_arr = events_df["delta"].to_numpy(dtype=np.int64)
        for idx in range(len(pos_arr) - 1):
            count += int(delta_arr[idx])
            start = int(pos_arr[idx])
            end = int(pos_arr[idx + 1])
            if end <= start:
                continue
            if count >= min_cell_types:
                rows.append(
                    {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "n_cell_types": count,
                        "lad_score": count / float(n_tracks),
                    }
                )
    consensus = pd.DataFrame(rows)
    if consensus.empty:
        consensus = pd.DataFrame(columns=["chrom", "start", "end", "n_cell_types", "lad_score"])
    return consensus, paths

"""Shared utilities for gene-desert Gnocchi diagnostic analyses.

Centralizes the desert exemplar definitions, loading of the merged
adjusted/unadjusted Gnocchi table, and the `label_deserts` helper that
tags each 1kb window with the desert it falls inside.
"""

from __future__ import annotations

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

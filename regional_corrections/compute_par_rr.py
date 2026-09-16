#!/usr/bin/env python3
"""Compute the regional feature adjustment (rr) and adjusted Gnocchi z for chrX PAR.

Reproduces the released gnomAD pipeline's regional adjustment
(`run_nc_constraint_gnomad_v31_main.py`, "Adjust the effects of regional genomic
features on mutation rates, per 1kb") and applies it to the chrX PAR windows,
whose features were missing from the published autosome-only feature matrix and
therefore silently defaulted to rr = 1.

Recipe, verified to reproduce the published autosomal `expected` to a median
relative error of 4e-09 (see --validate):

    df_x = (raw_features - ft_mean) / ft_std      # shipped per-context raw stats
    pred = logit.predict(add_constant(pca.transform(df_x)))
    ave  = logit.predict(add_constant(zeros(1, n_features)))
    rr   = pred / ave                             # 1.0 at the average window
    expected_adj = sum_over_contexts(expected_raw * rr)

Masking. Six of the 13 features have no usable data in PAR (see
build_par_features.MASKED_FEATURES). They are retained as model inputs -- the
PCA requires its fixed input width -- but their standardized value is pinned to
0, i.e. "assume the genome-average window". Two traps this avoids:
  * leaving them NaN would trip the upstream per-context `.dropna()` and delete
    every PAR window, silently restoring rr = 1 -- the exact bug being fixed;
  * dropping the columns would change the PCA input width.

Caveat that belongs with every number this produces: `recomb_male` is selected
in 31 of 32 contexts and is among the masked features, yet obligate male
recombination (~20x genome average) is PAR1's defining feature. This correction
accounts for sequence and annotation composition, not recombination.

Usage:
    python regional_corrections/compute_par_rr.py             # adjust chrX PAR
    python regional_corrections/compute_par_rr.py --validate  # autosomal gate
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import types
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_utils import compute_gnocchi_z  # noqa: E402
from utils.desert_utils import DATA_DIR, RESULTS_DIR  # noqa: E402

from build_par_features import MASKED_FEATURES  # noqa: E402

PICKLE_DIR = os.path.join(DATA_DIR, "logit_pickles")
SEL_TABLE = os.path.join(DATA_DIR, "genomic_features13_sel.txt")
PAR_FEATURES = os.path.join(DATA_DIR, "genomic_features13_chrX_par_1kb.txt.gz")
PAR_GNOCCHI = os.path.join(DATA_DIR, "gnocchi_1kb_chrX_par_public.txt.gz")
EXPECTED_CTX = os.path.join(DATA_DIR, "expected_counts_per_context_methyl_genome_1kb.txt.gz")
PAR_EXPECTED_CACHE = os.path.join(DATA_DIR, "expected_counts_per_context_methyl_chrX_par.txt.gz")
OUT_TABLE = os.path.join(RESULTS_DIR, "par_regional_adjustment.tsv.gz")

# Features the pipeline drops for CpG contexts before selecting columns.
FT_CORR_MET = ["GC_content", "SINE", "met_sperm", "Nucleosome", "CpG_island"]
CPG_CONTEXTS = ["ACG", "CCG", "GCG", "TCG"]

# Upstream clips z to this range.
Z_CLIP = (-10.0, 10.0)


def _register_pandas_shim() -> None:
    """Allow unpickling models written under pandas <2.

    The released pickles embed references to `pandas.core.indexes.numeric`,
    removed in pandas 2.x. Without this, some contexts fail to load.
    """
    name = "pandas.core.indexes.numeric"
    if name in sys.modules:
        return
    shim = types.ModuleType(name)
    for attr in ("Int64Index", "Float64Index", "NumericIndex", "UInt64Index"):
        setattr(shim, attr, pd.Index)
    sys.modules[name] = shim


def feature_base(column: str) -> str:
    """'cDNM_maternal_05M_1k' -> 'cDNM_maternal_05M'."""
    return column.rsplit("_", 1)[0]


def load_selected_features() -> dict[str, list[str]]:
    """Per-context model input columns, in the order the models were fit on.

    Order is authoritative and matches each `.ft_mean_std.txt` exactly (verified
    for all 32 contexts).
    """
    sel = pd.read_csv(SEL_TABLE, sep="\t")
    out: dict[str, list[str]] = {}
    for ctx in sorted(sel["context"].unique()):
        s = sel[sel["context"] == ctx]
        if ctx in CPG_CONTEXTS:
            s = s[~s["feature"].isin(FT_CORR_MET)]
        out[ctx] = list(s["feature"] + "_" + s["window"])
    return out


def load_context_model(ctx: str):
    """Load one context's logit, PCA and standardization parameters."""
    base = os.path.join(PICKLE_DIR, f"logit_regularized_dnm01_{ctx}_pbonf_pca")
    with open(base + ".pkl", "rb") as fh:
        logit = pickle.load(fh)
    with open(base + ".pca.pkl", "rb") as fh:
        pca = pickle.load(fh)
    ms = pd.read_csv(base + ".ft_mean_std.txt", sep="\t", header=None,
                     names=["feature", "mean", "std"])
    return logit, pca, dict(zip(ms["feature"], ms["mean"])), dict(zip(ms["feature"], ms["std"]))


def compute_context_rr(features: pd.DataFrame, ctx: str, ft_sel: list[str],
                       masked: list[str] | None) -> pd.DataFrame:
    """Relative rate per window for one trinucleotide context.

    Returns columns `element_id` and `rr_<ctx>`. Windows missing any *unmasked*
    feature are excluded here and later filled with rr = 1, matching upstream.
    """
    logit, pca, ft_mean, ft_std = load_context_model(ctx)
    masked_cols = [c for c in ft_sel if masked and feature_base(c) in masked]
    keep_cols = [c for c in ft_sel if c not in masked_cols]

    frame = features[["element_id"] + ft_sel].drop_duplicates()
    frame = frame.dropna(subset=keep_cols)
    if frame.empty:
        return pd.DataFrame(columns=["element_id", f"rr_{ctx}"])

    x = frame[ft_sel].copy()
    for col in ft_sel:
        x[col] = (x[col] - ft_mean[col]) / ft_std[col]
    # Pin masked features to the genome-average window.
    for col in masked_cols:
        x[col] = 0.0

    pred = logit.predict(sm.add_constant(pca.transform(x), has_constant="add"))
    ave = logit.predict(
        sm.add_constant(pd.DataFrame([0] * len(ft_sel)).transpose(), has_constant="add")
    )[0]
    return pd.DataFrame({"element_id": frame["element_id"].to_numpy(), f"rr_{ctx}": pred / ave})


def assemble_rr(features: pd.DataFrame, selected: dict[str, list[str]],
                masked: list[str] | None) -> pd.DataFrame:
    """Long-form (element_id, context, rr) across all contexts.

    Mirrors upstream: outer-merge the per-context frames then `.fillna(1)`, so a
    window lacking features for some context contributes an unadjusted rr = 1
    for that context rather than being dropped from the window total.
    """
    frames = []
    used: dict[str, int] = {}
    for ctx, ft_sel in selected.items():
        masked_n = len([c for c in ft_sel if masked and feature_base(c) in masked])
        used[ctx] = len(ft_sel) - masked_n
        frames.append(compute_context_rr(features, ctx, ft_sel, masked))
        print(f"    {ctx}: {len(ft_sel)} model inputs, {used[ctx]} informative")
    wide = reduce(lambda a, b: pd.merge(a, b, on="element_id", how="outer"), frames).fillna(1)

    long = []
    for ctx in selected:
        col = f"rr_{ctx}"
        if col not in wide.columns:
            wide[col] = 1.0
        long.append(pd.DataFrame({"element_id": wide["element_id"], "context": ctx,
                                  "rr": wide[col]}))
    out = pd.concat(long, ignore_index=True)
    out.attrs["features_used"] = used
    return out


def load_par_expected_by_context() -> pd.DataFrame:
    """Per-(window, context) unadjusted expected counts for chrX PAR.

    The genome-wide table is ~540MB, so the chrX slice is extracted once and
    cached; chrX coverage there is PAR-only (2,497 windows x 32 contexts).
    """
    if os.path.exists(PAR_EXPECTED_CACHE):
        return pd.read_csv(PAR_EXPECTED_CACHE, sep="\t")
    print("  extracting chrX rows from the per-context expected table (one-time) ...")
    rows = []
    with gzip.open(EXPECTED_CTX, "rt") as fh:
        fh.readline()
        for line in fh:
            if line.startswith("chrX-"):
                eid, ctx, possible, expected = line.rstrip("\n").split("\t")
                rows.append((eid, ctx, int(possible), float(expected)))
    df = pd.DataFrame(rows, columns=["element_id", "context", "possible", "expected"])
    df.to_csv(PAR_EXPECTED_CACHE, sep="\t", index=False, compression="gzip")
    print(f"  cached {len(df):,} rows -> {PAR_EXPECTED_CACHE}")
    return df


def adjusted_expected(expected_by_ctx: pd.DataFrame, rr_long: pd.DataFrame) -> pd.DataFrame:
    """expected_adj = sum over contexts of expected_raw * rr."""
    m = expected_by_ctx.merge(rr_long, on=["element_id", "context"], how="inner")
    m["predicted"] = m["expected"] * m["rr"]
    agg = m.groupby("element_id").agg(
        expected_raw=("expected", "sum"),
        expected_adj=("predicted", "sum"),
        rr_mean=("rr", "mean"),
    )
    return agg.reset_index()


def run_par() -> None:
    _register_pandas_shim()
    selected = load_selected_features()

    print("Loading chrX PAR feature matrix ...")
    features = pd.read_csv(PAR_FEATURES, sep="\t")
    print(f"  {len(features):,} windows")

    print("Computing rr per context (6 features masked) ...")
    rr_long = assemble_rr(features, selected, MASKED_FEATURES)
    used = rr_long.attrs["features_used"]

    print("Loading per-context expected counts ...")
    exp_ctx = load_par_expected_by_context()

    agg = adjusted_expected(exp_ctx, rr_long)

    par = pd.read_csv(PAR_GNOCCHI, sep="\t")
    out = par.merge(agg, on="element_id", how="left")
    out["z_unadj"] = compute_gnocchi_z(out["observed"], out["expected_raw"])
    out["z_adj_new"] = compute_gnocchi_z(out["observed"], out["expected_adj"])
    for col in ("z_unadj", "z_adj_new"):
        out[col] = out[col].clip(*Z_CLIP)
    # Repo convention (see unadjusted_gnocchi_analysis.py): delta_z = z_unadj - z_adj,
    # so PAR figures stay sign-consistent with the gene-desert analyses.
    out["delta_z"] = out["z_unadj"] - out["z_adj_new"]
    out["n_features_used_min"] = min(used.values())
    out["n_features_used_median"] = int(np.median(list(used.values())))

    cols = ["element_id", "chrom", "start", "end", "observed", "possible",
            "expected_raw", "expected_adj", "rr_mean", "z_unadj", "z_adj_new", "delta_z",
            "n_features_used_min", "n_features_used_median", "pct_pass", "mean_coverage", "pass_qc"]
    out[cols].to_csv(OUT_TABLE, sep="\t", index=False, compression="gzip")
    print(f"\n  wrote {OUT_TABLE}")

    report(out, used)


def report(out: pd.DataFrame, used: dict[str, int]) -> None:
    """Diagnostics required by the plan's sanity-check list."""
    print("\n=== chrX PAR regional adjustment ===")
    out = out.copy()
    out["region"] = np.where(out["start"] < 3_000_000, "PAR1", "PAR2")

    # QC reproduction: validates our reading of the shipped table.
    recomputed = (out["pct_pass"] >= 0.8) & out["mean_coverage"].between(25, 35) & (out["possible"] >= 1000)
    print(f"  pass_qc reproduced from pct_pass/coverage/possible: "
          f"{int((recomputed == out['pass_qc']).sum())}/{len(out)} agree "
          f"({int(out['pass_qc'].sum())} pass)")

    n_unit = int((out["rr_mean"].sub(1).abs() < 1e-12).sum())
    print(f"  windows still at rr == 1 exactly: {n_unit} (was 2,497 before this fix)")

    for region, sub in out.groupby("region"):
        print(f"\n  {region} (n={len(sub)})")
        print(f"    rr_mean        : mean {sub['rr_mean'].mean():.4f}  sd {sub['rr_mean'].std():.4f}  "
              f"range [{sub['rr_mean'].min():.4f}, {sub['rr_mean'].max():.4f}]")
        print(f"    expected shift : median {100 * (sub['expected_adj'] / sub['expected_raw'] - 1).median():+.2f}%")
        print(f"    z_unadj        : mean {sub['z_unadj'].mean():+.4f}  median {sub['z_unadj'].median():+.4f}")
        print(f"    z_adj_new      : mean {sub['z_adj_new'].mean():+.4f}  median {sub['z_adj_new'].median():+.4f}")
        print(f"    delta_z        : mean {sub['delta_z'].mean():+.4f}  "
              f"p5 {sub['delta_z'].quantile(.05):+.4f}  p95 {sub['delta_z'].quantile(.95):+.4f}")

    support_check()

    thin = {c: n for c, n in sorted(used.items(), key=lambda kv: kv[1])[:5]}
    print(f"\n  thinnest contexts (informative features retained): {thin}")
    print(f"  masked features: {', '.join(MASKED_FEATURES)}")
    print("  NOTE: recomb_male is selected in 31/32 contexts and is masked -- this")
    print("        correction covers sequence/annotation composition, NOT recombination.")


def support_check() -> None:
    """Flag PAR feature values outside the autosomal training range.

    Percentiles are misleading here because most 1kb windows have exactly 0 for
    the annotation features, so a large tie-mass sits at percentile 0. Comparing
    against the autosomal [min, max] is the meaningful test.
    """
    from utils.desert_utils import load_features  # noqa: E402

    par = pd.read_csv(PAR_FEATURES, sep="\t")
    auto = load_features()
    cols = [c for c in par.columns if c != "element_id" and par[c].notna().any()]
    total = below = above = 0
    offenders = []
    for col in cols:
        ref = auto[col].dropna()
        vals = par[col].dropna()
        nb = int((vals < ref.min()).sum())
        na = int((vals > ref.max()).sum())
        total += len(vals)
        below += nb
        above += na
        if nb + na:
            offenders.append((col, nb, na, float(ref.max()), float(vals.max())))
    print(f"\n  feature values outside the autosomal training range: "
          f"{below + above}/{total} ({100 * (below + above) / total:.3f}%)  "
          f"[{below} below min, {above} above max]")
    for col, nb, na, ref_max, par_max in offenders:
        print(f"    {col}: {na} above autosomal max ({par_max:.2f} vs {ref_max:.2f})")


def run_mask_ablation(n_rows: int) -> None:
    """Quantify what masking costs, measured on autosomes where truth exists.

    Runs the verified machinery twice on the same windows -- all 13 features vs
    the 6-masked PAR configuration -- so the PAR numbers can be reported with an
    empirical error bar rather than an assumption.
    """
    import contextlib
    import io

    from utils.desert_utils import load_features  # noqa: E402

    _register_pandas_shim()
    selected = load_selected_features()

    rows = []
    with gzip.open(EXPECTED_CTX, "rt") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i >= n_rows:
                break
            eid, ctx, _, expected = line.rstrip("\n").split("\t")
            rows.append((eid, ctx, float(expected)))
    exp_ctx = pd.DataFrame(rows, columns=["element_id", "context", "expected"])
    features = load_features()
    features = features[features["element_id"].isin(set(exp_ctx["element_id"]))].reset_index(drop=True)

    print("=== MASKING ABLATION (autosomal windows) ===")
    eff = {}
    for label, mask in [("all 13 features", None), ("6 masked (PAR config)", MASKED_FEATURES)]:
        with contextlib.redirect_stdout(io.StringIO()):
            agg = adjusted_expected(exp_ctx, assemble_rr(features, selected, mask)).set_index("element_id")
        eff[label] = agg["expected_adj"] / agg["expected_raw"]
        print(f"  {label:24s} rr_eff mean {eff[label].mean():.4f}  sd {eff[label].std():.4f}  "
              f"median shift {100 * (eff[label] - 1).median():+.2f}%")
    j = pd.DataFrame(eff).dropna()
    full, masked = j.iloc[:, 0], j.iloc[:, 1]
    print(f"\n  corr(full, masked)   : {np.corrcoef(full, masked)[0, 1]:.4f}")
    print(f"  SD retained          : {masked.std() / full.std():.1%}")
    print(f"  masking bias         : median {100 * (masked / full - 1).median():+.2f}% on expected")


def run_validate(n_rows: int) -> int:
    """Autosomal gate: reproduce published `expected` with all 13 features, unmasked.

    A unit test of the machinery, not an autosomal result. PAR offers no ground
    truth, so this is the only place the recipe can be checked at all.
    """
    _register_pandas_shim()
    selected = load_selected_features()

    print(f"Streaming {n_rows:,} rows of per-context expected counts ...")
    rows = []
    with gzip.open(EXPECTED_CTX, "rt") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i >= n_rows:
                break
            eid, ctx, possible, expected = line.rstrip("\n").split("\t")
            rows.append((eid, ctx, float(expected)))
    exp_ctx = pd.DataFrame(rows, columns=["element_id", "context", "expected"])
    wins = set(exp_ctx["element_id"])
    print(f"  {len(wins):,} windows")

    from utils.desert_utils import load_features  # noqa: E402
    features = load_features()
    features = features[features["element_id"].isin(wins)].reset_index(drop=True)

    print("Computing rr per context (unmasked, all 13 features) ...")
    rr_long = assemble_rr(features, selected, masked=None)
    agg = adjusted_expected(exp_ctx, rr_long).set_index("element_id")

    pub = pd.read_csv("data/constraint_z_genome_1kb.qc.download.txt.gz", sep="\t",
                      usecols=["element_id", "expected"]).set_index("element_id")["expected"]
    cmp = agg.join(pub.rename("published"), how="inner").dropna()
    rel = (cmp["expected_adj"] - cmp["published"]).abs() / cmp["published"]

    print("\n=== RR GATE ===")
    print(f"  windows compared : {len(cmp):,}")
    print(f"  corr             : {np.corrcoef(cmp['expected_adj'], cmp['published'])[0, 1]:.8f}")
    print(f"  median rel err   : {rel.median():.4e}")
    print(f"  frac < 0.1%      : {(rel < 1e-3).mean():.6f}")
    passed = rel.median() < 1e-3 and (rel < 1e-3).mean() > 0.99
    print(f"  verdict          : {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="Run the autosomal correctness gate and exit.")
    ap.add_argument("--mask-ablation", action="store_true",
                    help="Measure on autosomes what the 6-feature masking costs, and exit.")
    ap.add_argument("--validate-rows", type=int, default=1_500_000)
    args = ap.parse_args()
    if args.validate:
        sys.exit(run_validate(args.validate_rows))
    if args.mask_ablation:
        run_mask_ablation(args.validate_rows)
        return
    run_par()


if __name__ == "__main__":
    main()

"""
C2ST diagnosis: WHY can a classifier separate synthetic rows from real
ones with AUC ~1.0?

    python c2st_diagnose.py --file output/generate/DT4H_Synthetic_tvae_seed0.csv

Two analyses:
  1. Single-feature tells: for every column, the AUC of that column ALONE
     separating real from synthetic (rank AUC for numerics; frequency-
     ratio scoring for categoricals). The top of this list names the
     features that give the game away.
  2. Granularity ablation: real clinical values live on grids (integer
     ages and blood pressures, two-decimal assays) while samplers emit
     arbitrary floats. Synthetic numerics are snapped to each column's
     real decimal granularity and the full C2ST is re-run. A large AUC
     drop attributes the separability to value formatting rather than
     joint structure -- exactly the evidence the 'granularity-snapping
     decode' future-work item needs.

Prints a report; add --json <path> to save it.
"""

import argparse
import json

import numpy as np
import pandas as pd
import polars as pl

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case
from pipeline.steps.evaluate.c2st import c2st_auc
from pipeline.steps.generate.step import GenerateStep


def single_feature_auc(real: pd.Series, synth: pd.Series) -> float:
    """RAW AUC of one column separating real (0) from synthetic (1).

    Numerics: rank AUC (Mann-Whitney), with NaN placed on its own rank
    below the observed values so missingness differences count.
    Categoricals: score each row by the synthetic/real frequency ratio
    of its category -- the AUC of the optimal single-column classifier.

    Deliberately NOT folded to max(auc, 1-auc): the folded statistic has
    null expectation ABOVE 0.5 (pure noise folds upward), which made
    unremarkable columns read as mild tells. The raw value keeps the
    direction too: >0.5 means synthetic rows score higher under this
    column's ranking, <0.5 means real rows do.
    """
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    if pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(synth):
        vals = pd.concat([pd.to_numeric(real, errors="coerce"),
                          pd.to_numeric(synth, errors="coerce")], ignore_index=True)
        filled = vals.fillna(vals.min() - 1.0)
        score = filled.rank().to_numpy()
    else:
        r = real.astype("object").where(real.notna(), "Missing").astype(str)
        s = synth.astype("object").where(synth.notna(), "Missing").astype(str)
        pr = r.value_counts(normalize=True)
        ps = s.value_counts(normalize=True)
        cats = pr.index.union(ps.index)
        ratio = {c: (ps.get(c, 1e-9)) / (pr.get(c, 1e-9)) for c in cats}
        score = np.concatenate([r.map(ratio).to_numpy(dtype=float),
                                s.map(ratio).to_numpy(dtype=float)])
    return round(float(roc_auc_score(y, score)), 4)


def real_granularity(col: pd.Series, sample: int = 2000) -> float | None:
    """The real column's value grid: 10^-d for the maximum number of
    decimal places observed (capped at 6). None if not numeric."""
    v = pd.to_numeric(col, errors="coerce").dropna()
    if v.empty:
        return None
    if len(v) > sample:
        # Random and seeded rather than the frame's head: row order can
        # carry structure (load order, site, time), and the head is then
        # not exchangeable with the rest of the column.
        v = v.sample(n=sample, random_state=0)
    decimals = 0
    for x in v:
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        decimals = max(decimals, len(s.split(".")[1]) if "." in s else 0)
        if decimals >= 6:
            break
    return 10.0 ** -decimals


def snap_to_granularity(synth: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    out = synth.copy()
    snapped = 0
    for c in real.columns:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(real[c]):
            continue
        g = real_granularity(real[c])
        if g is None:
            continue
        v = pd.to_numeric(out[c], errors="coerce")
        out[c] = (v / g).round() * g
        snapped += 1
    print(f"Snapped {snapped} numeric column(s) to the real data's value grids.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose C2ST separability.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    config = PipelineConfig()

    train = pl.read_parquet(config.train_output_path).to_pandas()
    train, _ = GenerateStep._decode_numeric_missing(train, config)
    synth = pd.read_csv(args.file, low_memory=False)
    synth, _ = align_categorical_case(synth, train)
    columns = [c for c in train.columns
               if c in synth.columns and train[c].nunique(dropna=False) > 1]

    print(f"Real train: {train.shape[0]} rows | synthetic: {synth.shape[0]} rows | "
          f"{len(columns)} comparable columns\n")

    print("Single-feature tells (raw AUC of ONE column separating real=0 from "
          "synthetic=1;\n0.5 = uninformative, deviation in EITHER direction is a tell):")
    tells = sorted(((c, single_feature_auc(train[c], synth[c])) for c in columns),
                   key=lambda t: -abs(t[1] - 0.5))
    for c, a in tells[: args.top]:
        kind = "num" if pd.api.types.is_numeric_dtype(train[c]) else "cat"
        direction = "synth-high" if a > 0.5 else "real-high"
        print(f"  {a:.4f}  [{kind}, {direction}] {c}")
    n_strong = sum(1 for _, a in tells if abs(a - 0.5) > 0.1)
    print(f"  ... {n_strong}/{len(tells)} columns alone deviate from 0.5 by more than 0.1\n")

    print("Full C2ST, as released:")
    res_raw = c2st_auc(train, synth, columns, seed=config.seed)
    auc_raw = res_raw["auc"]
    print(f"  AUC = {auc_raw} ± {res_raw['auc_sd']} (out-of-fold, "
          f"{res_raw['columns_used']} columns, coverage {res_raw['schema_coverage']})")

    print("\nFull C2ST after snapping synthetic numerics to real value grids:")
    snapped = snap_to_granularity(synth, train)
    res_snap = c2st_auc(train, snapped, columns, seed=config.seed)
    auc_snap = res_snap["auc"]
    print(f"  AUC = {auc_snap} ± {res_snap['auc_sd']}")

    drop = round(auc_raw - auc_snap, 4)
    print(f"\nVerdict: granularity accounts for an AUC drop of {drop}. "
          + ("Most separability is value formatting, not joint structure."
         if auc_snap < 0.75 else
         "Separability persists after snapping -- joint structure (or other tells) dominates."))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"file": args.file, "c2st_raw": auc_raw,
                       "c2st_raw_detail": res_raw,
                       "c2st_granularity_snapped": auc_snap,
                       "c2st_granularity_snapped_detail": res_snap,
                       "single_feature_tells": tells[:50]}, f, indent=2)
        print(f"Saved -> {args.json}")


if __name__ == "__main__":
    main()

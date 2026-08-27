"""
Release-candidate post-processor: the two bounded remedies the release
gate's failures call for, applied to an already-generated synthetic CSV
without retraining anything.

    python postprocess_candidate.py --file output/generate/DT4H_Synthetic_tvae_seed0.csv \
        [--model output/generate/models/tvae_seed0.pkl] [--out <path>] \
        [--no-snap] [--no-filter]

  1. GRANULARITY SNAP: numeric values are rounded onto each column's
     empirical grid (integer ages and blood pressures, one-decimal
     weights), inferred from the decoded training split. Cosmetic by
     measurement (C2ST unchanged in the ablation), but gives released
     records face validity.
  2. NEAREST-RECORD REJECTION FILTER: records closer to a training
     record than the holdout p5 threshold -- the distance criterion the
     gate enforces -- are dropped. With --model, replacements are drawn
     from the saved generator (same post-processing chain as the
     pipeline) until the target row count is restored or --max-rounds
     is exhausted; without it, the output is simply smaller.

DP runs (run id containing 'eps') are REFUSED: the rejection filter is
data-dependent post-processing against the training data, which voids
the differential-privacy guarantee the run exists for. --force-dp
overrides with a loud warning; the output then carries no DP claim.

The output is re-audited (representation, leakage in decoded space,
distance re-check) and written next to the input as
<name>_postprocessed.csv (or --out). It is a NEW release candidate:
run release_gate.py on it, and treat its fidelity as changed -- the
filter deliberately trades marginal fidelity for distance margin, and
the evaluate step quantifies that honestly if pointed at the file.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case, harmonize_dtypes
from pipeline.common.granularity import infer_granularity, snap_to_granularity
from pipeline.common.representation_audit import audit_representation, summarize as rep_summary
from pipeline.steps.generate import leakage
from pipeline.steps.generate.step import GenerateStep
from pipeline.steps.privacy.distance import build_encoder, nearest_two_distances

MAX_ROUNDS_DEFAULT = 10


def _distances_to_train(frame, train, encode, columns):
    """Nearest-training-record distance for every row of `frame`,
    computed over `columns` -- the intersection of the candidate's and
    the training frame's columns, with `encode` built on that same
    subset. A width-limited candidate used to be NA-padded to full
    width, which guaranteed large distances against the full-width
    threshold and made the filter vacuous."""
    t_num, t_cat = encode(train[columns])
    s_num, s_cat = encode(frame[columns])
    d1, _ = nearest_two_distances(s_num, s_cat, t_num, t_cat)
    return d1


def _fresh_rows(model_path, n, train_sentinel, train_decoded, constants, config):
    """Sample replacement rows from the saved generator through the same
    post-processing chain as the generate step: align, re-attach
    constants, column order, verbatim-leakage check in sentinel space,
    sentinel decode."""
    from pipeline.common.model_io import load_generator

    synth = load_generator(model_path)
    batch = synth.sample(n)
    batch, _ = align_categorical_case(batch, train_sentinel)
    if constants:
        pad = pd.DataFrame({c: [v] * len(batch) for c, v in constants.items()},
                           index=batch.index)
        batch = pd.concat([batch, pad], axis=1)
    batch = batch[[c for c in train_sentinel.columns if c in batch.columns]].copy()
    leak = leakage.check_exact_duplicates(batch, train_sentinel)
    if leak.get("exact_duplicates_of_training_rows", 0) > 0:
        raise RuntimeError("Replacement batch reproduced a training row verbatim -- "
                           "refusing to continue.")
    batch, _ = GenerateStep._decode_numeric_missing(batch, config)
    return batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Snap + distance-filter a synthetic release candidate.")
    parser.add_argument("--file", required=True, help="Candidate DT4H_Synthetic_*.csv")
    parser.add_argument("--model", help="Saved generator .pkl for top-up sampling (optional; "
                        "requires the generation-time numpy, see requirements.txt).")
    parser.add_argument("--out", help="Output CSV (default: <input>_postprocessed.csv)")
    parser.add_argument("--no-snap", action="store_true", help="Skip granularity snapping.")
    parser.add_argument("--no-filter", action="store_true", help="Skip the distance filter.")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS_DEFAULT,
                        help="Top-up sampling rounds when --model is given.")
    parser.add_argument("--force-dp", action="store_true",
                        help="Post-process a DP run's output anyway. The distance "
                             "filter is data-dependent post-processing that VOIDS the "
                             "run's differential-privacy guarantee -- only pass this "
                             "with a governance decision behind it.")
    args = parser.parse_args()
    config = PipelineConfig()
    # Default name deliberately does NOT match the DT4H_Synthetic_*.csv
    # glob the analysis steps ingest -- a post-processed candidate must
    # not be silently double-counted as an extra run. Rename it
    # explicitly if it is promoted to a released file.
    stem = os.path.basename(args.file).replace(".csv", "")
    if stem.startswith("DT4H_Synthetic_"):
        stem = stem[len("DT4H_Synthetic_"):]
    out_path = args.out or os.path.join(os.path.dirname(args.file),
                                        f"DT4H_Candidate_{stem}_postprocessed.csv")

    # A DP run's output must NOT go through the rejection filter: dropping
    # records by their distance to the TRAINING data is data-dependent
    # post-processing, which voids the differential-privacy guarantee the
    # run was made for (the guarantee only survives data-INDEPENDENT
    # post-processing). Run ids carry the budget as 'eps<value>'.
    if "eps" in stem:
        if not args.force_dp:
            print(f"🚫 '{stem}' looks like a DP run (run id contains 'eps'). "
                  f"The nearest-record rejection filter consults the training data, "
                  f"so applying it VOIDS the DP guarantee -- the released file could "
                  f"no longer honestly claim its epsilon. Refusing. If the file is "
                  f"knowingly released WITHOUT its DP claim, re-run with --force-dp.")
            return 2
        print("⚠️  ⚠️  ⚠️  --force-dp: post-processing a DP run's output. The output "
              "file's differential-privacy guarantee is VOID -- it must not be "
              "released or documented as differentially private. ⚠️  ⚠️  ⚠️")

    train = pl.read_parquet(config.train_output_path).to_pandas()
    train_decoded, _ = GenerateStep._decode_numeric_missing(train.copy(), config)

    priv_path = os.path.join(config.step_dir("privacy"), "DT4H_Privacy_Assessment.json")
    with open(priv_path) as f:
        committed_p5 = json.load(f)["holdout_baseline"]["dcr_p5"]

    enc_path = os.path.join(config.step_dir("preprocess"), "DT4H_Numeric_Missing_Encoding.json")
    encoding = json.load(open(enc_path)) if os.path.exists(enc_path) else {}

    summary_path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
    constants = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            constants = json.load(f).get("constant_columns_held_out", {})

    candidate = pd.read_csv(args.file, low_memory=False, float_precision="round_trip")
    candidate, _ = align_categorical_case(candidate, train_decoded)
    target_rows = len(candidate)
    print(f"Candidate: {target_rows} x {candidate.shape[1]} ({os.path.basename(args.file)})")

    # SUBSET-AWARE distances: encoder and threshold both live on the
    # intersection of candidate and train columns. A width-limited
    # candidate compared NA-padded against the FULL-width holdout p5
    # gets guaranteed-large distances and the filter turns vacuous, so
    # the p5 baseline is recomputed on the shared subset in that case.
    subset = [c for c in train.columns if c in candidate.columns]
    encode, _, _ = build_encoder(train[subset], encoding)
    if len(subset) == len(train.columns):
        p5 = committed_p5
    else:
        holdout = pl.read_parquet(config.holdout_output_path).to_pandas()
        h_num, h_cat = encode(holdout[subset])
        t_num, t_cat = encode(train[subset])
        dh, _ = nearest_two_distances(h_num, h_cat, t_num, t_cat)
        p5 = round(float(np.percentile(dh, 5)), 6)
        print(f"⚠️  width-limited candidate: {len(subset)}/{len(train.columns)} "
              f"columns shared with the training frame; holdout p5 recomputed on "
              f"that subset = {p5} (committed full-width p5 {committed_p5} does "
              f"not apply).")

    grid = infer_granularity(train_decoded[[c for c in candidate.columns
                                            if c in train_decoded.columns]])

    if not args.no_snap:
        candidate, snapped = snap_to_granularity(candidate, grid)
        print(f"Granularity snap: {sum(snapped.values())} cell(s) in {len(snapped)} "
              f"column(s) moved onto the observed grid "
              f"({len(grid)} gridded columns detected).")

    if not args.no_filter:
        d1 = _distances_to_train(candidate, train, encode, subset)
        close = d1 < p5
        print(f"Distance filter: {int(close.sum())}/{len(candidate)} record(s) "
              f"({close.mean():.1%}) below the holdout p5 threshold ({p5}, "
              f"over {len(subset)} column(s)) -- dropping them "
              f"(natural rate for real unseen patients: 5%).")
        candidate = candidate.loc[~close].reset_index(drop=True)

        rounds = 0
        while args.model and len(candidate) < target_rows and rounds < args.max_rounds:
            rounds += 1
            need = target_rows - len(candidate)
            print(f"  top-up round {rounds}: sampling {need * 2} replacement row(s)...")
            batch = _fresh_rows(args.model, max(need * 2, 200), train, train_decoded,
                                constants, config)
            if not args.no_snap:
                batch, _ = snap_to_granularity(batch, grid)
            bd = _distances_to_train(batch, train, encode, subset)
            keep = batch.loc[bd >= p5]
            # A generator resampled with the same state can return the
            # same batch again -- defensively drop rows identical to an
            # already-kept candidate row (or repeated inside the batch)
            # before topping up, so the output never carries duplicates.
            if len(keep):
                pooled = pd.concat([candidate, keep], ignore_index=True)
                dup = pooled.duplicated(keep="first").to_numpy()[len(candidate):]
                if dup.any():
                    print(f"    dropped {int(dup.sum())} duplicate replacement row(s) "
                          f"(identical to already-kept rows or repeated in the batch)")
                keep = keep[~dup]
            keep = keep.head(need)
            print(f"    kept {len(keep)} (passed the distance threshold); "
                  f"still need {max(need - len(keep), 0)}")
            if len(keep):
                candidate = pd.concat([candidate, keep], ignore_index=True)
        if len(candidate) < target_rows:
            print(f"⚠️  Output has {len(candidate)} rows (< target {target_rows})"
                  + ("" if args.model else " -- pass --model to top up from the saved generator."))

    # Final self-checks before writing: this file is a release candidate.
    # (Decode sets pd.NA into float columns; in-memory concat of top-up
    # rows can upcast those to object -- coerce back before auditing.)
    candidate, coerced = harmonize_dtypes(candidate, train_decoded)
    if coerced:
        print(f"Harmonized {len(coerced)} numeric column(s) upcast by in-memory concat.")
    rep = audit_representation(candidate, train_decoded)
    print(f"Representation: {rep_summary(rep)}")
    leak = leakage.check_exact_duplicates(candidate, train_decoded)
    print(leakage.summarize(leak))
    d1 = _distances_to_train(candidate, train, encode, subset)
    print(f"Final distance profile: min {d1.min():.4f}, "
          f"{(d1 < p5).mean():.1%} below the p5 threshold "
          f"({p5}, over {len(subset)} column(s)).")

    candidate.to_csv(out_path, index=False)
    print(f"✅ {len(candidate)} x {candidate.shape[1]} -> {out_path}")
    print(f"Next: python release_gate.py --file {out_path}")
    return 0 if rep["clean"] and leak.get("exact_duplicates_of_training_rows", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

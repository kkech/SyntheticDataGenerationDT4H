"""
Constraint-aware sampling: draw synthetic records from a SAVED generator
and reject rows that violate the committed clinical-coherence rule set,
without retraining anything.

    python coherent_sample.py --model output/generate/models/mst_eps15_seed0.pkl \
        --rows 3520 --out output/generate/DT4H_Candidate_mst_eps15_coherent.csv

(Name outputs DT4H_Candidate_* -- NOT DT4H_Synthetic_* -- so the
analysis steps' file glob does not silently ingest them as extra runs;
rename explicitly if a file is promoted to the release.)

The released 363-rule set (mined from the real training split; the
holdout's own violation rate is the fair bar) already functions as an
executable specification of row-level clinical coherence. This tool
turns it into a sampling layer: batches are drawn, post-processed
through the same chain as the pipeline (spelling alignment, verbatim-
leakage check in sentinel space, sentinel decode), and rows violating
ANY rule are rejected until the target count of clean rows is reached.

Honest caveats, printed and to be reported with any use:
  * Rejection changes the sampled distribution -- the output
    over-represents rule-consistent regions relative to the raw model.
    The real holdout itself violates rules at ~0.2%, so an all-clean
    file is slightly cleaner than real data. Re-evaluate the output's
    fidelity and gate it before distribution.
  * A model with a high raw violation rate needs proportionally more
    sampling; --max-rounds bounds the cost.
"""

import argparse
import json
import os
import sys

import pandas as pd
import polars as pl

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case, harmonize_dtypes
from pipeline.common.model_io import load_generator
from pipeline.common.representation_audit import audit_representation, summarize as rep_summary
from pipeline.steps.coherence import rules as R
from pipeline.steps.generate import leakage
from pipeline.steps.generate.step import GenerateStep


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule-rejection sampling from a saved generator.")
    parser.add_argument("--model", required=True, help="Saved generator .pkl")
    parser.add_argument("--rows", type=int, required=True, help="Clean rows to produce.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--batch", type=int, default=None,
                        help="Rows sampled per round (default: 2x the target).")
    args = parser.parse_args()
    config = PipelineConfig()

    rules_path = os.path.join(config.step_dir("coherence"), "DT4H_Coherence_Rules.json")
    with open(rules_path) as f:
        ruleset = json.load(f)["rules"]
    print(f"Loaded {len(ruleset)} coherence rules from {rules_path}")

    train = pl.read_parquet(config.train_output_path).to_pandas()
    train_decoded, _ = GenerateStep._decode_numeric_missing(train.copy(), config)

    summary_path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
    constants = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            constants = json.load(f).get("constant_columns_held_out", {})

    synth = load_generator(args.model)
    print(f"Loaded generator: {getattr(synth, 'name', type(synth).__name__)}")

    batch_size = args.batch or max(args.rows * 2, 1000)
    kept_frames = []
    kept = sampled = 0
    for round_no in range(1, args.max_rounds + 1):
        batch = synth.sample(batch_size)
        sampled += len(batch)
        batch, _ = align_categorical_case(batch, train)
        if constants:
            pad = pd.DataFrame({c: [v] * len(batch) for c, v in constants.items()},
                               index=batch.index)
            batch = pd.concat([batch, pad], axis=1)
        batch = batch[[c for c in train.columns if c in batch.columns]].copy()

        # Leakage check in sentinel space, BEFORE decode -- same order as
        # the generate step, same reasoning.
        leak = leakage.check_exact_duplicates(batch, train)
        if leak.get("exact_duplicates_of_training_rows", 0) > 0:
            raise RuntimeError("Sampled batch reproduced a training row verbatim -- "
                               "refusing to continue.")

        batch, _ = GenerateStep._decode_numeric_missing(batch, config)
        bad = R.row_violation_mask(batch, ruleset)
        clean = batch.loc[~bad]
        kept_frames.append(clean)
        kept += len(clean)
        print(f"  round {round_no}: {len(clean)}/{len(batch)} rows clean "
              f"({bad.mean():.1%} rejected); total clean {min(kept, args.rows)}/{args.rows}")
        if kept >= args.rows:
            break

    out = pd.concat(kept_frames, ignore_index=True).head(args.rows)
    out, _ = harmonize_dtypes(out, train_decoded)
    if out.empty:
        print(f"🚫 No clean rows at all in {sampled} sampled -- this model's rows "
              f"essentially always violate the rule set; rejection sampling cannot "
              f"repair it (that is itself a finding). Nothing written.")
        return 1
    if len(out) < args.rows:
        print(f"⚠️  Only {len(out)} clean rows after {args.max_rounds} rounds "
              f"({sampled} sampled) -- the model's raw violation rate is high; "
              f"raise --max-rounds or --batch.")

    # Verify the output really is rule-clean and honest about the rest.
    final_summary = R.summarize_rule_results(R.evaluate_rules(out, ruleset))
    rep = audit_representation(out, train_decoded)
    leak = leakage.check_exact_duplicates(out, train_decoded)
    print(f"\nOutput violation rate: {final_summary['overall_violation_rate']} "
          f"(real holdout's own rate is the fair bar; an all-clean file is "
          f"slightly cleaner than real data -- report that)")
    print(f"Representation: {rep_summary(rep)}")
    print(leakage.summarize(leak))
    print(f"Rejection acceptance rate: {kept}/{sampled} = {kept / max(sampled, 1):.1%}")

    out.to_csv(args.out, index=False)
    print(f"✅ {len(out)} x {out.shape[1]} -> {args.out}")
    print(f"Next: python release_gate.py --file {args.out}")
    return 0 if (rep["clean"] and final_summary["overall_violation_rate"] in (0, 0.0)
                 and leak.get("exact_duplicates_of_training_rows", 1) == 0
                 and len(out) == args.rows) else 1


if __name__ == "__main__":
    sys.exit(main())

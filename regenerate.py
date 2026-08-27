"""
Generate more synthetic records from a previously fitted generator,
without retraining.

    python regenerate.py --model output/generate/models/ctgan.pkl \
                         --rows 10000 --out extra_synthetic.csv

Applies the same post-processing as the generate step, in the same order:
re-attach held-out constant columns, restore the real column order, run
the verbatim-leakage check IN SENTINEL SPACE, then decode the numeric
missingness sentinels back to null. Requires the generation summary,
the numeric encoding map and the preprocessed parquet produced by the
pipeline run that trained the model.

Two things this tool takes as seriously as the pipeline does:

  * SEEDING. It used to seed nothing, so anything sampling from a global
    RNG -- Private-PGM's sampler behind MST/AIM above all -- produced a
    different file every invocation, with nothing recorded to explain
    why. It now calls the generate step's own set_global_seeds() and
    writes a provenance sidecar (<out>.provenance.json) naming the seed,
    the model, the leakage result and the git revision.
  * LEAKAGE. A verbatim training record in the output is a published
    patient record. The output is now written to a QUARANTINED path
    (<out>.LEAKED.csv) and the process exits non-zero, so a leaky batch
    cannot be picked up by a script that only checked that the file
    exists.
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.steps.generate import leakage  # noqa: E402
from pipeline.steps.generate.step import GenerateStep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample from a saved fitted generator.")
    parser.add_argument("--model", required=True, help="Path to a saved generator .pkl (output/generate/models/<name>.pkl).")
    parser.add_argument("--rows", type=int, required=True, help="Number of synthetic rows to generate.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for this sampling run (default: the pipeline's "
                             "config seed). Seeds python/numpy/torch globals -- which "
                             "is what MST/AIM's sampler draws from -- and, when given "
                             "explicitly, also overrides the fitted model's own "
                             "sampling seed. Recorded in <out>.provenance.json.")
    args = parser.parse_args()

    config = PipelineConfig()

    import pandas as pd
    import polars as pl

    from pipeline.common.model_io import load_generator
    from pipeline.steps.generate.reproducibility import git_revision, set_global_seeds

    seed = config.seed if args.seed is None else args.seed
    seed_state = set_global_seeds(seed)
    print(f"Seeded every global RNG with {seed} "
          f"({'--seed' if args.seed is not None else 'config default'}).")

    synth = load_generator(args.model)
    print(f"Loaded generator: {getattr(synth, 'name', type(synth).__name__)} (params: {getattr(synth, 'params', {})})")
    if args.seed is not None and isinstance(getattr(synth, "params", None), dict):
        # Only on an explicit --seed: leaving it alone by default means a
        # plain re-run of this tool reproduces the model's own first
        # batch, which is what makes the committed CSVs checkable.
        synth.params["seed"] = seed

    summary_path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    constants = summary.get("constant_columns_held_out", {})

    # Leakage reference is the TRAIN split -- the rows the generator
    # actually saw (falls back to the full frame for pre-holdout models).
    ref_path = (config.train_output_path if os.path.exists(config.train_output_path)
                else config.preprocessed_output_path)
    real = pl.read_parquet(ref_path).to_pandas()

    print(f"Sampling {args.rows} rows...")
    synthetic = synth.sample(args.rows)

    from pipeline.common.alignment import align_categorical_case, report as align_report
    synthetic, respelled = align_categorical_case(synthetic, real)
    print(align_report(respelled))

    if constants:
        held_out = pd.DataFrame(
            {col: [value] * len(synthetic) for col, value in constants.items()},
            index=synthetic.index,
        )
        synthetic = pd.concat([synthetic, held_out], axis=1)
    synthetic = synthetic[[c for c in real.columns if c in synthetic.columns]].copy()

    leak = leakage.check_exact_duplicates(synthetic, real)
    print(leakage.summarize(leak))
    n_leaked = leak.get("exact_duplicates_of_training_rows", 0)

    synthetic, decoded = GenerateStep._decode_numeric_missing(synthetic, config)
    if decoded:
        print(f"Decoded sentinels back to null in {len(decoded)} column(s) "
              f"({sum(decoded.values())} cells).")

    # A leaky batch never lands on the path the caller asked for: it goes
    # to a name nobody distributes by accident, and the exit code says so.
    out_path = args.out
    if n_leaked:
        base = args.out[:-4] if args.out.lower().endswith(".csv") else args.out
        out_path = base + ".LEAKED.csv"

    synthetic.to_csv(out_path, index=False)

    sidecar = out_path + ".provenance.json"
    with open(sidecar, "w") as f:
        json.dump({
            "tool": "regenerate.py",
            "model": args.model,
            "rows_requested": args.rows,
            "rows_written": int(synthetic.shape[0]),
            "seed": seed,
            "seed_explicit": args.seed is not None,
            "seed_state": seed_state,
            "leakage": leak,
            "quarantined": bool(n_leaked),
            "git": git_revision(),
            "output_path": out_path,
        }, f, indent=2, default=str)

    if n_leaked:
        print(f"\n{'!' * 70}\n🚨 {n_leaked} synthetic row(s) exactly reproduce a training "
              f"record.\n   QUARANTINED -> {out_path}\n   This file contains real patient "
              f"records. Do not distribute it, do not rename it into place; delete it or "
              f"investigate the generator.\n{'!' * 70}")
        print(f"Provenance sidecar -> {sidecar}")
        return 1

    print(f"✅ {synthetic.shape[0]} x {synthetic.shape[1]} -> {out_path}")
    print(f"Provenance sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

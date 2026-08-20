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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a saved fitted generator.")
    parser.add_argument("--model", required=True, help="Path to a saved generator .pkl (output/generate/models/<name>.pkl).")
    parser.add_argument("--rows", type=int, required=True, help="Number of synthetic rows to generate.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    config = PipelineConfig()

    import pandas as pd
    import polars as pl

    with open(args.model, "rb") as f:
        synth = pickle.load(f)
    print(f"Loaded generator: {getattr(synth, 'name', type(synth).__name__)} (params: {getattr(synth, 'params', {})})")

    summary_path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    constants = summary.get("constant_columns_held_out", {})

    real = pl.read_parquet(config.preprocessed_output_path).to_pandas()

    print(f"Sampling {args.rows} rows...")
    synthetic = synth.sample(args.rows)

    if constants:
        held_out = pd.DataFrame(
            {col: [value] * len(synthetic) for col, value in constants.items()},
            index=synthetic.index,
        )
        synthetic = pd.concat([synthetic, held_out], axis=1)
    synthetic = synthetic[[c for c in real.columns if c in synthetic.columns]].copy()

    leak = leakage.check_exact_duplicates(synthetic, real)
    print(leakage.summarize(leak))
    if leak.get("exact_duplicates_of_training_rows", 0) > 0:
        print("🚨 Refusing silently: output written anyway, but DO NOT distribute it as-is.")

    synthetic, decoded = GenerateStep._decode_numeric_missing(synthetic, config)
    if decoded:
        print(f"Decoded sentinels back to null in {len(decoded)} column(s) "
              f"({sum(decoded.values())} cells).")

    synthetic.to_csv(args.out, index=False)
    print(f"✅ {synthetic.shape[0]} x {synthetic.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()

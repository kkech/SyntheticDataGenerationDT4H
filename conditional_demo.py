"""
Conditional sampling demo: generate synthetic patients satisfying
user-specified conditions from a SAVED generator, without retraining --
e.g. 500 female patients aged over 80.

    python conditional_demo.py --model output/generate/models/tvae_seed0.pkl \
        --rows 500 --condition patient_demographics_gender=female \
        --out conditional_sample.csv

Only the SDV-backed models (gaussian_copula, tvae, ctgan) support exact
conditional sampling; for the smartnoise DP models this script falls
back to rejection sampling (generate, filter, repeat), which also works
for numeric conditions like `patient_demographics_age>80`.

Output goes through the same post-processing chain as the pipeline:
constants re-attached, column order restored, verbatim-leakage check in
sentinel space, sentinel decode. The output is NOT auto-committed --
run release_gate.py on it before distributing.
"""

import argparse
import json
import os
import pickle
import re

import pandas as pd
import polars as pl

from pipeline.config import PipelineConfig
from pipeline.steps.generate import leakage
from pipeline.steps.generate.step import GenerateStep

MAX_REJECTION_ROUNDS = 40


def parse_condition(text: str):
    m = re.match(r"^([\w.]+)\s*(=|==|>|<|>=|<=)\s*(.+)$", text)
    if not m:
        raise ValueError(f"Cannot parse condition: {text}")
    col, op, val = m.group(1), m.group(2).replace("==", "="), m.group(3)
    return col, op, val


def matches(df: pd.DataFrame, conditions) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, op, val in conditions:
        if col not in df.columns:
            raise ValueError(f"Column {col} not in generated output.")
        if op == "=":
            mask &= df[col].astype(str).str.lower() == str(val).lower()
        else:
            num = pd.to_numeric(df[col], errors="coerce")
            v = float(val)
            mask &= {"": None, ">": num > v, "<": num < v,
                     ">=": num >= v, "<=": num <= v}[op]
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional synthetic sampling demo.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--condition", action="append", required=True,
                        help="col=value or col>value (repeatable, ANDed)")
    parser.add_argument("--out", default="conditional_sample.csv")
    args = parser.parse_args()
    config = PipelineConfig()

    conditions = [parse_condition(c) for c in args.condition]
    with open(args.model, "rb") as f:
        synth = pickle.load(f)
    print(f"Loaded generator {getattr(synth, 'name', '?')} from {args.model}")
    print(f"Conditions: {conditions}")

    sdv_model = getattr(synth, "_model", None)
    exact = [c for c in conditions if c[1] == "="]
    collected = []
    if sdv_model is not None and hasattr(sdv_model, "sample_from_conditions") and \
            len(exact) == len(conditions):
        from sdv.sampling import Condition

        cond = Condition({col: val for col, _, val in exact}, num_rows=args.rows)
        print("Exact conditional sampling via SDV...")
        collected.append(sdv_model.sample_from_conditions([cond]))
    else:
        print("Rejection sampling (works for any model and numeric conditions)...")
        need = args.rows
        for round_no in range(MAX_REJECTION_ROUNDS):
            batch = synth.sample(max(need * 4, 2000))
            keep = batch[matches(batch, conditions)]
            if len(keep):
                collected.append(keep)
                need -= len(keep)
            print(f"  round {round_no + 1}: kept {len(keep)}, still need {max(need, 0)}")
            if need <= 0:
                break
        if need > 0:
            print(f"⚠️  Only found {args.rows - need} matching rows in "
                  f"{MAX_REJECTION_ROUNDS} rounds -- the condition may be rare "
                  f"in the modelled population.")

    synthetic = pd.concat(collected, ignore_index=True).head(args.rows)

    summary_path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
    with open(summary_path) as f:
        constants = json.load(f).get("constant_columns_held_out", {})
    if constants:
        pad = pd.DataFrame({c: [v] * len(synthetic) for c, v in constants.items()},
                           index=synthetic.index)
        synthetic = pd.concat([synthetic, pad], axis=1)

    ref_path = (config.train_output_path if os.path.exists(config.train_output_path)
                else config.preprocessed_output_path)
    real = pl.read_parquet(ref_path).to_pandas()
    synthetic = synthetic[[c for c in real.columns if c in synthetic.columns]].copy()

    print(leakage.summarize(leakage.check_exact_duplicates(synthetic, real)))
    synthetic, decoded = GenerateStep._decode_numeric_missing(synthetic, config)
    if decoded:
        print(f"Decoded sentinels to null in {len(decoded)} column(s).")

    synthetic.to_csv(args.out, index=False)
    print(f"✅ {len(synthetic)} conditional rows -> {args.out} "
          f"(run release_gate.py before distributing)")


if __name__ == "__main__":
    main()

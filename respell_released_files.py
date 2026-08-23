"""
One-time repair of released synthetic CSVs generated before the
representation fix: rewrite each file with its categorical spellings
aligned to the real schema (True -> true, 2017.0 -> 2017, ...), so the
distributed files are canonical on disk instead of relying on every
consumer to realign at read time.

    python respell_released_files.py            # rewrites in place
    python respell_released_files.py --dry-run  # report only

No values change semantically -- only their spelling. Numeric columns
are untouched, unseen categories are preserved, and every rewritten
file is re-audited (zero cross-representation mismatches) before it
replaces the original. Files generated after the fix are already clean
and are left untouched (reported as such).
"""

import argparse
import glob
import os

import pandas as pd
import polars as pl

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case
from pipeline.common.representation_audit import audit_representation
from pipeline.steps.generate.step import GenerateStep


def main() -> int:
    parser = argparse.ArgumentParser(description="Align released CSV spellings to the real schema.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    args = parser.parse_args()
    config = PipelineConfig()

    train = pl.read_parquet(config.train_output_path).to_pandas()
    # Released files are in DECODED space; align against the decoded frame.
    train_decoded, _ = GenerateStep._decode_numeric_missing(train.copy(), config)

    files = sorted(glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv")))
    if not files:
        print("No DT4H_Synthetic_*.csv files found -- nothing to do.")
        return 1

    rewritten = clean = failed = 0
    for path in files:
        # round_trip: the default CSV float parser is approximate (last-ULP
        # wobble), which would make byte-idempotence impossible; exact
        # parsing preserves every numeric literal as originally written.
        df = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        aligned, respelled = align_categorical_case(df, train_decoded)
        name = os.path.basename(path)
        # `respelled` counts cells whose spelling differed from the frame as
        # PARSED, which for boolean columns is inevitable (pandas bool-parses
        # any true/false spelling). What matters for the file on disk is
        # whether re-serializing the aligned frame changes its bytes.
        new_text = aligned.to_csv(index=False)
        with open(path) as f:
            old_text = f.read()
        if new_text == old_text:
            print(f"  ✅ {name}: already canonical on disk")
            clean += 1
            continue
        rep = audit_representation(aligned, train_decoded)
        if not rep["clean"] or rep["categorical_nulls"]:
            print(f"  🚨 {name}: file would change but the result does not audit "
                  f"clean -- NOT rewritten. Investigate first.")
            failed += 1
            continue
        cells = sum(respelled.values())
        if args.dry_run:
            print(f"  📝 {name}: would re-spell up to {cells} cell(s) in "
                  f"{len(respelled)} column(s)")
        else:
            with open(path, "w") as f:
                f.write(new_text)
            print(f"  ✏️  {name}: re-spelled up to {cells} cell(s) in "
                  f"{len(respelled)} column(s)")
        rewritten += 1

    verb = "would be rewritten" if args.dry_run else "rewritten"
    print(f"\n{rewritten} file(s) {verb}, {clean} already canonical, {failed} refused.")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

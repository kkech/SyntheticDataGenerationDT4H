import glob
import json
import os
import shutil

import polars as pl

from exploreData import analyze_column, write_markdown

# --- CONFIGURATION ---
INPUT_FOLDER = "/mnt/data/transfer-2026-08-12-12-05-35-m.j.boonstra-3/"

# Written inside the repo itself (next to this script) so the output can be
# `git add`-ed directly, without copying files in from /mnt/data by hand.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(REPO_ROOT, "for_repo")

N_SAMPLE_ROWS = 20
SAMPLE_SEED = 0

# Full resolved dataset, saved LOCALLY ONLY (never copied into OUTPUT_FOLDER
# / the repo). preprocessUC1FeatureSet.py's INPUT_PATH points here by
# default, so the actual UC1 preprocessing runs against the real full data.
LOCAL_FULL_OUTPUT_PATH = "/mnt/data/DT4Hnew/UC1_Resolved_Full.parquet"


def inspect_folder(folder: str) -> None:
    print(f"--- 📂 INSPECTING {folder} ---")
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        kind = "dir" if os.path.isdir(path) else "file"
        size = "" if kind == "dir" else f", {os.path.getsize(path):,} bytes"
        print(f"  [{kind}] {entry}{size}")


def load_metadata(folder: str, output_folder: str) -> None:
    """
    Copies whatever is at <folder>/metadata (file or directory) into the
    output folder verbatim. This is schema/column definition info, not
    patient-level data, so it's safe to carry through unmodified.
    """
    meta_path = os.path.join(folder, "metadata")
    if not os.path.exists(meta_path):
        print("⚠️  No 'metadata' file/folder found next to the data.")
        return

    dest = os.path.join(output_folder, "metadata")
    if os.path.isdir(meta_path):
        shutil.copytree(meta_path, dest, dirs_exist_ok=True)
        print(f"Copied metadata folder ({len(os.listdir(meta_path))} entries) -> {dest}")
    else:
        shutil.copy2(meta_path, dest)
        print(f"Copied metadata file -> {dest}")


def load_part_files(folder: str) -> pl.DataFrame | None:
    """
    part-NNNNN-<uuid>-c000.snappy.parquet is the standard naming Spark uses
    for row-wise partitions of a single logical table -- these all share
    one schema and should be concatenated, not joined.
    """
    part_files = sorted(glob.glob(os.path.join(folder, "part-*.parquet")))
    if not part_files:
        return None

    print(f"\nFound {len(part_files)} Spark-style part file(s):")
    frames = []
    for f in part_files:
        df = pl.read_parquet(f)
        print(f"  {os.path.basename(f)}: {df.height} rows x {df.width} cols")
        frames.append(df)

    combined = pl.concat(frames, how="vertical_relaxed")
    print(f"Concatenated parts: {combined.height} rows x {combined.width} cols")
    return combined


def load_data_parquet(folder: str) -> pl.DataFrame | None:
    path = os.path.join(folder, "data.parquet")
    if not os.path.exists(path):
        return None
    df = pl.read_parquet(path)
    print(f"\ndata.parquet: {df.height} rows x {df.width} cols")
    return df


def resolve_primary_dataset(df_data: pl.DataFrame | None, df_parts: pl.DataFrame | None) -> pl.DataFrame:
    """
    We don't know upfront whether data.parquet is a separate/redundant
    export or the already-merged version of the part files, so this makes
    the decision explicit and loud rather than silently picking one.
    """
    if df_data is None and df_parts is None:
        raise ValueError("Neither data.parquet nor part-*.parquet files were found.")
    if df_data is None:
        print("\nUsing concatenated part files as the dataset (no data.parquet found).")
        return df_parts
    if df_parts is None:
        print("\nUsing data.parquet as the dataset (no part files found).")
        return df_data

    same_cols = set(df_data.columns) == set(df_parts.columns)
    print(f"\ndata.parquet columns {'MATCH' if same_cols else 'DIFFER FROM'} concatenated part-file columns.")
    if same_cols:
        print("Treating data.parquet as the canonical merged file; ignoring part files to avoid double-counting.")
        print("⚠️  If that's wrong (e.g. data.parquet is a different subset/sample), edit resolve_primary_dataset().")
        return df_data

    print("⚠️  Schemas differ -- defaulting to the concatenated part files, but this needs manual review:")
    print(f"   data.parquet-only columns: {sorted(set(df_data.columns) - set(df_parts.columns))[:10]}")
    print(f"   parts-only columns: {sorted(set(df_parts.columns) - set(df_data.columns))[:10]}")
    return df_parts


def analyze_full_dataset(df: pl.DataFrame, output_folder: str) -> None:
    """Reuses the privacy-conscious per-column profiler from exploreData.py."""
    total_rows = df.height
    print(f"\n--- 📊 PROFILING full dataset: {total_rows} rows, {df.width} columns ---")

    analysis = {col: analyze_column(df, col) for col in df.columns}

    json_path = os.path.join(output_folder, "DT4H_Column_Analysis.json")
    with open(json_path, "w") as f:
        json.dump(
            {"total_rows": total_rows, "total_columns": df.width, "columns": analysis},
            f,
            indent=2,
            default=str,
        )
    print(f"✅ JSON analysis saved to: {json_path}")

    md_path = os.path.join(output_folder, "DT4H_Column_Analysis.md")
    write_markdown(analysis, total_rows, md_path)
    print(f"✅ Markdown summary saved to: {md_path}")


def write_sample(df: pl.DataFrame, output_folder: str) -> None:
    n = min(N_SAMPLE_ROWS, df.height)
    sample = df.sample(n=n, seed=SAMPLE_SEED)
    path = os.path.join(output_folder, "DT4H_Sample20.parquet")
    sample.write_parquet(path)
    print(f"✅ {n}-row sample saved to: {path}")


def prepare_test_data():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    inspect_folder(INPUT_FOLDER)

    df_data = load_data_parquet(INPUT_FOLDER)
    df_parts = load_part_files(INPUT_FOLDER)
    df_full = resolve_primary_dataset(df_data, df_parts)

    load_metadata(INPUT_FOLDER, OUTPUT_FOLDER)
    analyze_full_dataset(df_full, OUTPUT_FOLDER)
    write_sample(df_full, OUTPUT_FOLDER)

    os.makedirs(os.path.dirname(LOCAL_FULL_OUTPUT_PATH), exist_ok=True)
    df_full.write_parquet(LOCAL_FULL_OUTPUT_PATH)
    print(f"✅ Full resolved dataset saved LOCALLY (not for git) to: {LOCAL_FULL_OUTPUT_PATH}")

    print(f"\n🎉 DONE. Review the contents of {OUTPUT_FOLDER} then push them to the repo:")
    print("   - metadata                          (schema info)")
    print("   - DT4H_Column_Analysis.json / .md    (full-dataset statistics)")
    print(f"   - DT4H_Sample20.parquet              ({N_SAMPLE_ROWS}-row sample for code testing)")
    print(f"\n(Do NOT push {LOCAL_FULL_OUTPUT_PATH} -- it's the full dataset, stays local. "
          f"preprocessUC1FeatureSet.py reads it from there by default.)")


if __name__ == "__main__":
    prepare_test_data()

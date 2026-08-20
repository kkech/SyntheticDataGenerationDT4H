"""
Step: profile_data

Reads the full local dataset (written by load_data) and produces a small,
privacy-safe package for committing to git: full-dataset column statistics
(JSON + Markdown), a random row sample, and a copy of the transfer's
metadata.json schema. No other row-level data leaves the machine.
"""

import json
import os
import shutil

import polars as pl

from pipeline.common.profiling import analyze_column, write_markdown
from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep

METADATA_CANDIDATES = ["metadata.json", "metadata", "metadata.parquet"]


class ProfileDataStep(PipelineStep):
    name = "profile_data"

    def run(self, config: PipelineConfig) -> None:
        if not os.path.exists(config.local_full_dataset_path):
            raise FileNotFoundError(
                f"{config.local_full_dataset_path} not found -- run the load_data step first."
            )
        df = pl.read_parquet(config.local_full_dataset_path)
        os.makedirs(config.for_repo_dir, exist_ok=True)

        self._copy_metadata(config)
        self._write_analysis(df, config)
        self._write_sample(df, config)

    def _copy_metadata(self, config: PipelineConfig) -> None:
        src = next(
            (
                os.path.join(config.transfer_folder, c)
                for c in METADATA_CANDIDATES
                if os.path.exists(os.path.join(config.transfer_folder, c))
            ),
            None,
        )
        if src is None:
            print(f"⚠️  No metadata file/folder found in {config.transfer_folder} "
                  f"(checked: {METADATA_CANDIDATES}).")
            return

        dest = os.path.join(config.for_repo_dir, os.path.basename(src))
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"Copied metadata folder ({len(os.listdir(src))} entries) -> {dest}")
        else:
            shutil.copy2(src, dest)
            print(f"Copied metadata file -> {dest}")

    def _write_analysis(self, df: pl.DataFrame, config: PipelineConfig) -> None:
        print(f"Profiling {df.height} rows x {df.width} columns...")
        analysis = {col: analyze_column(df, col) for col in df.columns}

        json_path = os.path.join(config.for_repo_dir, "DT4H_Column_Analysis.json")
        with open(json_path, "w") as f:
            json.dump(
                {"total_rows": df.height, "total_columns": df.width, "columns": analysis},
                f,
                indent=2,
                default=str,
            )
        print(f"Saved column analysis (JSON) -> {json_path}")

        md_path = os.path.join(config.for_repo_dir, "DT4H_Column_Analysis.md")
        write_markdown(analysis, df.height, md_path)
        print(f"Saved column analysis (Markdown) -> {md_path}")

    def _write_sample(self, df: pl.DataFrame, config: PipelineConfig) -> None:
        n = min(config.sample_rows, df.height)
        sample = df.sample(n=n, seed=config.sample_seed)
        path = os.path.join(config.for_repo_dir, "DT4H_Sample20.parquet")
        sample.write_parquet(path)
        print(f"Saved {n}-row sample -> {path}")

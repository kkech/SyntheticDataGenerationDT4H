"""
Step: profile_data

Reads the full local dataset (written by load_data) and produces a small,
privacy-safe package into output/profile_data/: full-dataset column
statistics (JSON + Markdown), a random row sample, and a copy of the
transfer's metadata.json schema. No other row-level data leaves the machine.
"""

import json
import os

import polars as pl

from pipeline.common.artifacts import copy_metadata, write_row_sample
from pipeline.common.profiling import analyze_column, write_markdown
from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep


class ProfileDataStep(PipelineStep):
    name = "profile_data"

    def run(self, config: PipelineConfig) -> None:
        if not os.path.exists(config.local_full_dataset_path):
            raise FileNotFoundError(
                f"{config.local_full_dataset_path} not found -- run the load_data step first."
            )
        df = pl.read_parquet(config.local_full_dataset_path)
        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        copy_metadata(config.transfer_folder, out_dir)
        self._write_analysis(df, out_dir)
        write_row_sample(
            df, os.path.join(out_dir, "DT4H_Sample20.parquet"), config.sample_rows, config.sample_seed
        )

    def _write_analysis(self, df: pl.DataFrame, out_dir: str) -> None:
        print(f"Profiling {df.height} rows x {df.width} columns...")
        analysis = {col: analyze_column(df, col) for col in df.columns}

        json_path = os.path.join(out_dir, "DT4H_Column_Analysis.json")
        with open(json_path, "w") as f:
            json.dump(
                {"total_rows": df.height, "total_columns": df.width, "columns": analysis},
                f,
                indent=2,
                default=str,
            )
        print(f"Saved column analysis (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Column_Analysis.md")
        write_markdown(analysis, df.height, md_path)
        print(f"Saved column analysis (Markdown) -> {md_path}")

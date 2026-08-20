"""
Step: profile_preprocessed_data

Same privacy-safe profiler as profile_data, but run against the OUTPUT of
the preprocess step instead of the raw loaded data. Lets before/after
preprocessing statistics be compared directly (e.g. confirm imputation
left no nulls, combined feature columns look sane, column count dropped
as expected). Also copies metadata.json again so this step's output
(output/profile_preprocessed_data/) is self-contained even if run on its
own (e.g. via --only).
"""

import json
import os

import polars as pl

from pipeline.common.artifacts import copy_metadata, write_row_sample
from pipeline.common.profiling import analyze_column, write_markdown
from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep


class ProfilePreprocessedDataStep(PipelineStep):
    name = "profile_preprocessed_data"

    def run(self, config: PipelineConfig) -> None:
        if not os.path.exists(config.preprocessed_output_path):
            raise FileNotFoundError(
                f"{config.preprocessed_output_path} not found -- run the preprocess step first."
            )
        df = pl.read_parquet(config.preprocessed_output_path)
        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        copy_metadata(config.transfer_folder, out_dir)
        self._write_analysis(df, out_dir)
        write_row_sample(
            df,
            os.path.join(out_dir, "DT4H_Preprocessed_Sample20.parquet"),
            config.sample_rows,
            config.sample_seed,
        )

    def _write_analysis(self, df: pl.DataFrame, out_dir: str) -> None:
        print(f"Profiling {df.height} rows x {df.width} columns (preprocessed)...")
        analysis = {col: analyze_column(df, col) for col in df.columns}

        json_path = os.path.join(out_dir, "DT4H_Preprocessed_Column_Analysis.json")
        with open(json_path, "w") as f:
            json.dump(
                {"total_rows": df.height, "total_columns": df.width, "columns": analysis},
                f,
                indent=2,
                default=str,
            )
        print(f"Saved preprocessed column analysis (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Preprocessed_Column_Analysis.md")
        write_markdown(analysis, df.height, md_path)
        print(f"Saved preprocessed column analysis (Markdown) -> {md_path}")

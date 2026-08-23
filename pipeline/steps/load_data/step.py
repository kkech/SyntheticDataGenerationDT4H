"""
Step: load_data

Concatenates the transfer folder's Spark part-*.parquet files (row-wise
partitions of a single logical table, all sharing one schema) into the
full dataset and saves it locally. This is real patient-level data and
must never be committed to git (excluded via .gitignore).
"""

import glob
import os

import polars as pl

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep


class LoadDataStep(PipelineStep):
    name = "load_data"

    def run(self, config: PipelineConfig) -> None:
        part_files = sorted(glob.glob(os.path.join(config.transfer_folder, "part-*.parquet")))
        if not part_files:
            raise FileNotFoundError(f"No part-*.parquet files found in {config.transfer_folder}.")

        print(f"Found {len(part_files)} part file(s):")
        frames = []
        for f in part_files:
            df = pl.read_parquet(f)
            print(f"  {os.path.basename(f)}: {df.height} rows x {df.width} cols")
            frames.append(df)

        combined = pl.concat(frames, how="vertical_relaxed")
        print(f"Concatenated: {combined.height} rows x {combined.width} cols")

        os.makedirs(os.path.dirname(config.local_full_dataset_path), exist_ok=True)
        combined.write_parquet(config.local_full_dataset_path)
        print(f"Saved full dataset (local only, not for git) -> {config.local_full_dataset_path}")

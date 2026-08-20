"""
Central configuration for the UC1 data pipeline: every path each step
reads from or writes to, in one place. Steps take a PipelineConfig
instance rather than hardcoding paths, so tests can point them at a small
sample instead of the real full dataset.
"""

import os
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    # Where the raw transfer (Spark part-*.parquet files) lives.
    transfer_folder: str = "/mnt/data/transfer-2026-08-12-12-05-35-m.j.boonstra-3/"

    # Single root for every step's output, organized as
    # output_dir/<step_name>/. Lives inside the repo so everything in it
    # can be committed directly -- except the two full-data parquet files
    # (output/load_data/UC1_Resolved_Full.parquet and
    # output/preprocess/UC1_Preprocessed.parquet), which are real
    # patient-level data and are excluded via .gitignore.
    output_dir: str = os.path.join(REPO_ROOT, "output")

    sample_rows: int = 20
    sample_seed: int = 0

    metadata_path: str = None  # defaults to <output_dir>/profile_data/metadata.json
    apply_dummy_imputation: bool = True

    # Defaults to <output_dir>/load_data/UC1_Resolved_Full.parquet (gitignored)
    local_full_dataset_path: str = None
    # Defaults to <output_dir>/preprocess/UC1_Preprocessed.parquet (gitignored)
    preprocessed_output_path: str = None

    # Tracks which steps have completed. Not patient data -- safe to
    # commit if you want step-completion visible in git history, or leave
    # local; your choice.
    status_path: str = os.path.join(REPO_ROOT, "pipeline_status.json")

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = os.path.join(self.output_dir, "profile_data", "metadata.json")
        if self.local_full_dataset_path is None:
            self.local_full_dataset_path = os.path.join(self.output_dir, "load_data", "UC1_Resolved_Full.parquet")
        if self.preprocessed_output_path is None:
            self.preprocessed_output_path = os.path.join(self.output_dir, "preprocess", "UC1_Preprocessed.parquet")

    def step_dir(self, step_name: str) -> str:
        """The dedicated output subfolder for a given step: output_dir/<step_name>/."""
        return os.path.join(self.output_dir, step_name)

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

    # Root for step outputs that must stay local -- this is real
    # patient-level data and must never be committed to git. Every step
    # that touches full patient data gets its own subfolder here:
    # output_dir/load_data/, output_dir/preprocess/, etc.
    output_dir: str = "/mnt/data/DT4Hnew/output"

    # profile_data's output is the one exception: it's a small,
    # privacy-safe package (column stats, a row sample, a metadata copy)
    # specifically meant for committing to git, so it's this step's
    # dedicated output folder, living inside the repo instead of output_dir.
    for_repo_dir: str = os.path.join(REPO_ROOT, "for_repo")
    sample_rows: int = 20
    sample_seed: int = 0

    metadata_path: str = None  # defaults to <for_repo_dir>/metadata.json
    apply_dummy_imputation: bool = True

    # Defaults to <output_dir>/load_data/UC1_Resolved_Full.parquet
    local_full_dataset_path: str = None
    # Defaults to <output_dir>/preprocess/UC1_Preprocessed.parquet
    preprocessed_output_path: str = None

    # Tracks which steps have completed. Not patient data -- safe to
    # commit if you want step-completion visible in git history, or leave
    # local; your choice.
    status_path: str = os.path.join(REPO_ROOT, "pipeline_status.json")

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = os.path.join(self.for_repo_dir, "metadata.json")
        if self.local_full_dataset_path is None:
            self.local_full_dataset_path = os.path.join(self.output_dir, "load_data", "UC1_Resolved_Full.parquet")
        if self.preprocessed_output_path is None:
            self.preprocessed_output_path = os.path.join(self.output_dir, "preprocess", "UC1_Preprocessed.parquet")

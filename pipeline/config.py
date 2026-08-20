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

    # --- generate step ---
    # Which synthesizers to run, by registry name. Non-DP: ctgan, tvae,
    # gaussian_copula. DP: aim, mst, patectgan, dpctgan.
    #
    # Ordered cheapest-first, so a misconfiguration surfaces in seconds
    # rather than after a ~20-minute CTGAN run. Each model's output and
    # the run summary are written as it finishes, so partial results
    # survive an interrupted or failed later model.
    #
    #   gaussian_copula  seconds   no training, statistical baseline
    #   tvae             minutes   usually stronger than CTGAN
    #   ctgan            ~20 min   the long-standing baseline
    #   mst              varies    DP, cheaper than AIM
    #   aim              varies    DP, best utility but heaviest
    #
    # AIM is preferred over the project's original dpctgan because
    # utility benchmarks consistently favour marginal-based methods over
    # DP-GANs on tabular data. It is listed last because AIM builds on
    # Private-PGM, which is documented to struggle as column count grows,
    # and this dataset is ~329 columns wide: if it exhausts memory, set
    # max_columns to trial a subset, or stop at mst.
    synthesizers: tuple = ("gaussian_copula", "tvae", "ctgan", "mst", "aim")
    synthesizer_params: dict = None  # per-synthesizer overrides, keyed by name

    # None = generate as many rows as the real dataset. Matching the real
    # row count is the usual convention for a released synthetic twin;
    # set an explicit number to over- or under-sample deliberately.
    n_synthetic_rows: int = None
    epochs: int = 500
    # Sized for a 16 GB T4 with the whole card free. SDV requires this to
    # be divisible by 10 (pac=10). Drop it (240, 120, 60) if training hits
    # CUDA OOM -- ~250 categorical columns make the conditional vector
    # wide, so memory scales with column count as well as batch size.
    batch_size: int = 500
    epsilon: float = 15.0

    # Seeds every RNG the synthesizers use, and is recorded in the run
    # provenance. Required for a reproducible published dataset.
    seed: int = 0

    # Constant columns carry no signal: they waste model capacity and, for
    # DP synthesizers, privacy budget. Held out during training and
    # re-attached verbatim afterwards, so the output schema is unchanged.
    drop_constant_columns: bool = True
    # Cap training width, for trialling Private-PGM-based synthesizers
    # (aim/mst) before a full-width run. None = use every column.
    max_columns: int = None

    # Defaults to <output_dir>/load_data/UC1_Resolved_Full.parquet (gitignored)
    local_full_dataset_path: str = None
    # Defaults to <output_dir>/preprocess/UC1_Preprocessed.parquet (gitignored)
    preprocessed_output_path: str = None

    # Tracks which steps have completed. Not patient data -- safe to
    # commit if you want step-completion visible in git history, or leave
    # local; your choice.
    status_path: str = os.path.join(REPO_ROOT, "pipeline_status.json")

    def __post_init__(self) -> None:
        if self.synthesizer_params is None:
            self.synthesizer_params = {}
        if self.metadata_path is None:
            self.metadata_path = os.path.join(self.output_dir, "profile_data", "metadata.json")
        if self.local_full_dataset_path is None:
            self.local_full_dataset_path = os.path.join(self.output_dir, "load_data", "UC1_Resolved_Full.parquet")
        if self.preprocessed_output_path is None:
            self.preprocessed_output_path = os.path.join(self.output_dir, "preprocess", "UC1_Preprocessed.parquet")

    def step_dir(self, step_name: str) -> str:
        """The dedicated output subfolder for a given step: output_dir/<step_name>/."""
        return os.path.join(self.output_dir, step_name)

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

    # --- holdout split ---
    # Fraction of preprocessed rows held out BEFORE generation. The
    # generators never see these rows, which is what makes the
    # evaluation honest: TSTR tests on them, the privacy step uses their
    # distance-to-training distribution as the memorization null, and
    # the evaluate step uses train-vs-holdout distances as the
    # sampling-noise floor that says what "as close as real data gets"
    # looks like at this sample size. Split is seeded and recorded in a
    # committed manifest (row indices only -- no patient data).
    holdout_fraction: float = 0.25

    # --- generate step: the run plan ---
    # The generate step executes a PLAN of runs, each a (synthesizer,
    # epsilon, seed, column set) combination, built by run_plan() below:
    #
    #   * non-DP models (gaussian_copula, tvae, ctgan): one run per seed
    #     in variance_seeds, so every headline number carries a mean and
    #     a standard deviation instead of a single lucky draw;
    #   * DP models (mst, dpctgan): a full epsilon sweep at the first
    #     seed -- the privacy-utility trade-off curve -- plus the
    #     remaining variance seeds at the headline epsilon;
    #   * aim: the epsilon sweep on the top `aim_max_columns` most
    #     outcome-relevant columns (Private-PGM cannot handle the full
    #     width -- it timed out at 6h on 211 columns), with its own
    #     shorter timeout so six runs cannot eat 36 hours.
    #
    # Ordered cheapest/most-reliable first, so a late failure costs only
    # the tail of the run. Set run_plan explicitly (tuple of dicts with
    # keys: synthesizer, seed, epsilon, columns, timeout_seconds) to
    # override the generated plan entirely.
    non_dp_synthesizers: tuple = ("gaussian_copula", "tvae", "ctgan")
    dp_synthesizers: tuple = ("dpctgan", "aim", "mst")
    dp_epsilons: tuple = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
    headline_epsilon: float = 15.0
    variance_seeds: tuple = (0, 1, 2)
    aim_max_columns: int = 50
    aim_timeout_seconds: int = 7200
    run_plan: tuple = None  # explicit override; None = build from the fields above
    synthesizer_params: dict = None  # per-synthesizer overrides, keyed by name

    # None = generate as many rows as the TRAINING split. Matching the
    # training row count keeps TSTR comparable (baseline and synthetic
    # classifiers see equally sized training sets); set an explicit
    # number to over- or under-sample deliberately.
    n_synthetic_rows: int = None
    epochs: int = 500
    # Sized for a 16 GB T4 with the whole card free. SDV requires this to
    # be divisible by 10 (pac=10). Drop it (240, 120, 60) if training hits
    # CUDA OOM -- ~250 categorical columns make the conditional vector
    # wide, so memory scales with column count as well as batch size.
    batch_size: int = 500
    # Fallback epsilon for a run spec without one. DP preprocessing
    # spends NOTHING: per-column domains are passed as public bounds
    # (they are released in the committed sentinel encoding map), so the
    # entire budget goes to synthesis at every epsilon in the sweep.
    epsilon: float = 15.0

    # Seeds every RNG the synthesizers use, and is recorded in the run
    # provenance. Required for a reproducible published dataset.
    seed: int = 0

    # Hard wall-clock limit per synthesizer (fit + sample), in seconds.
    # Added after AIM hung indefinitely at full column width (the
    # documented Private-PGM scaling failure mode): a stuck model now
    # fails cleanly with a TimeoutError in the summary and the remaining
    # synthesizers and the evaluate step still run, instead of the whole
    # pipeline sitting silent until someone kills it. Uses SIGALRM, so it
    # applies on Unix main-thread runs (i.e. normal `python main.py`).
    # None disables it.
    #
    # Calibrated against measured full-data runs, not guesses: CTGAN at
    # 500 epochs took ~14 min and MST -- CPU-bound and slow but
    # legitimately converging -- took 2.7 HOURS (9883s) to succeed. A
    # 1-hour limit would have killed that success, so the default is six
    # hours: far above every observed legitimate runtime, while still
    # bounding a truly wedged fit.
    synthesizer_timeout_seconds: int = 21600

    # --- utility step (TSTR) ---
    # None = auto-select up to utility_max_targets BOOLEAN outcome
    # variables from the feature-set metadata (best class balance first).
    # Set an explicit tuple of column names to override.
    utility_targets: tuple = None
    utility_max_targets: int = 5

    # Constant columns carry no signal: they waste model capacity and, for
    # DP synthesizers, privacy budget. Held out during training and
    # re-attached verbatim afterwards, so the output schema is unchanged.
    drop_constant_columns: bool = True

    # Defaults to <output_dir>/load_data/UC1_Resolved_Full.parquet (gitignored)
    local_full_dataset_path: str = None
    # Defaults to <output_dir>/preprocess/UC1_Preprocessed.parquet (gitignored)
    preprocessed_output_path: str = None
    # The 75/25 split of the preprocessed frame (both gitignored):
    # generators train ONLY on the train file; the holdout file is the
    # unseen-data reference for TSTR, privacy and the noise floor.
    train_output_path: str = None
    holdout_output_path: str = None

    # Tracks which steps have completed. Not patient data -- safe to
    # commit if you want step-completion visible in git history, or leave
    # local; your choice.
    status_path: str = os.path.join(REPO_ROOT, "pipeline_status.json")

    def __post_init__(self) -> None:
        if self.synthesizer_params is None:
            self.synthesizer_params = {}
        # DP-CTGAN trains with opacus per-sample gradients, which are far
        # more memory-hungry than ordinary training -- batch 500 OOMs a
        # 16 GB T4. These are the settings the project's original script
        # ran successfully on this exact GPU.
        self.synthesizer_params.setdefault("dpctgan", {"epochs": 300, "batch_size": 50})
        if self.metadata_path is None:
            self.metadata_path = os.path.join(self.output_dir, "profile_data", "metadata.json")
        if self.local_full_dataset_path is None:
            self.local_full_dataset_path = os.path.join(self.output_dir, "load_data", "UC1_Resolved_Full.parquet")
        if self.preprocessed_output_path is None:
            self.preprocessed_output_path = os.path.join(self.output_dir, "preprocess", "UC1_Preprocessed.parquet")
        if self.train_output_path is None:
            self.train_output_path = os.path.join(self.output_dir, "preprocess", "UC1_Train.parquet")
        if self.holdout_output_path is None:
            self.holdout_output_path = os.path.join(self.output_dir, "preprocess", "UC1_Holdout.parquet")

    def step_dir(self, step_name: str) -> str:
        """The dedicated output subfolder for a given step: output_dir/<step_name>/."""
        return os.path.join(self.output_dir, step_name)

    def resolved_run_plan(self) -> list[dict]:
        """The full list of generation runs, in execution order.

        Each entry: {run_id, synthesizer, seed, epsilon (DP only),
        columns ("top" = the AIM importance subset, None = full width),
        timeout_seconds (None = the global default)}.
        """
        if self.run_plan is not None:
            return [dict(spec) for spec in self.run_plan]

        def _eps_tag(eps: float) -> str:
            return f"eps{eps:g}".replace(".", "p")

        plan: list[dict] = []
        # 1. Non-DP models, every variance seed (cheap, reliable, first).
        for name in self.non_dp_synthesizers:
            for seed in self.variance_seeds:
                plan.append({"run_id": f"{name}_seed{seed}", "synthesizer": name,
                             "seed": seed, "epsilon": None, "columns": None,
                             "timeout_seconds": None})
        # 2. DP models in configured order (dpctgan ~40 min/run, then the
        #    unknown-cost aim, then the reliable-but-slow mst last so its
        #    near-certain results are the only thing a late crash risks).
        for name in self.dp_synthesizers:
            is_aim = name == "aim"
            for eps in self.dp_epsilons:
                plan.append({
                    "run_id": (f"aim{self.aim_max_columns}_" if is_aim else f"{name}_")
                              + f"{_eps_tag(eps)}_seed{self.variance_seeds[0]}",
                    "synthesizer": name, "seed": self.variance_seeds[0], "epsilon": eps,
                    "columns": "top" if is_aim else None,
                    "timeout_seconds": self.aim_timeout_seconds if is_aim else None,
                })
            if not is_aim:  # variance seeds at the headline epsilon
                for seed in self.variance_seeds[1:]:
                    plan.append({
                        "run_id": f"{name}_{_eps_tag(self.headline_epsilon)}_seed{seed}",
                        "synthesizer": name, "seed": seed,
                        "epsilon": self.headline_epsilon, "columns": None,
                        "timeout_seconds": None,
                    })
        return plan

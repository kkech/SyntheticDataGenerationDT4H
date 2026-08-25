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

    # Optional explicit path to the feature-set metadata JSON, for
    # deployments where it does not live inside the transfer folder
    # (main.py --metadata). None = search transfer_folder for the usual
    # candidate names. Copied to <output>/profile_data/metadata.json,
    # where every downstream step reads it.
    metadata_source: str = None

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
    # The sweep spans strong privacy (1) to a generous upper anchor (20,
    # included to test whether utility saturates above the headline
    # point). Variance seeds run at headline_epsilon: epsilon=15 is the
    # paper's DP operating point -- 20 is a curve anchor, not a release
    # candidate.
    dp_epsilons: tuple = (1.0, 5.0, 8.0, 10.0, 15.0, 20.0)
    headline_epsilon: float = 15.0
    variance_seeds: tuple = (0, 1, 2)
    aim_max_columns: int = 50
    aim_timeout_seconds: int = 7200
    run_plan: tuple = None  # explicit override; None = build from the fields above
    synthesizer_params: dict = None  # per-synthesizer overrides, keyed by name
    # Opt-in roadmap runs appended after the frozen base plan (main.py
    # --extended). Each answers one Future Work question; variants record
    # a qualified synthesizer name (record_as) so downstream grouping
    # never averages them into the base models:
    #   * tvae_qt / ctgan_qt  -- rank/quantile numeric transform;
    #   * tvae_cap256 / tvae_ep1000 -- TVAE capacity/epochs sweep;
    #   * tvae_ind            -- missingness-indicator encoding ablation;
    #   * aim40               -- AIM's full epsilon sweep at 40 columns
    #                            with a tighter measurement-rounds cap;
    #   * mst at epsilon=0.5  -- the low-budget anchor.
    extended_plan: bool = False

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
        # Same memory-stable configuration as dpctgan (same GAN machinery).
        self.synthesizer_params.setdefault("patectgan", {"epochs": 300, "batch_size": 50})
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
        if not self.extended_plan:
            return plan

        # 3. Extended (roadmap) runs, cheapest first. See the field's
        #    comment for what each answers.
        for seed in self.variance_seeds:
            plan.append({"run_id": f"tvae_qt_seed{seed}", "synthesizer": "tvae",
                         "record_as": "tvae_qt", "numeric_transform": "quantile",
                         "seed": seed, "epsilon": None, "columns": None,
                         "timeout_seconds": None})
        plan.append({"run_id": "tvae_cap256_seed0", "synthesizer": "tvae",
                     "record_as": "tvae_cap256", "seed": self.variance_seeds[0],
                     "epsilon": None, "columns": None, "timeout_seconds": None,
                     "params": {"embedding_dim": 256, "compress_dims": (256, 256),
                                "decompress_dims": (256, 256)}})
        plan.append({"run_id": "tvae_ep1000_seed0", "synthesizer": "tvae",
                     "record_as": "tvae_ep1000", "seed": self.variance_seeds[0],
                     "epsilon": None, "columns": None, "timeout_seconds": None,
                     "params": {"epochs": 1000}})
        plan.append({"run_id": "tvae_ind_seed0", "synthesizer": "tvae",
                     "record_as": "tvae_ind", "encoding": "indicator",
                     "seed": self.variance_seeds[0], "epsilon": None,
                     "columns": None, "timeout_seconds": None})
        plan.append({"run_id": "ctgan_qt_seed0", "synthesizer": "ctgan",
                     "record_as": "ctgan_qt", "numeric_transform": "quantile",
                     "seed": self.variance_seeds[0], "epsilon": None,
                     "columns": None, "timeout_seconds": None})
        # aim40 keeps the DEFAULT rounds cap (3 x columns): overriding it
        # lower was measured to destabilize snsynth's exponential
        # mechanism (per-round epsilon grows as rounds shrink; softmax
        # overflows to NaN). The width reduction alone cuts the cost.
        for eps in self.dp_epsilons:
            plan.append({"run_id": f"aim40_{_eps_tag(eps)}_seed{self.variance_seeds[0]}",
                         "synthesizer": "aim", "record_as": "aim40",
                         "seed": self.variance_seeds[0], "epsilon": eps,
                         "columns": 40,
                         "timeout_seconds": self.aim_timeout_seconds})
        # The in-house diffusion baseline (see synthesizers/ddpm.py):
        # three seeds like every non-DP family. More training updates
        # than the GAN defaults -- diffusion converges slower and the
        # model is a small MLP, so this stays cheap on the GPU.
        for seed in self.variance_seeds:
            plan.append({"run_id": f"ddpm_seed{seed}", "synthesizer": "ddpm",
                         "seed": seed, "epsilon": None, "columns": None,
                         "timeout_seconds": None,
                         "params": {"epochs": 2000}})
        # Logic-GUIDED diffusion: identical model, plus the mined
        # implication rules as a sampling-time prior (the paper's own
        # coherence instrument closed into the generator). One run;
        # the ddpm seeds above are its exact unguided control.
        plan.append({"run_id": "ddpm_g_seed0", "synthesizer": "ddpm",
                     "record_as": "ddpm_g", "seed": self.variance_seeds[0],
                     "epsilon": None, "columns": None, "timeout_seconds": None,
                     "params": {"epochs": 2000, "guidance_scale": 5.0}})
        # PATE-CTGAN: the second DP-GAN framework, at three sweep points
        # (a full sweep would only replicate dpctgan's budget-independent
        # failure mode if it fails, and three points suffice if it does
        # not).
        for eps in (1.0, 5.0, self.headline_epsilon):
            plan.append({"run_id": f"patectgan_{_eps_tag(eps)}_seed{self.variance_seeds[0]}",
                         "synthesizer": "patectgan", "seed": self.variance_seeds[0],
                         "epsilon": eps, "columns": None, "timeout_seconds": None})
        # The low-budget anchor gets its own bound: at very small epsilon
        # snsynth/MST's domain compression admits far larger domains and
        # Private-PGM estimation slows nonlinearly (measured), so this
        # run must not be able to eat the campaign. A timeout here is a
        # reportable scaling result, like AIM's.
        plan.append({"run_id": f"mst_{_eps_tag(0.5)}_seed{self.variance_seeds[0]}",
                     "synthesizer": "mst", "seed": self.variance_seeds[0],
                     "epsilon": 0.5, "columns": None, "timeout_seconds": 14400})
        return plan

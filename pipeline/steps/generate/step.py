"""
Step: generate

Executes the configured RUN PLAN (seeds x epsilons x models -- see
PipelineConfig.resolved_run_plan) on the TRAINING split only, and writes
one synthetic dataset per run. Changing the experiment is a config
change rather than a code change.

Built for a dataset intended for publication, which imposes three
requirements beyond "produce some rows":

  * Reproducibility -- every run records its seed, library versions, git
    revision, hardware, and a checksum of the exact training file.
  * Leakage evidence -- every output is checked for verbatim reproduction
    of training records before it is considered usable.
  * Isolation -- each synthesizer runs independently, so one failure
    (missing library, CUDA OOM, Private-PGM exhausting memory on column
    count) neither loses the other results nor hides itself.
"""

import contextlib
import json
import os
import signal
import time
import traceback

import polars as pl

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.generate import leakage
from pipeline.steps.generate.reproducibility import provenance
from pipeline.steps.generate.synthesizers import build_synthesizer


class GenerateStep(PipelineStep):
    name = "generate"

    def run(self, config: PipelineConfig) -> None:
        if not os.path.exists(config.train_output_path):
            raise FileNotFoundError(
                f"{config.train_output_path} not found -- run the preprocess step "
                f"(which writes the train/holdout split) first."
            )

        df_pl = pl.read_parquet(config.train_output_path)
        real = df_pl.to_pandas()
        print(f"Loaded TRAINING split: {real.shape[0]} rows x {real.shape[1]} columns "
              f"(the {config.holdout_fraction:.0%} holdout is never shown to any generator)")

        prov = provenance(config.train_output_path, config.seed)
        self._report_environment(prov)

        train, constants = self._split_constant_columns(real, config)

        categorical, continuous = self._split_column_types(train)
        print(f"Training columns: {len(continuous)} continuous, {len(categorical)} categorical")

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        plan = config.resolved_run_plan()
        print(f"\nRun plan: {len(plan)} run(s) -- "
              + ", ".join(spec["run_id"] for spec in plan))

        # PRE-FLIGHT: if the plan contains DP runs, the reviewed public
        # domain declaration must exist NOW. Each DP run would raise on
        # its own anyway, but discovering that 20 runs in -- after the
        # non-DP models have trained for hours -- helps nobody, and a
        # campaign that cannot produce a valid epsilon claim should not
        # start at all.
        dp_runs = [s["run_id"] for s in plan if self._is_dp(s["synthesizer"])]
        if dp_runs:
            from pipeline.steps.generate.synthesizers.smartnoise_models import (
                load_public_domains,
            )

            domains, sha = load_public_domains(config.public_domains_path)
            print(f"Public numeric domains: {len(domains)} reviewed column range(s) "
                  f"from {config.public_domains_path} (sha256 {sha[:12]}...) -- these "
                  f"bound all {len(dp_runs)} DP run(s); no bound is derived from the "
                  f"training data.")

        # Width-limited runs: "top" = the standard AIM subset
        # (config.aim_max_columns); an integer k = the top-k subset by the
        # same auditable selection. Selections are computed once per
        # distinct width.
        widths = {config.aim_max_columns if spec.get("columns") == "top" else spec["columns"]
                  for spec in plan if spec.get("columns")}
        column_subsets = {k: self._select_top_columns(train, config, out_dir, k=k)
                          for k in sorted(widths)}
        top_columns = column_subsets.get(config.aim_max_columns)

        n_rows = config.n_synthetic_rows or real.shape[0]
        summary = {
            "provenance": prov,
            "input_rows": int(df_pl.height),
            "input_columns": int(df_pl.width),
            "training_columns": int(train.shape[1]),
            "constant_columns_held_out": constants,
            "n_synthetic_rows": n_rows,
            "continuous_columns": len(continuous),
            "categorical_columns": len(categorical),
            "run_plan_size": len(plan),
            "top_columns_for_width_limited_runs": top_columns,
            "runs": [],
        }

        for i, spec in enumerate(plan, 1):
            print(f"\n[run {i}/{len(plan)}]", end="")
            summary["runs"].append(
                self._run_one(spec, train, real, constants, config, out_dir, n_rows,
                              column_subsets)
            )
            # Rewrite the summary after every run rather than once at the
            # end: a plan this long must not lose the results already
            # obtained if a later model hangs or the process is
            # interrupted.
            self._write_summary(summary, out_dir, quiet=True)

        self._write_summary(summary, out_dir)

        ok = [r for r in summary["runs"] if r["status"] == "ok"]
        failed = [r for r in summary["runs"] if r["status"] == "failed"]
        leaky = [r for r in ok if r.get("leakage", {}).get("exact_duplicates_of_training_rows", 0) > 0]

        print(f"\n{len(ok)} run(s) succeeded, {len(failed)} failed.")
        if leaky:
            print(f"🚨 {len(leaky)} output(s) contain verbatim training records: "
                  f"{[r['run_id'] for r in leaky]} -- these must not be published as-is.")
        if not ok:
            raise RuntimeError(
                "Every run failed -- see "
                f"{os.path.join(out_dir, 'DT4H_Generation_Summary.json')} for the errors."
            )

    # --- helpers ---

    @staticmethod
    def _is_dp(synthesizer_name: str) -> bool:
        """Whether a plan entry names a DP synthesizer, without building
        it (the registry class attribute is the single source of truth)."""
        from pipeline.steps.generate.synthesizers import REGISTRY

        cls = REGISTRY.get(synthesizer_name)
        return bool(cls and cls.is_dp)

    def _report_environment(self, prov: dict) -> None:
        env = prov["environment"]
        print(f"Seed: {prov['seed_state']['seed']} | python {env['python']}")
        if env.get("cuda_available"):
            print(f"GPU: {env['gpu_name']} -- {env['gpu_memory_free_gb']} GB free "
                  f"of {env['gpu_memory_total_gb']} GB (CUDA {env.get('cuda_version')})")
            if env["gpu_memory_free_gb"] / max(env["gpu_memory_total_gb"], 1e-9) < 0.6:
                print("⚠️  Much of this GPU is in use by another process. "
                      "Lower config.batch_size if training hits CUDA OOM.")
        elif env.get("cuda_available") is False:
            print("⚠️  No CUDA GPU detected: GPU-backed synthesizers fall back to CPU (slow).")
        if prov["git"]["dirty"] is True:
            print("⚠️  Working tree has uncommitted changes -- this run is not reproducible "
                  "from the recorded git commit alone.")
        elif prov["git"]["dirty"] is None:
            # None means git could not be queried, which is NOT evidence
            # of a clean tree -- do not let silence read as a clean bill.
            print("⚠️  Could not determine the git working-tree state (git unavailable "
                  "or not a repository) -- the recorded revision is unverified.")

    def _split_constant_columns(self, df, config: PipelineConfig):
        """
        Constant columns carry no signal, so training on them wastes model
        capacity and, for DP synthesizers, privacy budget. They are held
        out and re-attached verbatim after sampling, so the published
        schema still matches the real data.
        """
        if not config.drop_constant_columns:
            return df, {}

        constants = {c: df[c].iloc[0] for c in df.columns if df[c].nunique(dropna=False) <= 1}
        if constants:
            print(f"Holding out {len(constants)} constant column(s) from training "
                  f"(re-attached verbatim after sampling).")
            df = df.drop(columns=list(constants))
        return df, {k: str(v) for k, v in constants.items()}

    def _select_top_columns(self, train, config: PipelineConfig, out_dir: str,
                            k: int = None) -> list[str]:
        """The outcome-relevance column subset for width-limited (AIM)
        runs, computed once on the train split and written to a committed
        JSON so the selection is auditable. `k` defaults to
        config.aim_max_columns; other widths get their own JSON."""
        from pipeline.steps.generate.column_selection import (
            FORCED_CLINICAL_COLUMNS,
            select_important_columns,
        )
        from pipeline.steps.utility.targets import declared_outcomes, select_targets

        k = k or config.aim_max_columns
        outcomes = declared_outcomes(config.metadata_path)
        targets = select_targets(train, outcomes, config.utility_max_targets,
                                 explicit=config.utility_targets)
        forced = list(targets) + list(FORCED_CLINICAL_COLUMNS)
        print(f"\nSelecting top {k} outcome-relevant columns for "
              f"width-limited runs ({len(forced)} force-included: "
              f"{len(targets)} TSTR targets + demographics/NYHA)...")
        selected, ranking = select_important_columns(
            train, outcomes, forced, k)
        print(f"  Selected {len(selected)} columns (top ranked: {ranking[:5]})")

        path = os.path.join(out_dir,
                            "DT4H_AIM_Column_Selection.json" if k == config.aim_max_columns
                            else f"DT4H_Column_Selection_top{k}.json")
        with open(path, "w") as f:
            json.dump({
                "method": "mean absolute association (Spearman / Cramer's V / correlation "
                          "ratio) with the metadata-declared outcome variables, computed "
                          "on the training split; TSTR targets, age, gender and NYHA "
                          "force-included; other outcome columns excluded from the ranked pool",
                "k": k,
                "force_included": sorted(c for c in forced if c in train.columns),
                "selected_columns": selected,
                # RANKS ONLY, no scores. The scores are exact statistics
                # of the real training split; publishing them alongside a
                # DP-labelled dataset would release an unnoised function
                # of private data next to the file it bounded. The order
                # is what makes the selection auditable; the magnitudes
                # were only ever diagnostics.
                "ranked_pool": ranking,
                "disclosure": "the selection itself is data-dependent (computed on the "
                              "train split without noise); for DP runs it must be "
                              "disclosed as a non-private step, or replaced by a public "
                              "or DP column choice",
            }, f, indent=2)
        print(f"  Saved column selection (auditable) -> {path}")
        return selected

    @staticmethod
    def _split_column_types(df):
        import pandas as pd

        continuous = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in df.columns if c not in continuous]
        return categorical, continuous

    @staticmethod
    @contextlib.contextmanager
    def _time_limit(seconds, what):
        """Raise TimeoutError inside the block after `seconds`. SIGALRM
        interrupts CPU-bound fitting (AIM/MST) as well as torch training;
        silently a no-op where alarms are unavailable (non-Unix or
        non-main-thread), in which case there is simply no timeout."""
        if not seconds or not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(signum, frame):
            raise TimeoutError(f"{what} exceeded the {seconds}s time limit")

        try:
            old = signal.signal(signal.SIGALRM, _handler)
        except ValueError:  # not the main thread
            yield
            return
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def _run_one(self, spec, train, real, constants, config, out_dir, n_rows,
                 column_subsets) -> dict:
        from pipeline.steps.generate.reproducibility import set_global_seeds

        name = spec["synthesizer"]
        run_id = spec["run_id"]
        run_seed = spec.get("seed", config.seed)
        # Variant runs (quantile transform, indicator ablation, capacity
        # sweep) record a qualified name so downstream grouping never
        # averages them into the base model's seeds.
        recorded_name = spec.get("record_as", name)
        print(f"\n{'=' * 70}\n▶️  RUN: {run_id} (model {name}, seed {run_seed}"
              + (f", ε={spec['epsilon']}" if spec.get("epsilon") is not None else "")
              + f")\n{'=' * 70}")

        params = dict(config.synthesizer_params.get(name, {}))
        params.update(spec.get("params") or {})
        params.setdefault("epochs", config.epochs)
        params.setdefault("batch_size", config.batch_size)

        is_dp = self._is_dp(name)
        if spec.get("epsilon") is not None:
            params["epsilon"] = spec["epsilon"]
        elif is_dp:
            # No fallback epsilon, deliberately: a DP file whose budget
            # was inherited from a config default cannot be reported
            # honestly, and the plan is the place to fix it.
            raise ValueError(
                f"Run '{run_id}' uses the DP synthesizer '{name}' but its spec has no "
                f"epsilon. Every DP run must state its own budget explicitly.")
        if is_dp:
            # Bounds and sentinel identification are per-run inputs the DP
            # synthesizers cannot look up themselves (they never see the
            # config). The domain file is the reviewed public declaration;
            # the encoding map only says WHICH columns are sentinel-coded.
            from pipeline.steps.preprocess.transforms import NUMERIC_ENCODING_FILENAME

            params["public_domains_path"] = config.public_domains_path
            params["numeric_encoding_path"] = os.path.join(
                config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)

        # Width-limited runs train on an importance subset: "top" is the
        # standard AIM width, an integer selects that top-k.
        run_train = train
        width = spec.get("columns")
        width_k = config.aim_max_columns if width == "top" else width
        if width_k:
            subset = column_subsets[width_k]
            run_train = train[[c for c in train.columns if c in subset]]
            print(f"Width-limited run: {run_train.shape[1]} of {train.shape[1]} training "
                  f"columns (top-{width_k} selection, committed JSON).")

        # Optional invertible model-side transform (quantile / indicator).
        transform = None
        if spec.get("numeric_transform") or spec.get("encoding"):
            from pipeline.steps.generate.run_transforms import build_run_transform

            enc_path = os.path.join(config.step_dir("preprocess"),
                                    "DT4H_Numeric_Missing_Encoding.json")
            encoding = json.load(open(enc_path)) if os.path.exists(enc_path) else {}
            transform = build_run_transform(
                spec.get("numeric_transform") or spec.get("encoding"), encoding,
                seed=run_seed)
            run_train = transform.forward(run_train)
            print(f"Run transform '{transform.name}' applied "
                  f"(training frame now {run_train.shape[1]} columns; samples are "
                  f"inverse-transformed before all checks).")
        categorical, continuous = self._split_column_types(run_train)

        record = {"run_id": run_id, "synthesizer": recorded_name, "seed": run_seed,
                  "epsilon": spec.get("epsilon"), "params": params,
                  "base_synthesizer": name,
                  "run_transform": transform.name if transform else None,
                  "trained_columns": int(run_train.shape[1]),
                  "width_limited": bool(width_k)}
        started = time.time()
        try:
            # Per-run seeding: every run is independently reproducible
            # from its recorded seed, not from its position in the plan.
            record["seed_state"] = set_global_seeds(run_seed)
            params["seed"] = run_seed

            synth = build_synthesizer(name, **params)
            record.update(synth.describe())

            if synth.is_dp:
                print(f"Differential privacy ON -- total epsilon budget: {params['epsilon']}"
                      f" (bounds come from the reviewed public domain file; delta is "
                      f"recorded once the model is built)")
            else:
                print("No formal privacy guarantee (non-DP model).")
            if not synth.uses_gpu:
                print("CPU-only synthesizer (no GPU acceleration available for this method).")

            timeout = spec.get("timeout_seconds") or config.synthesizer_timeout_seconds
            if timeout:
                print(f"Training... (time limit: {timeout}s for fit, {timeout}s for sampling)")
            else:
                print("Training... (no time limit)")
            with self._time_limit(timeout, f"'{run_id}' fit"):
                synth.fit(run_train, categorical_columns=categorical, continuous_columns=continuous)
            # Refresh the description AFTER fitting: some models only
            # know fit-dependent facts then (e.g. ddpm's guidance-rule
            # count), and the summary must record the truth.
            record.update(synth.describe())

            # Persist the fitted generator so more synthetic records can
            # be produced later without retraining (see regenerate.py).
            # Saved locally only -- the models directory is gitignored,
            # since a fitted generator has memorized aspects of the real
            # patient data and must be treated as sensitively as the
            # data itself.
            model_path = os.path.join(out_dir, "models", f"{run_id}.pkl")
            try:
                # Transformed runs are saved WRAPPED, so a later load
                # samples ordinary sentinel-space rows -- never the
                # transformed space the model itself was shown.
                to_save = synth
                if transform is not None:
                    from pipeline.steps.generate.run_transforms import TransformedSynthesizer

                    to_save = TransformedSynthesizer(synth, transform)
                self._save_generator(to_save, model_path)
                from pipeline.common.model_io import save_environment_sidecar

                save_environment_sidecar(model_path)
                record["model_path"] = model_path
                print(f"Saved fitted generator (local only, gitignored) -> {model_path}")
            except Exception as save_err:
                record["model_save_error"] = f"{type(save_err).__name__}: {save_err}"
                print(f"⚠️  Could not save generator: {type(save_err).__name__}: {save_err}")

            from pipeline.steps.generate.synthesizers.sdv_models import save_metadata

            meta_path = os.path.join(out_dir, f"DT4H_SDV_Metadata_{run_id}.json")
            if save_metadata(getattr(synth, "_model", None), meta_path):
                record["sdv_metadata_path"] = meta_path
                print(f"Saved SDV metadata (for replicability) -> {meta_path}")

            print(f"Sampling {n_rows} rows...")
            with self._time_limit(timeout, f"'{run_id}' sampling"):
                synthetic = synth.sample(n_rows)
            if transform is not None:
                synthetic = transform.inverse(synthetic)
                print(f"  Inverse '{transform.name}' transform applied to samples.")

            # Generators emit boolean-like columns as actual booleans
            # ("True") while the real schema stores lowercase strings
            # ("true"); align BEFORE the leakage check -- with disjoint
            # spellings an exact-duplicate comparison can never fire.
            from pipeline.common.alignment import align_categorical_case, report

            synthetic, respelled = align_categorical_case(synthetic, train)
            print("  " + report(respelled))
            record["categorical_cells_respelled"] = sum(respelled.values())

            # Re-attach the held-out constants in one concat rather than a
            # column at a time: inserting ~75 columns individually
            # fragments the frame and pandas rightly complains about it.
            if constants:
                import pandas as pd

                held_out = pd.DataFrame(
                    {col: [value] * len(synthetic) for col, value in constants.items()},
                    index=synthetic.index,
                )
                synthetic = pd.concat([synthetic, held_out], axis=1)

            # Restore the real column order so the published file matches
            # the schema of the source dataset.
            synthetic = synthetic[[c for c in real.columns if c in synthetic.columns]].copy()

            # Leakage check BEFORE decoding, so both sides are in
            # sentinel space -- decoding first would turn a memorized
            # row's sentinel into null and let it slip past the
            # exact-match comparison against the sentinel-encoded real
            # frame.
            leak = leakage.check_exact_duplicates(synthetic, real)
            print(leakage.summarize(leak))

            synthetic, decoded = self._decode_numeric_missing(synthetic, config)
            if decoded:
                print(f"  Decoded sentinels back to null in {len(decoded)} numeric "
                      f"column(s): {sum(decoded.values())} cells restored to null "
                      f"('no event' / 'not measured').")
                record["numeric_missing_decoded"] = decoded

            path = os.path.join(out_dir, f"DT4H_Synthetic_{run_id}.csv")
            synthetic.to_csv(path, index=False)

            record.update({
                "status": "ok",
                "output_path": path,
                "output_rows": int(synthetic.shape[0]),
                "output_columns": int(synthetic.shape[1]),
                "leakage": leak,
                "duration_seconds": round(time.time() - started, 1),
            })
            print(f"✅ {run_id}: {synthetic.shape[0]} x {synthetic.shape[1]} -> {path} "
                  f"({record['duration_seconds']}s)")

        except Exception as e:  # keep going so one failure does not lose the whole run
            record.update({
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e)[:500],
                # Keep the traceback: the message alone ("could not find
                # bounds") does not say which library call produced it,
                # and a failed run is expensive to reproduce.
                "traceback": traceback.format_exc()[-4000:],
                "duration_seconds": round(time.time() - started, 1),
            })
            print(f"❌ {run_id} failed after {record['duration_seconds']}s: {type(e).__name__}: {e}")
            print(traceback.format_exc())

        return record

    @staticmethod
    def _save_generator(synth, path: str) -> None:
        """Serialize the fitted Synthesizer wrapper (params + underlying
        model) with cloudpickle, which handles the lambdas/closures inside
        SDV's transformers that plain pickle can reject. cloudpickle
        output loads with the standard pickle module."""
        import cloudpickle

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump(synth, f)

    @staticmethod
    def _decode_numeric_missing(synthetic, config):
        """
        Turn missingness sentinels back into null, and keep synthetic
        observed values inside the observed support.

        Preprocessing encoded every numeric null as a sentinel below that
        column's observed range. Generators emit continuous
        approximations, and the first evaluation run showed them smearing
        mass across the whole sentinel-to-minimum gap: values above the
        sentinel midpoint but below the real minimum survived the old
        threshold rule as impossible "observed" values (negative HDL,
        negative troponin). So the decode rule is now the observed
        minimum itself: anything below a column's real minimum is, by
        construction, not a value that occurs -- it is the model's
        rendering of "missing", and becomes null. The cost is honest and
        stated: a synthetic value slightly below the real minimum that a
        model intended as tail extrapolation is folded into missingness.

        NYHA gets the same treatment for its ordinal encoding: values
        are rounded to the nearest class, rounds of 0 or below (the "not
        assessed" sentinel) become null, and rounds above 4 clamp to 4.
        Idempotent, so it is safe to apply to already-decoded data.
        """
        import pandas as pd

        from pipeline.steps.preprocess.transforms import (
            NUMERIC_ENCODING_FILENAME,
            NYHA_COLUMN,
        )

        decoded = {}

        encoding_path = os.path.join(config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)
        if os.path.exists(encoding_path):
            with open(encoding_path) as f:
                encoding = json.load(f)
            for col, spec in encoding.items():
                if col not in synthetic.columns or not pd.api.types.is_numeric_dtype(synthetic[col]):
                    continue
                floor = spec.get("min_observed", spec["decode_threshold"])
                mask = synthetic[col] < floor
                n = int(mask.sum())
                if n:
                    synthetic.loc[mask, col] = pd.NA
                    decoded[col] = n
        else:
            print(f"⚠️  No numeric encoding map at {encoding_path} -- sentinels (if any) "
                  f"are left as-is. Re-run the preprocess step to produce the map.")

        if NYHA_COLUMN in synthetic.columns and pd.api.types.is_numeric_dtype(synthetic[NYHA_COLUMN]):
            import numpy as np

            # NOT Series.round(): pandas/numpy round half to EVEN, so 2.5
            # becomes 2 while 3.5 becomes 4 -- a systematic, class-dependent
            # bias in an ordinal clinical variable. floor(x + 0.5) rounds
            # half up uniformly, which is what "nearest NYHA class" means.
            col = pd.to_numeric(synthetic[NYHA_COLUMN], errors="coerce")
            col = np.floor(col + 0.5)
            not_assessed = col <= 0
            n = int(not_assessed.sum())
            synthetic[NYHA_COLUMN] = col.clip(upper=4)
            if n:
                synthetic.loc[not_assessed, NYHA_COLUMN] = pd.NA
                decoded[NYHA_COLUMN] = n

        return synthetic, decoded

    def _write_summary(self, summary: dict, out_dir: str, quiet: bool = False) -> None:
        json_path = os.path.join(out_dir, "DT4H_Generation_Summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        md_path = os.path.join(out_dir, "DT4H_Generation_Summary.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(summary))

        if not quiet:
            print(f"\nSaved generation summary -> {json_path}")
            print(f"Saved generation summary -> {md_path}")

    @staticmethod
    def _render_markdown(s: dict) -> str:
        p = s["provenance"]
        env, git = p["environment"], p["git"]
        lines = [
            "# Generation Summary",
            "",
            "## Reproducibility",
            f"- Seed: `{p['seed_state']['seed']}`",
            f"- Git commit: `{git['commit']}` (branch `{git['branch']}`"
            + (", **uncommitted changes present**" if git["dirty"] is True
               else ", working-tree state **unknown**" if git["dirty"] is None
               else "") + ")",
            f"- Training data: `{p['training_data']['path']}`",
            f"- Training data SHA-256: `{p['training_data']['sha256']}`",
            f"- Python {env['python']} on {env['platform']}",
            f"- GPU: {env.get('gpu_name', 'none')}"
            + (f" (CUDA {env.get('cuda_version')})" if env.get("gpu_name") else ""),
            "",
            "| package | version |",
            "|---|---|",
        ]
        for pkg, ver in p["library_versions"].items():
            if ver:
                lines.append(f"| {pkg} | {ver} |")

        lines += [
            "",
            "## Data",
            f"- Training split: {s['input_rows']} rows x {s['input_columns']} columns "
            f"(the holdout split is never shown to any generator)",
            f"- Trained on: {s['training_columns']} columns "
            f"({s['continuous_columns']} continuous, {s['categorical_columns']} categorical)",
            f"- Constant columns held out and re-attached verbatim: "
            f"{len(s['constant_columns_held_out'])}",
            f"- Synthetic rows generated per run: {s['n_synthetic_rows']}",
        ]
        if s.get("top_columns_for_width_limited_runs"):
            lines.append(
                f"- Width-limited (AIM) runs train on {len(s['top_columns_for_width_limited_runs'])} "
                f"outcome-relevant columns (selection: `DT4H_AIM_Column_Selection.json`)")
        lines += [
            "",
            "## Runs",
            "",
            "| run | model | ε | δ | seed | status | rows x cols | duration | verbatim training rows | notes |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in s["runs"]:
            dp = f"{r['epsilon']:g}" if r.get("epsilon") is not None else "-"
            # delta is only meaningful for the DP runs; recorded per run
            # by the synthesizer's describe(), together with the sha256 of
            # the public domain file that bounded it.
            dl = f"{r['delta']:.3g}" if r.get("delta") is not None else "-"
            if r["status"] == "ok":
                shape = f"{r['output_rows']} x {r['output_columns']}"
                lk = r.get("leakage", {})
                n = lk.get("exact_duplicates_of_training_rows")
                leak_cell = "0 ✅" if n == 0 else f"**{n}** 🚨"
                notes = ""
                dupes = lk.get("synthetic_duplicate_rows_within_output", 0)
                if dupes:
                    notes = f"{dupes} duplicate rows within output"
            else:
                shape, leak_cell = "-", "-"
                notes = f"{r.get('error_type')}: {str(r.get('error'))[:80]}"
            if r.get("width_limited"):
                notes = (notes + "; " if notes else "") + f"width-limited ({r['trained_columns']} cols)"
            if r.get("public_domains_sha256"):
                notes = (notes + "; " if notes else "") + \
                    f"bounds `{r['public_domains_sha256'][:12]}`"
            lines.append(
                f"| {r.get('run_id', r['synthesizer'])} | {r['synthesizer']} | {dp} | {dl} "
                f"| {r.get('seed', '-')} | {r['status']} | {shape} | "
                f"{r.get('duration_seconds')}s | {leak_cell} | {notes} |"
            )

        if s.get("single_run_updates"):
            lines += [
                "",
                "## Single-run updates",
                "",
                "Runs re-executed on their own after the campaign, typically "
                "to retry an infrastructural failure such as a time limit. The table above "
                "shows the latest attempt for each.",
                "",
                "| run | at (UTC) | status | previous | time limit | commit |",
                "|---|---|---|---|---|---|",
            ]
            for u in s["single_run_updates"]:
                lines.append(
                    f"| {u.get('run_id')} | {u.get('at')} | {u.get('status')} "
                    f"| {u.get('previous_status') or 'not in summary'} "
                    f"| {u.get('timeout_seconds') or 'default'}s "
                    f"| `{str(u.get('git_commit'))[:8]}` |"
                )

        lines += [
            "",
            "## Caveats",
            "- The leakage column counts EXACT reproductions of training rows only. It does "
            "not detect near-duplicates; the privacy step's distance-to-closest-record "
            "analysis against the holdout baseline covers the rest.",
            "- Non-DP models carry no formal privacy guarantee regardless of this check.",
            "- DP runs are bounded by the reviewed a-priori public domains in "
            "`public_domains.json` (sha-256 recorded per run above), so no epsilon is "
            "spent on -- and no bound is derived from -- the training data. Two "
            "residual, disclosed leaks remain: snsynth learns categorical vocabularies "
            "from the training data, and AIM's column selection is computed on the "
            "train split without noise.",
            "- Width-limited runs synthesize a column subset by design; their files have "
            "fewer columns and are evaluated over those columns only.",
        ]
        return "\n".join(lines) + "\n"

"""
Step: generate

Trains one or more synthesizers on the preprocessed data and writes
synthetic datasets. Which synthesizers run is config-driven
(config.synthesizers), so comparing CTGAN vs TVAE vs AIM vs MST is a
config change rather than a code change.

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

import json
import os
import time

import polars as pl

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.generate import leakage
from pipeline.steps.generate.reproducibility import provenance
from pipeline.steps.generate.synthesizers import build_synthesizer


class GenerateStep(PipelineStep):
    name = "generate"

    def run(self, config: PipelineConfig) -> None:
        if not os.path.exists(config.preprocessed_output_path):
            raise FileNotFoundError(
                f"{config.preprocessed_output_path} not found -- run the preprocess step first."
            )

        df_pl = pl.read_parquet(config.preprocessed_output_path)
        real = df_pl.to_pandas()
        print(f"Loaded preprocessed data: {real.shape[0]} rows x {real.shape[1]} columns")

        prov = provenance(config.preprocessed_output_path, config.seed)
        self._report_environment(prov)

        train, constants = self._split_constant_columns(real, config)
        train, dropped_for_width = self._limit_columns(train, config)

        categorical, continuous = self._split_column_types(train)
        print(f"Training columns: {len(continuous)} continuous, {len(categorical)} categorical")

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        n_rows = config.n_synthetic_rows or real.shape[0]
        summary = {
            "provenance": prov,
            "input_rows": int(df_pl.height),
            "input_columns": int(df_pl.width),
            "training_columns": int(train.shape[1]),
            "constant_columns_held_out": constants,
            "columns_dropped_by_max_columns": dropped_for_width,
            "n_synthetic_rows": n_rows,
            "continuous_columns": len(continuous),
            "categorical_columns": len(categorical),
            "runs": [],
        }

        for name in config.synthesizers:
            summary["runs"].append(
                self._run_one(name, train, real, categorical, continuous, constants, config, out_dir, n_rows)
            )
            # Rewrite the summary after every synthesizer rather than once
            # at the end: a long run (CTGAN at 500 epochs takes ~20 min on
            # a T4) should not lose the results already obtained if a
            # later model hangs or the process is interrupted.
            self._write_summary(summary, out_dir, quiet=True)

        self._write_summary(summary, out_dir)

        ok = [r for r in summary["runs"] if r["status"] == "ok"]
        failed = [r for r in summary["runs"] if r["status"] == "failed"]
        leaky = [r for r in ok if r.get("leakage", {}).get("exact_duplicates_of_training_rows", 0) > 0]

        print(f"\n{len(ok)} synthesizer(s) succeeded, {len(failed)} failed.")
        if leaky:
            print(f"🚨 {len(leaky)} output(s) contain verbatim training records: "
                  f"{[r['synthesizer'] for r in leaky]} -- these must not be published as-is.")
        if not ok:
            raise RuntimeError(
                "Every synthesizer failed -- see "
                f"{os.path.join(out_dir, 'DT4H_Generation_Summary.json')} for the errors."
            )

    # --- helpers ---

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
        if prov["git"]["dirty"]:
            print("⚠️  Working tree has uncommitted changes -- this run is not reproducible "
                  "from the recorded git commit alone.")

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

    def _limit_columns(self, df, config: PipelineConfig):
        """
        Optional width cap, for trialling Private-PGM-based synthesizers
        (aim/mst) before committing to a full-width run.
        """
        if not config.max_columns or df.shape[1] <= config.max_columns:
            return df, []
        keep = list(df.columns[: config.max_columns])
        dropped = [c for c in df.columns if c not in keep]
        print(f"⚠️  max_columns={config.max_columns}: training on {len(keep)} of {df.shape[1]} "
              f"columns. TRIAL configuration -- not a publishable full-width result.")
        return df[keep], dropped

    @staticmethod
    def _split_column_types(df):
        import pandas as pd

        continuous = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in df.columns if c not in continuous]
        return categorical, continuous

    def _run_one(self, name, train, real, categorical, continuous, constants, config, out_dir, n_rows) -> dict:
        print(f"\n{'=' * 70}\n▶️  SYNTHESIZER: {name}\n{'=' * 70}")
        params = dict(config.synthesizer_params.get(name, {}))
        params.setdefault("epochs", config.epochs)
        params.setdefault("batch_size", config.batch_size)
        params.setdefault("epsilon", config.epsilon)

        record = {"synthesizer": name, "params": params}
        started = time.time()
        try:
            synth = build_synthesizer(name, **params)
            record.update(synth.describe())

            if synth.is_dp:
                print(f"Differential privacy ON -- total epsilon budget: {params['epsilon']}")
            else:
                print("No formal privacy guarantee (non-DP model).")
            if not synth.uses_gpu:
                print("CPU-only synthesizer (no GPU acceleration available for this method).")

            print("Training...")
            synth.fit(train, categorical_columns=categorical, continuous_columns=continuous)

            from pipeline.steps.generate.synthesizers.sdv_models import save_metadata

            meta_path = os.path.join(out_dir, f"DT4H_SDV_Metadata_{name}.json")
            if save_metadata(getattr(synth, "_model", None), meta_path):
                record["sdv_metadata_path"] = meta_path
                print(f"Saved SDV metadata (for replicability) -> {meta_path}")

            print(f"Sampling {n_rows} rows...")
            synthetic = synth.sample(n_rows)

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

            leak = leakage.check_exact_duplicates(synthetic, real)
            print(leakage.summarize(leak))

            path = os.path.join(out_dir, f"DT4H_Synthetic_{name}.csv")
            synthetic.to_csv(path, index=False)

            record.update({
                "status": "ok",
                "output_path": path,
                "output_rows": int(synthetic.shape[0]),
                "output_columns": int(synthetic.shape[1]),
                "leakage": leak,
                "duration_seconds": round(time.time() - started, 1),
            })
            print(f"✅ {name}: {synthetic.shape[0]} x {synthetic.shape[1]} -> {path} "
                  f"({record['duration_seconds']}s)")

        except Exception as e:  # keep going so one failure does not lose the whole run
            record.update({
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e)[:500],
                "duration_seconds": round(time.time() - started, 1),
            })
            print(f"❌ {name} failed after {record['duration_seconds']}s: {type(e).__name__}: {e}")

        return record

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
            + (", **uncommitted changes present**" if git["dirty"] else "") + ")",
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
            f"- Input: {s['input_rows']} rows x {s['input_columns']} columns",
            f"- Trained on: {s['training_columns']} columns "
            f"({s['continuous_columns']} continuous, {s['categorical_columns']} categorical)",
            f"- Constant columns held out and re-attached verbatim: "
            f"{len(s['constant_columns_held_out'])}",
            f"- Synthetic rows generated: {s['n_synthetic_rows']}",
            "",
            "## Runs",
            "",
            "| synthesizer | DP | status | rows x cols | duration | verbatim training rows | notes |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in s["runs"]:
            dp = f"ε={r['params'].get('epsilon')}" if r.get("is_dp") else "no"
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
            lines.append(
                f"| {r['synthesizer']} | {dp} | {r['status']} | {shape} | "
                f"{r.get('duration_seconds')}s | {leak_cell} | {notes} |"
            )

        lines += [
            "",
            "## Caveats",
            "- The leakage column counts EXACT reproductions of training rows only. It does "
            "not detect near-duplicates and does not bound membership-inference risk; a full "
            "privacy assessment (distance-to-closest-record, membership inference) is still "
            "required before release.",
            "- Non-DP models carry no formal privacy guarantee regardless of this check.",
        ]
        if s["columns_dropped_by_max_columns"]:
            lines.append(f"- ⚠️ TRIAL RUN: {len(s['columns_dropped_by_max_columns'])} column(s) "
                         f"excluded via max_columns -- not a full-width result.")
        return "\n".join(lines) + "\n"

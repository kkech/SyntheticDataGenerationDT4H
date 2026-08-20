"""
Step: evaluate

Measures distribution distance between the three stages of the pipeline:

  * original      -- the raw loaded dataset (output/load_data/);
  * preprocessed  -- the training frame, with numeric sentinels decoded
                     back to null so its OBSERVED distributions are the
                     thing compared;
  * synthetic     -- every DT4H_Synthetic_*.csv the generate step wrote.

Three families of comparison, each answering a different question:

  original vs preprocessed   did preprocessing distort the data it kept?
                             (should be ~zero on every untouched column)
  preprocessed vs synthetic  how faithful is each generator to what it
                             was trained on? (the headline fidelity result)
  original vs synthetic      end-to-end: how far is the released file
                             from the source data?

Writes per-synthesizer JSON detail plus one Markdown overview to
output/evaluate/. Aggregate statistics only -- safe to commit.
"""

import glob
import json
import os

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.evaluate.metrics import compare_frames


class EvaluateStep(PipelineStep):
    name = "evaluate"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd
        import polars as pl

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError(
                f"No DT4H_Synthetic_*.csv in {config.step_dir('generate')} -- run the generate step first."
            )

        preprocessed = self._load_preprocessed_decoded(config)
        print(f"Preprocessed (sentinels decoded to null): {preprocessed.shape[0]} x {preprocessed.shape[1]}")

        original = None
        if os.path.exists(config.local_full_dataset_path):
            orig_pl = pl.read_parquet(config.local_full_dataset_path)
            # Flatten List-typed ARRAY[NOMINAL] columns the same way
            # preprocessing does, so their cells are scalars rather than
            # arrays (which the categorical metrics cannot hash).
            if os.path.exists(config.metadata_path):
                from pipeline.steps.preprocess.transforms import (
                    flatten_array_columns,
                    load_variable_metadata,
                    normalize_numeric_dtypes,
                )

                orig_pl, _ = flatten_array_columns(orig_pl, load_variable_metadata(config.metadata_path))
            # Same dtype normalization preprocess applies (Decimal ->
            # Float64), otherwise Decimal labs arrive as pandas object
            # columns, fall into the categorical branch, and score a
            # spurious TVD of 1.0 on string formatting differences.
            orig_pl, _ = normalize_numeric_dtypes(orig_pl)
            original = self._prepare_original(orig_pl.to_pandas(), config)
            print(f"Original: {original.shape[0]} x {original.shape[1]}")
        else:
            print(f"⚠️  Original dataset not found at {config.local_full_dataset_path} -- "
                  f"original-based comparisons skipped this run.")

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        results = {"comparisons": []}
        if original is not None:
            print("\nComparing original vs preprocessed...")
            results["comparisons"].append(
                compare_frames(original, preprocessed, "original", "preprocessed")
            )

        for path in synthetic_files:
            synth_name = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synthetic = pd.read_csv(path, low_memory=False)
            # Defensive re-decode (idempotent): synthetic files written
            # before a decode-rule fix may still contain gap-region or
            # sentinel values; applying the current rule here means the
            # evaluation always scores what the current pipeline would
            # publish, without retraining anything.
            from pipeline.steps.generate.step import GenerateStep

            synthetic, _ = GenerateStep._decode_numeric_missing(synthetic, config)
            print(f"\nComparing against synthetic '{synth_name}' ({synthetic.shape[0]} rows)...")

            entry = {
                "synthesizer": synth_name,
                "preprocessed_vs_synthetic": compare_frames(
                    preprocessed, synthetic, "preprocessed", f"synthetic[{synth_name}]"
                ),
            }
            if original is not None:
                entry["original_vs_synthetic"] = compare_frames(
                    original, synthetic, "original", f"synthetic[{synth_name}]"
                )
            results["comparisons"].append(entry)

            agg = entry["preprocessed_vs_synthetic"]["aggregates"]
            print(f"  preprocessed vs {synth_name}: "
                  f"KS mean={agg['ks'].get('mean')} (frac<0.1: {agg['ks_frac_below_0.1']}), "
                  f"TVD mean={agg['tvd'].get('mean')} (frac<0.05: {agg['tvd_frac_below_0.05']}), "
                  f"missing-rate MAD={agg['missing_rate_mean_abs_diff']}")

        json_path = os.path.join(out_dir, "DT4H_Evaluation.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved evaluation detail (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Evaluation.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"Saved evaluation overview (Markdown) -> {md_path}")

    # --- frame preparation ---

    def _load_preprocessed_decoded(self, config: PipelineConfig):
        """The training frame with sentinels decoded back to null, so the
        comparison sees observed distributions, not sentinel spikes."""
        import pandas as pd
        import polars as pl

        from pipeline.steps.preprocess.transforms import NUMERIC_ENCODING_FILENAME

        from pipeline.steps.generate.step import GenerateStep

        df = pl.read_parquet(config.preprocessed_output_path).to_pandas()
        df, _ = GenerateStep._decode_numeric_missing(df, config)
        return df

    def _prepare_original(self, df, config: PipelineConfig):
        """Light alignment so original-side comparisons measure real drift
        rather than encoding artifacts: NYHA LOINC codes are mapped to the
        same 1-4 ordinals the pipeline uses. Everything else is compared
        as-is (booleans/strings are normalized inside the metrics)."""
        from pipeline.steps.preprocess.transforms import (
            NYHA_COLUMN,
            build_nyha_map,
            load_variable_metadata,
        )

        if NYHA_COLUMN in df.columns and os.path.exists(config.metadata_path):
            var_meta = load_variable_metadata(config.metadata_path)
            nyha_map = build_nyha_map(var_meta)
            df = df.copy()
            df[NYHA_COLUMN] = df[NYHA_COLUMN].map(nyha_map)
        return df

    # --- reporting ---

    @staticmethod
    def _render_markdown(results: dict) -> str:
        lines = [
            "# Evaluation: original vs preprocessed vs synthetic",
            "",
            "Metrics are computed per column over observed values (nulls excluded); "
            "missingness rates are compared separately. KS and TVD are in [0,1], "
            "lower is closer; `W/std` is the Wasserstein distance in units of the "
            "reference standard deviation.",
            "",
            "| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        def _row(c):
            a = c["aggregates"]
            return (f"| {c['pair']} | {c['columns_compared']} "
                    f"| {a['ks'].get('mean', '-')} | {a['ks'].get('median', '-')} "
                    f"| {a['ks_frac_below_0.1'] if a['ks_frac_below_0.1'] is not None else '-'} "
                    f"| {a['wasserstein_std'].get('mean', '-')} "
                    f"| {a['tvd'].get('mean', '-')} "
                    f"| {a['tvd_frac_below_0.05'] if a['tvd_frac_below_0.05'] is not None else '-'} "
                    f"| {a['missing_rate_mean_abs_diff']} |")

        detail_sections = []
        for c in results["comparisons"]:
            if "pair" in c:  # original vs preprocessed
                lines.append(_row(c))
                detail_sections.append(c)
            else:
                lines.append(_row(c["preprocessed_vs_synthetic"]))
                detail_sections.append(c["preprocessed_vs_synthetic"])
                if "original_vs_synthetic" in c:
                    lines.append(_row(c["original_vs_synthetic"]))
                    detail_sections.append(c["original_vs_synthetic"])

        for c in detail_sections:
            lines += ["", f"## {c['pair']}", ""]
            if c["worst_numeric"]:
                lines.append("Worst numeric columns (by KS):")
                for r in c["worst_numeric"]:
                    lines.append(f"- `{r['column']}`: KS={r['ks_statistic']}, W/std={r['wasserstein_std']}, "
                                 f"mean {r['mean_a']} -> {r['mean_b']}, "
                                 f"missing {r['missing_rate_a']:.0%} -> {r['missing_rate_b']:.0%}")
            if c["worst_categorical"]:
                lines.append("Worst categorical columns (by TVD):")
                for r in c["worst_categorical"]:
                    lines.append(f"- `{r['column']}`: TVD={r['tvd']}, "
                                 f"{r['n_categories_a']} -> {r['n_categories_b']} categories, "
                                 f"missing {r['missing_rate_a']:.0%} -> {r['missing_rate_b']:.0%}")

        return "\n".join(lines) + "\n"

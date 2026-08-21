"""
Step: preprocess

Turns the full raw dataset into a GAN-ready feature set: combines
medication/condition columns, encodes NYHA via the metadata valueSet,
drops identifiers/datetimes, then resolves all remaining missingness:
numeric nulls become per-column sentinels (decoded back to null after
generation -- numeric missingness carries meaning and is never imputed),
boolean/categorical nulls become an explicit "Missing" category.

Also writes a summary of every transformation decision made (what was
combined/dropped/imputed and why) to output/preprocess/DT4H_Preprocessing_Summary
-- this is what happened, distinct from profile_preprocessed_data's
statistical profile of the resulting data.
"""

import json
import os

import polars as pl

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.preprocess import transforms as t


class PreprocessStep(PipelineStep):
    name = "preprocess"

    def run(self, config: PipelineConfig) -> None:
        if not os.path.exists(config.local_full_dataset_path):
            raise FileNotFoundError(
                f"{config.local_full_dataset_path} not found -- run the load_data step first."
            )

        df = pl.read_parquet(config.local_full_dataset_path)
        var_meta = t.load_variable_metadata(config.metadata_path)
        print(f"Loaded {df.height} rows x {df.width} columns.")

        summary = {
            "input_rows": df.height,
            "input_columns": df.width,
            "unique_patients": df["pid"].n_unique() if "pid" in df.columns else None,
        }
        if summary["unique_patients"] is not None:
            print(f"Unique patients (pid): {summary['unique_patients']}")

        print("Validating against metadata...")
        summary["metadata_validation"] = t.validate_against_metadata(df, var_meta)

        print("Checking expected non-null pairs...")
        summary["expected_nonnull_checks"] = t.report_expected_nonnull_mismatches(df)

        print("Flattening ARRAY[NOMINAL] columns...")
        df, summary["array_columns_flattened"] = t.flatten_array_columns(df, var_meta)

        # Runs early so every numeric column is a clean Float64 with nulls
        # (not Decimal, not NaN) before any imputation reads or writes it.
        print("Normalizing numeric dtypes...")
        df, summary["numeric_dtype_normalization"] = t.normalize_numeric_dtypes(df)

        print("Checking symptom columns...")
        summary["symptom_columns"] = t.report_symptom_columns(df)

        print("Combining medication columns...")
        df, summary["medications_combined"] = t.combine_medications(df)

        print("Combining condition columns...")
        df, summary["conditions_combined"] = t.combine_conditions(df)

        print("Encoding NYHA...")
        df, summary["nyha_encoding"] = t.encode_nyha(df, var_meta)

        print("Preferring _first/_last numeric variants...")
        df, summary["numeric_aggregates_dropped"] = t.prefer_first_last_numerics(df)

        print("Dropping identifier/datetime columns...")
        df, summary["identifiers_datetimes_dropped"] = t.drop_identifiers_and_datetimes(df, var_meta)

        print("Dropping near-unique identifier-like columns (safety net)...")
        df, summary["near_unique_columns_dropped"] = t.drop_near_unique_columns(df)

        print("Final null cleanup...")
        df, summary["nyha_missing_imputation"] = t.impute_nyha_missing(df)
        # Numeric nulls carry meaning ("no event" for time-to-event
        # columns, "not measured" for labs/vitals), so they are NOT
        # imputed: each gets a per-column sentinel below the observed
        # range, which the generate step decodes back to null in the
        # synthetic output.
        encoding_path = os.path.join(config.step_dir(self.name), t.NUMERIC_ENCODING_FILENAME)
        df, summary["numeric_missing_encoding"] = t.encode_numeric_missing(df, var_meta, encoding_path)
        df, summary["categorical_imputation"] = t.impute_categorical_and_boolean(df, var_meta)

        # Count NaN as well as null: polars keeps them distinct, so a
        # null-only check reports a reassuring zero while NaN cells sit in
        # the output. Both mean "missing" to a synthesizer.
        remaining_nulls = sum(df[c].null_count() for c in df.columns)
        remaining_nans = sum(
            int(df[c].is_nan().sum()) for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)
        )
        summary["remaining_null_cells"] = remaining_nulls
        summary["remaining_nan_cells"] = remaining_nans
        summary["output_rows"] = df.height
        summary["output_columns"] = df.width
        print(f"Remaining missing cells after all imputation: {remaining_nulls} null, {remaining_nans} NaN")

        if remaining_nulls or remaining_nans:
            raise ValueError(
                f"Preprocessing finished with missing values still present "
                f"({remaining_nulls} null, {remaining_nans} NaN) -- the output is not "
                f"ready for synthesis. Refusing to write it."
            )

        os.makedirs(os.path.dirname(config.preprocessed_output_path), exist_ok=True)
        df.write_parquet(config.preprocessed_output_path)
        print(f"Saved {df.height} rows x {df.width} columns -> {config.preprocessed_output_path}")

        summary["holdout_split"] = self._split_train_holdout(df, config)

        self._write_summary(summary, config)

    def _split_train_holdout(self, df: pl.DataFrame, config: PipelineConfig) -> dict:
        """Seeded row split BEFORE generation. The generators train only
        on the train file; the holdout rows are real patients the models
        never see, which is what makes TSTR testing, the privacy
        baseline, and the evaluation noise floor honest. The manifest
        (row positions only -- no patient data) is committed so the
        exact split is auditable and reproducible."""
        import numpy as np

        rng = np.random.default_rng(config.seed)
        n = df.height
        n_holdout = int(round(n * config.holdout_fraction))
        perm = rng.permutation(n)
        holdout_idx = np.sort(perm[:n_holdout])
        train_idx = np.sort(perm[n_holdout:])

        indexed = df.with_row_index("__rid")
        train_df = indexed.filter(pl.col("__rid").is_in(train_idx.tolist())).drop("__rid")
        holdout_df = indexed.filter(pl.col("__rid").is_in(holdout_idx.tolist())).drop("__rid")

        train_df.write_parquet(config.train_output_path)
        holdout_df.write_parquet(config.holdout_output_path)
        print(f"Holdout split (seed {config.seed}): {train_df.height} train rows -> "
              f"{config.train_output_path}, {holdout_df.height} holdout rows -> "
              f"{config.holdout_output_path}")

        manifest = {
            "seed": config.seed,
            "holdout_fraction": config.holdout_fraction,
            "n_total": n,
            "n_train": train_df.height,
            "n_holdout": holdout_df.height,
            "holdout_row_positions": holdout_idx.tolist(),
        }
        manifest_path = os.path.join(config.step_dir(self.name), "DT4H_Holdout_Split.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved holdout split manifest -> {manifest_path}")
        return {k: v for k, v in manifest.items() if k != "holdout_row_positions"}

    def _write_summary(self, summary: dict, config: PipelineConfig) -> None:
        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, "DT4H_Preprocessing_Summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved preprocessing summary (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Preprocessing_Summary.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(summary))
        print(f"Saved preprocessing summary (Markdown) -> {md_path}")

    @staticmethod
    def _render_markdown(s: dict) -> str:
        lines = [
            "# Preprocessing Summary",
            "",
            f"- Input: {s['input_rows']} rows x {s['input_columns']} columns"
            + (f" ({s['unique_patients']} unique patients)" if s.get("unique_patients") is not None else ""),
            f"- Output: {s['output_rows']} rows x {s['output_columns']} columns",
            f"- Remaining missing cells: {s['remaining_null_cells']} null, "
            f"{s.get('remaining_nan_cells', 'n/a')} NaN",
            (f"- Holdout split (seed {s['holdout_split']['seed']}): "
             f"{s['holdout_split']['n_train']} train / {s['holdout_split']['n_holdout']} holdout rows "
             f"({s['holdout_split']['holdout_fraction']:.0%} held out, never seen by any generator)"
             if s.get("holdout_split") else "- Holdout split: (not performed)"),
            "",
            "## Metadata validation",
            f"- {s['metadata_validation']['matched']} / {s['metadata_validation']['declared_in_metadata']} "
            f"declared columns matched in data",
        ]
        if s["metadata_validation"]["declared_but_missing_from_data"]:
            lines.append(f"- ⚠️ declared but missing from data: {s['metadata_validation']['declared_but_missing_from_data']}")
        if s["metadata_validation"]["in_data_but_not_declared"]:
            lines.append(f"- ⚠️ in data but not declared: {s['metadata_validation']['in_data_but_not_declared']}")

        lines += ["", "## Expected non-null pair checks"]
        for c in s["expected_nonnull_checks"]:
            if c.get("skipped"):
                lines.append(f"- (skipped) {c['col_a']} / {c['col_b']}: not found")
            else:
                flag = "⚠️ " if c["mismatch"] else ""
                lines.append(f"- {flag}{c['col_a']}: {c['n_a']} vs {c['col_b']}: {c['n_b']} ({c['note']})")

        lines += [
            "",
            "## Transformations",
            f"- ARRAY[NOMINAL] columns flattened: {s['array_columns_flattened']['flattened']}",
            f"- Symptom columns: {s['symptom_columns']['count']} present, "
            f"{s['symptom_columns']['currently_constant']} currently constant, kept (not dropped)",
            f"- Medications combined into {len(s['medications_combined']['features_created'])} feature(s) "
            f"(from {s['medications_combined']['source_columns_dropped']} source columns)",
            f"- Conditions combined into {len(s['conditions_combined']['features_created'])} feature(s) "
            f"(from {s['conditions_combined']['source_columns_dropped']} source columns)",
            f"- NYHA encoding: {'skipped (column not found)' if s['nyha_encoding'].get('skipped') else s['nyha_encoding']['map']}",
            f"- Numeric aggregate columns dropped (bare/_min/_max/_avg/_stddev): "
            f"{len(s['numeric_aggregates_dropped']['dropped'])}",
            f"- IDENTIFIER/DATETIME columns dropped: {s['identifiers_datetimes_dropped']['dropped']}",
            f"- Near-unique identifier-like columns dropped (safety net, not caught by declared type): "
            f"{s['near_unique_columns_dropped']['dropped']}",
            f"- Decimal columns cast to Float64: "
            f"{s.get('numeric_dtype_normalization', {}).get('decimal_cast_to_float', [])}",
        ]

        lines += [
            "",
            "## Final null cleanup",
            f"- NYHA: filled {s['nyha_missing_imputation']['filled']} missing value(s) with sentinel "
            f"{s['nyha_missing_imputation'].get('sentinel')}",
        ]
        nm = s["numeric_missing_encoding"]
        lines.append(
            f"- Numeric nulls are NOT imputed -- missingness carries meaning. "
            f"{nm['n_columns_encoded']} column(s) sentinel-encoded "
            f"({nm['n_no_event_columns']} time-to-event 'no event', "
            f"{nm['n_columns_encoded'] - nm['n_no_event_columns']} 'not measured'), "
            f"each with a per-column sentinel below the observed range, decoded back "
            f"to null in the synthetic output (map: `{nm['encoding_path']}`)."
        )
        lines.append(
            f"- Dropped {len(nm['dropped_too_few'])} numeric column(s) with fewer than "
            f"{nm['min_nonnull_required']} observed values:"
        )
        for d in nm["dropped_too_few"]:
            lines.append(f"  - `{d['column']}` (only {d['n_observed']} observed)")
        lines.append(
            f"- Categorical/boolean: normalized "
            f"{s['categorical_imputation'].get('normalized_columns', 'n/a')} column(s) to String; "
            f"{len(s['categorical_imputation']['filled_columns'])} of them had nulls filled "
            f"with an explicit 'Missing' category"
        )

        return "\n".join(lines) + "\n"

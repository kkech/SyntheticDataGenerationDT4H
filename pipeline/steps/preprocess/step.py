"""
Step: preprocess

Turns the full raw dataset into a GAN-ready feature set: combines
medication/condition columns, encodes NYHA via the metadata valueSet,
drops identifiers/datetimes, applies Machteld's temporary dummy
imputation for currently-missing labs (toggleable), then a generic final
imputation pass (bootstrap for numerics + missingness flags, "Missing"
category for booleans/categoricals) so no nulls remain.

Also writes a summary of every transformation decision made (what was
combined/dropped/imputed and why) to for_repo/DT4H_Preprocessing_Summary
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

        print("Checking symptom columns...")
        summary["symptom_columns"] = t.report_symptom_columns(df)

        print("Combining medication columns...")
        df, summary["medications_combined"] = t.combine_medications(df)

        print("Combining condition columns...")
        df, summary["conditions_combined"] = t.combine_conditions(df)

        print("Encoding NYHA...")
        df, summary["nyha_encoding"] = t.encode_nyha(df, var_meta)

        if config.apply_dummy_imputation:
            print("Applying temporary dummy imputation (Machteld's placeholder rules)...")
            df, summary["dummy_imputation"] = t.apply_dummy_imputation(df)
        else:
            print("Skipping dummy imputation (apply_dummy_imputation=False).")
            summary["dummy_imputation"] = {"disabled": True}

        print("Preferring _first/_last numeric variants...")
        df, summary["numeric_aggregates_dropped"] = t.prefer_first_last_numerics(df)

        print("Dropping identifier/datetime columns...")
        df, summary["identifiers_datetimes_dropped"] = t.drop_identifiers_and_datetimes(df, var_meta)

        print("Dropping near-unique identifier-like columns (safety net)...")
        df, summary["near_unique_columns_dropped"] = t.drop_near_unique_columns(df)

        print("Final null cleanup...")
        df, summary["nyha_missing_imputation"] = t.impute_nyha_missing(df)
        df, summary["numeric_imputation"] = t.impute_numeric_columns(df, var_meta)
        df, summary["categorical_imputation"] = t.impute_categorical_and_boolean(df, var_meta)

        remaining_nulls = sum(df[c].null_count() for c in df.columns)
        summary["remaining_null_cells"] = remaining_nulls
        summary["output_rows"] = df.height
        summary["output_columns"] = df.width
        print(f"Remaining null cells after all imputation: {remaining_nulls}")

        os.makedirs(os.path.dirname(config.preprocessed_output_path), exist_ok=True)
        df.write_parquet(config.preprocessed_output_path)
        print(f"Saved {df.height} rows x {df.width} columns -> {config.preprocessed_output_path}")

        self._write_summary(summary, config)

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
            f"- Remaining null cells: {s['remaining_null_cells']}",
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
        ]

        lines += ["", "## Dummy imputation (Machteld's temporary placeholder rules)"]
        if s["dummy_imputation"].get("disabled"):
            lines.append("- Skipped (apply_dummy_imputation=False)")
        else:
            for fill in s["dummy_imputation"]["fills"]:
                lines.append(f"- Filled {fill['n_filled']} value(s) in `{fill['target']}` "
                              f"(triggered by `{fill['trigger']}` present)")
            for sk in s["dummy_imputation"]["skipped"]:
                lines.append(f"- (skipped) {sk}")

        lines += [
            "",
            "## Final null cleanup",
            f"- NYHA: filled {s['nyha_missing_imputation']['filled']} missing value(s) with sentinel "
            f"{s['nyha_missing_imputation'].get('sentinel')}",
            f"- Numeric: imputed {len(s['numeric_imputation']['imputed'])} column(s) "
            f"(bootstrap from observed values), added {s['numeric_imputation']['was_missing_flags_added']} "
            f"'_was_missing' flag(s), dropped {len(s['numeric_imputation']['dropped_too_few'])} "
            f"column(s) with too few observed values",
        ]
        for col in s["numeric_imputation"]["dropped_too_few"]:
            lines.append(f"  - dropped: `{col}`")
        lines.append(
            f"- Categorical/boolean: filled {len(s['categorical_imputation']['filled_columns'])} "
            f"column(s) with explicit 'Missing' category"
        )

        return "\n".join(lines) + "\n"

"""
Step: preprocess

Turns the full raw dataset into a GAN-ready feature set: combines
medication/condition columns, encodes NYHA via the metadata valueSet,
drops identifiers/datetimes/symptoms, applies Machteld's temporary dummy
imputation for currently-missing labs (toggleable), then a generic final
imputation pass (bootstrap for numerics + missingness flags, "Missing"
category for booleans/categoricals) so no nulls remain.
"""

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
        if "pid" in df.columns:
            print(f"Unique patients (pid): {df['pid'].n_unique()}")

        print("Validating against metadata...")
        t.validate_against_metadata(df, var_meta)

        print("Checking expected non-null pairs...")
        t.report_expected_nonnull_mismatches(df)

        print("Flattening ARRAY[NOMINAL] columns...")
        df = t.flatten_array_columns(df, var_meta)

        print("Checking symptom columns...")
        t.report_symptom_columns(df)

        print("Combining medication columns...")
        df = t.combine_medications(df)

        print("Combining condition columns...")
        df = t.combine_conditions(df)

        print("Encoding NYHA...")
        df = t.encode_nyha(df, var_meta)

        if config.apply_dummy_imputation:
            print("Applying temporary dummy imputation (Machteld's placeholder rules)...")
            df = t.apply_dummy_imputation(df)
        else:
            print("Skipping dummy imputation (apply_dummy_imputation=False).")

        print("Preferring _first/_last numeric variants...")
        df = t.prefer_first_last_numerics(df)

        print("Dropping identifier/datetime columns...")
        df = t.drop_identifiers_and_datetimes(df, var_meta)

        print("Final null cleanup...")
        df = t.impute_nyha_missing(df)
        df = t.impute_numeric_columns(df, var_meta)
        df = t.impute_categorical_and_boolean(df, var_meta)

        remaining_nulls = sum(df[c].null_count() for c in df.columns)
        print(f"Remaining null cells after all imputation: {remaining_nulls}")

        os.makedirs(os.path.dirname(config.preprocessed_output_path), exist_ok=True)
        df.write_parquet(config.preprocessed_output_path)
        print(f"Saved {df.height} rows x {df.width} columns -> {config.preprocessed_output_path}")

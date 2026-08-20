import glob
import os
import re

import numpy as np
import polars as pl

# --- CONFIGURATION ---
# Point this at the local folder where the two UC1 parquet files (from the
# transfer inbox) were copied. They are combined into one frame, keyed on
# pseudo_id, before any of the transforms below run.
INPUT_FOLDER = "/mnt/data/DT4Hnew/uc1_transfer/"
OUTPUT_PATH = "/mnt/data/DT4Hnew/UC1_Preprocessed.parquet"

# Machteld's export had missing lab/vital values that shouldn't be missing
# yet (e.g. troponin isn't wired up until "next week"). This is a temporary
# stand-in so the pipeline can be built and tested now; set to False the
# moment the corrected export arrives and skip this step entirely.
APPLY_DUMMY_IMPUTATION = True

# LOINC answer codes -> NYHA class ordinal (1 = least severe, 4 = most severe)
NYHA_LOINC_MAP = {
    "LA28404-4": 1,
    "LA28405-1": 2,
    "LA28406-9": 3,
    "LA28407-7": 4,
}


def load_and_merge(folder: str) -> pl.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if len(files) != 2:
        raise ValueError(f"Expected exactly 2 parquet files in {folder}, found {len(files)}: {files}")

    a, b = (pl.read_parquet(f) for f in files)
    print(f"--- 📂 MERGING {os.path.basename(files[0])} + {os.path.basename(files[1])} ---")

    # This assumes the two files split by COLUMN (same patients, different
    # feature groups) and joins them on pseudo_id, matching the convention
    # in dataCleaner.py. If they instead split by ROW (different patient
    # batches, same schema), a join is wrong and they should be concat'd.
    shared_cols = (set(a.columns) & set(b.columns)) - {"pseudo_id"}
    shared_ids = set(a["pseudo_id"].to_list()) & set(b["pseudo_id"].to_list())
    if shared_cols:
        print(f"⚠️  WARNING: {len(shared_cols)} column(s) besides pseudo_id appear in BOTH files: "
              f"{sorted(shared_cols)[:10]}{'...' if len(shared_cols) > 10 else ''}")
        print("   This suggests the files may split by ROW (patient batches), not by column.")
        print("   If so, use pl.concat([a, b]) instead of a join -- check with Machteld before trusting this output.")
    print(f"   {a.height} rows in file 1, {b.height} rows in file 2, "
          f"{len(shared_ids)} shared pseudo_id(s) between them.")

    merged = a.join(b, on="pseudo_id", how="full", coalesce=True)
    print(f"Merged frame: {merged.height} rows, {merged.width} columns.")
    return merged


def report_expected_nonnull_mismatches(df: pl.DataFrame) -> None:
    """
    Sanity checks from Machteld's email: certain lab/vital pairs are
    clinically expected to have similar non-null counts (they're ordered
    together / measured together). Large gaps flag an export problem worth
    reporting back, rather than a real clinical pattern.
    """
    pairs = [
        ("lab_results_hdl_value", "lab_results_ldl_value", "expected similar (ordered together)"),
        ("lab_results_potassium_value", "lab_results_sodium_value", "expected similar (ordered together)"),
        ("lab_results_albuminBS_value", "lab_results_ntProBnp_value", "expected similar"),
        ("lab_results_albuminBS_value", "lab_results_crpNonHs_value", "expected similar"),
        ("lab_results_albuminBS_value", "lab_results_hba1c_value", "expected similar"),
        ("vital_signs_heartRate_value", "vital_signs_oxygenSaturation_value", "expected similar, oxygen sat maybe slightly lower"),
    ]

    print("\n--- 🩺 EXPECTED NON-NULL COUNT CHECKS ---")
    for col_a, col_b, note in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            print(f"  (skip) {col_a} / {col_b}: one or both columns not found")
            continue
        n_a = df.height - df[col_a].null_count()
        n_b = df.height - df[col_b].null_count()
        flag = "⚠️ " if n_b == 0 or n_a == 0 or abs(n_a - n_b) / max(n_a, 1) > 0.5 else ""
        print(f"  {flag}{col_a}: {n_a} non-null vs {col_b}: {n_b} non-null ({note})")


def drop_symptom_columns(df: pl.DataFrame) -> pl.DataFrame:
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    print(f"\nDropping {len(symptom_cols)} symptom_* columns (NLP module not yet integrated; all null).")
    return df.drop(symptom_cols)


def combine_medications(df: pl.DataFrame) -> pl.DataFrame:
    """
    med_admins_<X>         / med_requests_<X>          -> med_<X>
    med_admins_history_<X> / med_requests_history_<X>  -> med_<X>_history

    A medication is considered present if either the "admins" or "requests"
    table flags it (matches Machteld's rule: med_admins_diuretics=1 or
    med_requests_diuretics=0 -> med_diuretics=1). Null is treated as "not
    flagged" (False), not "unknown" -- these are presence/absence flags.
    """
    admin_pat = re.compile(r"^med_admins_(?!history_)(.+)$")
    admin_hist_pat = re.compile(r"^med_admins_history_(.+)$")

    med_types = sorted({m.group(1) for c in df.columns if (m := admin_pat.match(c))})
    med_hist_types = sorted({m.group(1) for c in df.columns if (m := admin_hist_pat.match(c))})

    new_cols = []
    drop_cols = []

    for med in med_types:
        present = [c for c in (f"med_admins_{med}", f"med_requests_{med}") if c in df.columns]
        if not present:
            continue
        new_cols.append(
            pl.any_horizontal([pl.col(c).fill_null(False) for c in present]).alias(f"med_{med}")
        )
        drop_cols.extend(present)

    for med in med_hist_types:
        present = [c for c in (f"med_admins_history_{med}", f"med_requests_history_{med}") if c in df.columns]
        if not present:
            continue
        new_cols.append(
            pl.any_horizontal([pl.col(c).fill_null(False) for c in present]).alias(f"med_{med}_history")
        )
        drop_cols.extend(present)

    df = df.with_columns(new_cols)
    df = df.drop([c for c in set(drop_cols) if c in df.columns])
    print(f"Combined medication columns into {len(new_cols)} features "
          f"({len(med_types)} current + {len(med_hist_types)} history).")
    return df


def combine_conditions(df: pl.DataFrame) -> pl.DataFrame:
    """
    conditions_<X>_pre_dc / _pre_adm / _during_pET -> conditions_<X>
    (True if the condition was flagged in any of the three windows.)

    Columns that don't match one of these three suffixes (e.g. the
    heart-failure-specific "..._occurred_prior_to_18_months" column) are
    left untouched, since they're not part of this combination rule.
    """
    suffixes = ("_pre_dc", "_pre_adm", "_during_pET")
    pat = re.compile(r"^conditions_(.+?)(?:_pre_dc|_pre_adm|_during_pET)$")

    base_names = sorted({m.group(1) for c in df.columns if (m := pat.match(c))})

    new_cols = []
    drop_cols = []
    for base in base_names:
        variants = [f"conditions_{base}{suf}" for suf in suffixes if f"conditions_{base}{suf}" in df.columns]
        if not variants:
            continue
        new_cols.append(
            pl.any_horizontal([pl.col(c).fill_null(False) for c in variants]).alias(f"conditions_{base}")
        )
        drop_cols.extend(variants)

    df = df.with_columns(new_cols)
    df = df.drop([c for c in set(drop_cols) if c in df.columns])
    print(f"Combined condition columns into {len(new_cols)} features.")
    return df


def prefer_first_last_numerics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Per Machteld's guidance ("for general processing of numerical values
    please use the ones ending with _first and _last"): if a bare/aggregate
    column exists alongside _first/_last variants of the same measurement,
    drop the bare one and keep only _first/_last.
    """
    first_last = {c for c in df.columns if c.endswith("_first") or c.endswith("_last")}
    bases = {re.sub(r"_(first|last)$", "", c) for c in first_last}
    drop_cols = [c for c in bases if c in df.columns]
    if drop_cols:
        print(f"\nDropping {len(drop_cols)} bare numeric aggregate(s) in favor of _first/_last variants:")
        print(f"  {drop_cols}")
        df = df.drop(drop_cols)
    return df


def encode_nyha(df: pl.DataFrame) -> pl.DataFrame:
    if "nyha_class" not in df.columns:
        return df
    return df.with_columns(pl.col("nyha_class").replace(NYHA_LOINC_MAP, default=None).alias("nyha_class"))


# --- Dummy imputation (TEMPORARY, see APPLY_DUMMY_IMPUTATION above) ---

def _sample_right_skewed(rng: np.random.Generator, low: float, high: float, mean: float | None, n: int) -> np.ndarray:
    """
    Draw n samples in [low, high] with a right-skewed shape (long tail
    toward the high end, "few measurements in the higher ranges" per the
    email) hitting the requested mean. Mean defaults to the range midpoint
    when not specified.
    """
    if mean is None:
        mean = (low + high) / 2
    span = high - low
    target = min(max((mean - low) / span, 1e-3), 1 - 1e-3)
    a = 2.0
    b = a * (1 - target) / target
    return low + rng.beta(a, b, size=n) * span


# (trigger_col, [(target_col, low, high, mean_or_None), ...])
DUMMY_IMPUTATION_RULES = [
    ("lab_results_albuminBS_value", [
        ("lab_results_glucose_value", 0, 60, 7),
        ("lab_results_hba1c%_value", 3, 17, 7),
        ("lab_results_hba1c_value", 15, 167, 50),
        ("lab_results_ntProBnp_value", 100, 8000, 3000),
        ("lab_results_crpNonHs_value", 10, 500, 50),
    ]),
    ("lab_results_hdl_value", [
        ("lab_results_tropTnHs_value", 2, 400, None),
        ("lab_results_ldl_value", 0, 20, 2),
    ]),
    ("lab_results_potassium_value", [
        ("lab_results_sodium_value", 131, 175, 140),
    ]),
    ("vital_signs_heartRate_value", [
        ("vital_signs_oxygenSaturation_value", 85, 99, 96),
    ]),
    ("electrocardiographs_ecg_qrs_duration_pET", [
        ("electrocardiographs_ecg_qrs_axis_pET", 30, 120, None),
    ]),
]


def _resolve_variants(df: pl.DataFrame, base_col: str) -> list[str]:
    """
    Numeric measurements may show up as a bare column, or as _first/_last
    variants (or both, until prefer_first_last_numerics prunes the bare
    one). Resolve to whichever of these actually exist so imputation
    doesn't silently miss the columns downstream code will actually use.
    """
    candidates = [base_col, f"{base_col}_first", f"{base_col}_last"]
    return [c for c in candidates if c in df.columns]


def apply_dummy_imputation(df: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    print("\n--- 🧪 APPLYING TEMPORARY DUMMY IMPUTATION (remove once real export lands) ---")

    for trigger_base, targets in DUMMY_IMPUTATION_RULES:
        trigger_cols = _resolve_variants(df, trigger_base)
        if not trigger_cols:
            print(f"  (skip) trigger column '{trigger_base}' (or _first/_last) not found")
            continue

        trigger_present = df[trigger_cols[0]].is_not_null()
        for c in trigger_cols[1:]:
            trigger_present = trigger_present | df[c].is_not_null()

        for target_base, low, high, mean in targets:
            target_cols = _resolve_variants(df, target_base)
            if not target_cols:
                print(f"  (skip) target column '{target_base}' (or _first/_last) not found")
                continue

            for target_col in target_cols:
                needs_fill = trigger_present & df[target_col].is_null()
                n_fill = int(needs_fill.sum())
                if n_fill == 0:
                    continue

                values = _sample_right_skewed(rng, low, high, mean, n_fill)
                filled = df[target_col].to_numpy(zero_copy_only=False).astype(float)
                filled[needs_fill.to_numpy()] = values
                df = df.with_columns(pl.Series(target_col, filled))
                print(f"  Filled {n_fill} value(s) in '{target_col}' (triggered by '{trigger_base}' present).")

    return df


def preprocess():
    df = load_and_merge(INPUT_FOLDER)

    total_patients = df["pseudo_id"].n_unique()
    print(f"Unique patients: {total_patients}")

    report_expected_nonnull_mismatches(df)

    df = drop_symptom_columns(df)
    df = combine_medications(df)
    df = combine_conditions(df)
    df = encode_nyha(df)

    # Imputation must run before pruning bare aggregates: it needs to see
    # (and fill) whichever of the bare/_first/_last variants exist.
    if APPLY_DUMMY_IMPUTATION:
        df = apply_dummy_imputation(df)
    else:
        print("\nSkipping dummy imputation (APPLY_DUMMY_IMPUTATION=False).")

    df = prefer_first_last_numerics(df)

    df.write_parquet(OUTPUT_PATH)
    print(f"\n✅ SUCCESS: Saved {df.height} rows x {df.width} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()

import json
import os
import re

import numpy as np
import polars as pl

# --- CONFIGURATION ---
# Point this at the single resolved parquet file for the full dataset --
# i.e. whatever prepareTestData.py produces (concatenated part-*.parquet
# files), saved locally. Confirmed against the real
# for_repo/DT4H_Column_Analysis.json: the ID column is "pid" (not
# "pseudo_id"), and the file is already one row per patient -- no
# join/merge step is needed here, unlike the two-file assumption in the
# original email.
INPUT_PATH = "/mnt/data/DT4Hnew/UC1_Resolved_Full.parquet"
OUTPUT_PATH = "/mnt/data/DT4Hnew/UC1_Preprocessed.parquet"

# The official DataTools4Heart/AI4HF feature-set schema, committed to the
# repo (it's schema info, not patient data) -- kept alongside this script
# so the declared dataType/valueSet per variable drives the pipeline
# instead of guessing from column-name patterns.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(REPO_ROOT, "for_repo", "metadata.json")

# Machteld's export had missing lab/vital values that shouldn't be missing
# yet (e.g. troponin isn't wired up until "next week"). This is a temporary
# stand-in so the pipeline can be built and tested now; set to False the
# moment the corrected export arrives and skip this step entirely.
APPLY_DUMMY_IMPUTATION = True


def load_dataset(path: str) -> pl.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run prepareTestData.py first (or point INPUT_PATH at it).")
    df = pl.read_parquet(path)
    print(f"--- 📂 LOADED {os.path.basename(path)}: {df.height} rows, {df.width} columns ---")
    return df


def load_variable_metadata(path: str) -> dict:
    """
    Returns {variable_name: full_variable_dict} from the official
    DataTools4Heart/AI4HF feature-set schema (metadata.json), covering
    baseVariables + features + outcomes -- 528 entries, one per column in
    the current export. Each entry has at least "dataType" (IDENTIFIER /
    DATETIME / NOMINAL / NUMERIC / BOOLEAN / ARRAY[NOMINAL]) and often a
    "valueSet" for coded NOMINAL fields.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- metadata.json should be committed under for_repo/.")
    with open(path) as f:
        raw = json.load(f)
    entry = raw["entries"][0]
    all_vars = entry["baseVariables"] + entry["features"] + entry["outcomes"]
    return {v["name"]: v for v in all_vars}


def validate_against_metadata(df: pl.DataFrame, var_meta: dict) -> None:
    """QA check: flag any drift between the data's actual columns and the declared schema."""
    df_cols = set(df.columns)
    meta_cols = set(var_meta.keys())
    missing_from_data = meta_cols - df_cols
    missing_from_meta = df_cols - meta_cols

    print("\n--- 🔍 METADATA vs DATA COLUMN CHECK ---")
    print(f"  {len(meta_cols)} declared in metadata.json, {len(df_cols)} present in data, "
          f"{len(meta_cols & df_cols)} match.")
    if missing_from_data:
        print(f"  ⚠️  {len(missing_from_data)} declared but not in data: {sorted(missing_from_data)[:10]}")
    if missing_from_meta:
        print(f"  ⚠️  {len(missing_from_meta)} in data but not declared: {sorted(missing_from_meta)[:10]}")


def flatten_array_columns(df: pl.DataFrame, var_meta: dict) -> pl.DataFrame:
    """
    ARRAY[NOMINAL] columns (e.g. ecg_type_of_rhythms) store a List(String)
    per cell -- GAN training needs scalars, so this takes the first element
    of each list (empty list or null both become null).
    """
    array_cols = [name for name, v in var_meta.items() if v.get("dataType") == "ARRAY[NOMINAL]" and name in df.columns]
    if not array_cols:
        return df
    print(f"\nFlattening {len(array_cols)} ARRAY[NOMINAL] column(s) to scalar (first list element): {array_cols}")
    return df.with_columns([pl.col(c).list.first().alias(c) for c in array_cols])


def drop_identifiers_and_datetimes(df: pl.DataFrame, var_meta: dict) -> pl.DataFrame:
    """
    IDENTIFIER columns (pid, encounterId) are direct identifiers and must
    never feed a synthesis model. DATETIME columns here are either pipeline
    bookkeeping (eventTime/exitTime/referenceTimePoint -- cohort window
    boundaries, not clinical data) or exact admission/discharge dates whose
    clinically useful signal is already captured by the derived
    encounters_admissionYear (NOMINAL) and encounters_lengthOfStay
    (NUMERIC) features. Keeping the exact dates would add reidentification
    risk and near-unique noise for no real modeling benefit, so all
    DATETIME columns are dropped too.
    """
    drop_cols = [name for name, v in var_meta.items()
                 if v.get("dataType") in ("IDENTIFIER", "DATETIME") and name in df.columns]
    if drop_cols:
        print(f"\nDropping {len(drop_cols)} IDENTIFIER/DATETIME column(s): {drop_cols}")
        df = df.drop(drop_cols)
    return df


def build_nyha_map(var_meta: dict, column: str) -> dict:
    """
    Derives the LOINC-code -> ordinal-severity mapping from metadata.json's
    valueSet (display strings "Class-I".."Class-IV") instead of a
    hardcoded dict, so it stays correct if the underlying codes change.
    """
    concepts = var_meta.get(column, {}).get("valueSet", {}).get("concept", [])
    mapping = {}
    for c in concepts:
        m = re.search(r"Class-([IV]+)$", c["display"])
        if not m:
            continue
        roman = m.group(1)
        ordinal = {"I": 1, "II": 2, "III": 3, "IV": 4}.get(roman)
        if ordinal is not None:
            mapping[c["code"]] = ordinal
    return mapping


def _first_existing(df: pl.DataFrame, base_col: str) -> str | None:
    """The real export has no bare column, only _first/_last/_min/_max/_avg/
    _stddev variants -- resolve to whichever exists, preferring _first."""
    for candidate in (base_col, f"{base_col}_first", f"{base_col}_last"):
        if candidate in df.columns:
            return candidate
    return None


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
    for base_a, base_b, note in pairs:
        col_a = _first_existing(df, base_a)
        col_b = _first_existing(df, base_b)
        if col_a is None or col_b is None:
            print(f"  (skip) {base_a} / {base_b}: one or both not found (checked bare/_first/_last)")
            continue
        n_a = df.height - df[col_a].null_count()
        n_b = df.height - df[col_b].null_count()
        flag = "⚠️ " if n_b == 0 or n_a == 0 or abs(n_a - n_b) / max(n_a, 1) > 0.5 else ""
        print(f"  {flag}{col_a}: {n_a} non-null vs {col_b}: {n_b} non-null ({note})")


def drop_symptom_columns(df: pl.DataFrame) -> pl.DataFrame:
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    print(f"\nDropping {len(symptom_cols)} symptom_* columns (NLP module not yet integrated; all null).")
    return df.drop(symptom_cols)


def _strip_any_suffix(name: str) -> str:
    return name[: -len("_any")] if name.endswith("_any") else name


def combine_medications(df: pl.DataFrame) -> pl.DataFrame:
    """
    med_admins_<X>_any         / med_requests_<X>_any          -> med_<X>
    med_admins_history_<X>_any / med_requests_history_<X>_any  -> med_<X>_history

    A medication is considered present if either the "admins" or "requests"
    table flags it (matches Machteld's rule: med_admins_diuretics=1 or
    med_requests_diuretics=0 -> med_diuretics=1). Null is treated as "not
    flagged" (False), not "unknown" -- these are presence/absence flags.
    The trailing "_any" in the real column names is dropped from the
    combined output name since it's redundant once combined.
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
            pl.any_horizontal([pl.col(c).fill_null(False) for c in present]).alias(f"med_{_strip_any_suffix(med)}")
        )
        drop_cols.extend(present)

    for med in med_hist_types:
        present = [c for c in (f"med_admins_history_{med}", f"med_requests_history_{med}") if c in df.columns]
        if not present:
            continue
        new_cols.append(
            pl.any_horizontal([pl.col(c).fill_null(False) for c in present]).alias(f"med_{_strip_any_suffix(med)}_history")
        )
        drop_cols.extend(present)

    df = df.with_columns(new_cols)
    df = df.drop([c for c in set(drop_cols) if c in df.columns])
    print(f"Combined medication columns into {len(new_cols)} features "
          f"({len(med_types)} current + {len(med_hist_types)} history).")
    return df


def combine_conditions(df: pl.DataFrame) -> pl.DataFrame:
    """
    conditions_<X>_pre_dc_any / _pre_adm_any / _during_pET_any -> conditions_<X>
    (True if the condition was flagged in any of the three windows.)

    Columns that don't match one of these three suffixes (e.g. the
    heart-failure-specific "..._occurred_prior_to_18_months_any" and
    "..._hf_within_18mo_any" columns, or the numeric
    "conditions_heartFailure_timeFromEarliest_first") are left untouched,
    since they're not part of this combination rule.
    """
    suffixes = ("_pre_dc_any", "_pre_adm_any", "_during_pET_any")
    pat = re.compile(r"^conditions_(.+?)(?:_pre_dc_any|_pre_adm_any|_during_pET_any)$")

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


# Every numeric measurement in the real export shows up as up to 6 variants:
# _first, _last, _min, _max, _avg, _stddev (no bare column). Machteld's
# guidance is to use only _first/_last for general processing.
OTHER_NUMERIC_SUFFIXES = ("_min", "_max", "_avg", "_stddev")


def prefer_first_last_numerics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Wherever a measurement has _first/_last variants, drop its bare column
    (if any) and its _min/_max/_avg/_stddev siblings, keeping only
    _first/_last.
    """
    first_last = {c for c in df.columns if c.endswith("_first") or c.endswith("_last")}
    bases = {re.sub(r"_(first|last)$", "", c) for c in first_last}

    drop_cols = []
    for base in bases:
        drop_cols.append(base)  # bare aggregate, if present
        drop_cols.extend(f"{base}{suf}" for suf in OTHER_NUMERIC_SUFFIXES)
    drop_cols = [c for c in drop_cols if c in df.columns]

    if drop_cols:
        print(f"\nDropping {len(drop_cols)} numeric aggregate column(s) in favor of _first/_last variants "
              f"(bare/_min/_max/_avg/_stddev).")
        df = df.drop(drop_cols)
    return df


NYHA_COLUMN = "nyha_nyha_pET"


def encode_nyha(df: pl.DataFrame, var_meta: dict) -> pl.DataFrame:
    if NYHA_COLUMN not in df.columns:
        print(f"\n(skip) NYHA column '{NYHA_COLUMN}' not found.")
        return df
    nyha_map = build_nyha_map(var_meta, NYHA_COLUMN)
    print(f"\nEncoding {NYHA_COLUMN} via metadata valueSet: {nyha_map}")
    return df.with_columns(pl.col(NYHA_COLUMN).replace(nyha_map, default=None).alias(NYHA_COLUMN))


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
    df = load_dataset(INPUT_PATH)
    var_meta = load_variable_metadata(METADATA_PATH)

    if "pid" in df.columns:
        print(f"Unique patients (pid): {df['pid'].n_unique()}")

    validate_against_metadata(df, var_meta)
    report_expected_nonnull_mismatches(df)

    df = flatten_array_columns(df, var_meta)
    df = drop_symptom_columns(df)
    df = combine_medications(df)
    df = combine_conditions(df)
    df = encode_nyha(df, var_meta)

    # Imputation must run before pruning bare aggregates: it needs to see
    # (and fill) whichever of the bare/_first/_last variants exist.
    if APPLY_DUMMY_IMPUTATION:
        df = apply_dummy_imputation(df)
    else:
        print("\nSkipping dummy imputation (APPLY_DUMMY_IMPUTATION=False).")

    df = prefer_first_last_numerics(df)
    df = drop_identifiers_and_datetimes(df, var_meta)

    df.write_parquet(OUTPUT_PATH)
    print(f"\n✅ SUCCESS: Saved {df.height} rows x {df.width} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()

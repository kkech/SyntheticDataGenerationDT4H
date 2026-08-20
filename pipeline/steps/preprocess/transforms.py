"""
UC1 feature-set transforms, driven by the official DataTools4Heart/AI4HF
schema (metadata.json) rather than guessing types from column-name
patterns. Each function is independently testable and takes/returns a
plain polars DataFrame.
"""

import json
import os
import re

import numpy as np
import polars as pl

NYHA_COLUMN = "nyha_nyha_pET"

# Numeric measurements show up as up to 6 variants: _first, _last, _min,
# _max, _avg, _stddev (no bare column). Only _first/_last are kept for
# general processing.
OTHER_NUMERIC_SUFFIXES = ("_min", "_max", "_avg", "_stddev")

# A category value is only reported/kept distinct if at least this many
# rows share it -- see NUMERIC_MIN_NONNULL below for the numeric analog.
NUMERIC_MIN_NONNULL = 5

# NYHA is ordinal (1-4) by the time imputation runs, not a free numeric
# value -- a missing assessment gets this sentinel instead of a
# bootstrap-sampled (fabricated) severity class.
NYHA_MISSING_SENTINEL = 0


# --- metadata ---

def load_variable_metadata(path: str) -> dict:
    """
    Returns {variable_name: full_variable_dict} from the official
    DataTools4Heart/AI4HF feature-set schema, covering baseVariables +
    features + outcomes. Each entry has "dataType" (IDENTIFIER / DATETIME /
    NOMINAL / NUMERIC / BOOLEAN / ARRAY[NOMINAL]) and often a "valueSet"
    for coded NOMINAL fields.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- metadata.json should be committed under for_repo/.")
    with open(path) as f:
        raw = json.load(f)
    entry = raw["entries"][0]
    all_vars = entry["baseVariables"] + entry["features"] + entry["outcomes"]
    return {v["name"]: v for v in all_vars}


def validate_against_metadata(df: pl.DataFrame, var_meta: dict) -> dict:
    """QA check: flags any drift between the declared schema and the actual data columns."""
    df_cols = set(df.columns)
    meta_cols = set(var_meta.keys())
    missing_from_data = sorted(meta_cols - df_cols)
    missing_from_meta = sorted(df_cols - meta_cols)

    print(f"  {len(meta_cols)} declared in metadata.json, {len(df_cols)} present in data, "
          f"{len(meta_cols & df_cols)} match.")
    if missing_from_data:
        print(f"  ⚠️  {len(missing_from_data)} declared but not in data: {missing_from_data[:10]}")
    if missing_from_meta:
        print(f"  ⚠️  {len(missing_from_meta)} in data but not declared: {missing_from_meta[:10]}")

    return {
        "declared_in_metadata": len(meta_cols),
        "present_in_data": len(df_cols),
        "matched": len(meta_cols & df_cols),
        "declared_but_missing_from_data": missing_from_data,
        "in_data_but_not_declared": missing_from_meta,
    }


# --- data-quality QA (from Machteld's email) ---

def _first_existing(df: pl.DataFrame, base_col: str) -> str | None:
    for candidate in (base_col, f"{base_col}_first", f"{base_col}_last"):
        if candidate in df.columns:
            return candidate
    return None


EXPECTED_NONNULL_PAIRS = [
    ("lab_results_hdl_value", "lab_results_ldl_value", "expected similar (ordered together)"),
    ("lab_results_potassium_value", "lab_results_sodium_value", "expected similar (ordered together)"),
    ("lab_results_albuminBS_value", "lab_results_ntProBnp_value", "expected similar"),
    ("lab_results_albuminBS_value", "lab_results_crpNonHs_value", "expected similar"),
    ("lab_results_albuminBS_value", "lab_results_hba1c_value", "expected similar"),
    ("vital_signs_heartRate_value", "vital_signs_oxygenSaturation_value", "expected similar, oxygen sat maybe slightly lower"),
]


def report_expected_nonnull_mismatches(df: pl.DataFrame) -> list[dict]:
    """
    Sanity checks from Machteld's email: certain lab/vital pairs are
    clinically expected to have similar non-null counts. Large gaps flag
    an export problem worth reporting back, not a real clinical pattern.
    """
    results = []
    for base_a, base_b, note in EXPECTED_NONNULL_PAIRS:
        col_a = _first_existing(df, base_a)
        col_b = _first_existing(df, base_b)
        if col_a is None or col_b is None:
            print(f"  (skip) {base_a} / {base_b}: one or both not found (checked bare/_first/_last)")
            results.append({"col_a": base_a, "col_b": base_b, "note": note, "skipped": True})
            continue
        n_a = df.height - df[col_a].null_count()
        n_b = df.height - df[col_b].null_count()
        mismatch = n_b == 0 or n_a == 0 or abs(n_a - n_b) / max(n_a, 1) > 0.5
        flag = "⚠️ " if mismatch else ""
        print(f"  {flag}{col_a}: {n_a} non-null vs {col_b}: {n_b} non-null ({note})")
        results.append({
            "col_a": col_a, "n_a": n_a, "col_b": col_b, "n_b": n_b, "note": note, "mismatch": mismatch,
        })
    return results


# --- type-driven cleanup ---

def flatten_array_columns(df: pl.DataFrame, var_meta: dict) -> tuple[pl.DataFrame, dict]:
    """ARRAY[NOMINAL] columns store a List(String) per cell -- flattened to
    scalar (first list element) since GAN training needs scalars."""
    array_cols = [name for name, v in var_meta.items()
                  if v.get("dataType") == "ARRAY[NOMINAL]" and name in df.columns]
    if not array_cols:
        return df, {"flattened": []}
    print(f"  Flattening {len(array_cols)} ARRAY[NOMINAL] column(s) to scalar: {array_cols}")
    df = df.with_columns([pl.col(c).list.first().alias(c) for c in array_cols])
    return df, {"flattened": array_cols}


def report_symptom_columns(df: pl.DataFrame) -> dict:
    """
    Per Machteld: symptom_* columns are currently all-False (NLP module
    not yet integrated) but should stay IN the data rather than be
    dropped -- if the NLP module comes online, these columns gain real
    signal on a future data refresh with no script change needed. This
    just reports their current state; nothing is removed.
    """
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    constant = [c for c in symptom_cols if df[c].drop_nulls().n_unique() <= 1]
    if symptom_cols:
        print(f"  {len(symptom_cols)} symptom_* column(s) present, {len(constant)} currently constant "
              f"(NLP module not yet integrated) -- kept in the data, not dropped.")
    return {"count": len(symptom_cols), "currently_constant": len(constant), "dropped": False}


def drop_identifiers_and_datetimes(df: pl.DataFrame, var_meta: dict) -> tuple[pl.DataFrame, dict]:
    """
    IDENTIFIER columns (pid, encounterId) are direct identifiers and must
    never feed a synthesis model. DATETIME columns are either pipeline
    bookkeeping (cohort window boundaries, not clinical data) or exact
    admission/discharge dates whose clinically useful signal is already
    captured by the derived admissionYear/lengthOfStay features -- keeping
    exact dates adds reidentification risk for no modeling benefit.
    """
    drop_cols = [name for name, v in var_meta.items()
                 if v.get("dataType") in ("IDENTIFIER", "DATETIME") and name in df.columns]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} IDENTIFIER/DATETIME column(s): {drop_cols}")
        df = df.drop(drop_cols)
    return df, {"dropped": drop_cols}


# A string column this close to 100% unique behaves like a raw identifier
# regardless of how metadata.json tags it -- e.g.
# patient_demographics_sourceIdentifier is declared NOMINAL, not
# IDENTIFIER, but is 100% unique per row in the real export.
NEAR_UNIQUE_THRESHOLD = 0.9


def drop_near_unique_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Safety net beyond drop_identifiers_and_datetimes: catches
    identifier-like string columns the declared type missed. Only
    triggers on near-total uniqueness (>90% of rows), so it won't catch
    ordinary high-cardinality categoricals -- just ones that are
    effectively a row-level identifier.
    """
    height = df.height
    drop_cols = []
    for col in df.columns:
        if df[col].dtype != pl.String:
            continue
        n_unique = df[col].n_unique()
        if height and n_unique / height > NEAR_UNIQUE_THRESHOLD:
            drop_cols.append(col)

    if drop_cols:
        print(f"  Dropping {len(drop_cols)} near-unique identifier-like column(s) "
              f"not caught by declared type: {drop_cols}")
        df = df.drop(drop_cols)
    return df, {"dropped": drop_cols}


# --- medication / condition combining (Machteld's email) ---

def _strip_any_suffix(name: str) -> str:
    return name[: -len("_any")] if name.endswith("_any") else name


def combine_medications(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    med_admins_<X>_any         / med_requests_<X>_any          -> med_<X>
    med_admins_history_<X>_any / med_requests_history_<X>_any  -> med_<X>_history

    A medication is considered present if either the "admins" or "requests"
    table flags it. Null is treated as "not flagged" (False), not
    "unknown" -- these are presence/absence flags.
    """
    admin_pat = re.compile(r"^med_admins_(?!history_)(.+)$")
    admin_hist_pat = re.compile(r"^med_admins_history_(.+)$")

    med_types = sorted({m.group(1) for c in df.columns if (m := admin_pat.match(c))})
    med_hist_types = sorted({m.group(1) for c in df.columns if (m := admin_hist_pat.match(c))})

    new_cols = []
    drop_cols = []
    features_created = []

    for med in med_types:
        present = [c for c in (f"med_admins_{med}", f"med_requests_{med}") if c in df.columns]
        if not present:
            continue
        feature_name = f"med_{_strip_any_suffix(med)}"
        new_cols.append(pl.any_horizontal([pl.col(c).fill_null(False) for c in present]).alias(feature_name))
        drop_cols.extend(present)
        features_created.append(feature_name)

    for med in med_hist_types:
        present = [c for c in (f"med_admins_history_{med}", f"med_requests_history_{med}") if c in df.columns]
        if not present:
            continue
        feature_name = f"med_{_strip_any_suffix(med)}_history"
        new_cols.append(pl.any_horizontal([pl.col(c).fill_null(False) for c in present]).alias(feature_name))
        drop_cols.extend(present)
        features_created.append(feature_name)

    df = df.with_columns(new_cols)
    df = df.drop([c for c in set(drop_cols) if c in df.columns])
    print(f"  Combined medication columns into {len(new_cols)} feature(s) "
          f"({len(med_types)} current + {len(med_hist_types)} history).")
    return df, {"features_created": features_created, "source_columns_dropped": len(set(drop_cols))}


def combine_conditions(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    conditions_<X>_pre_dc_any / _pre_adm_any / _during_pET_any -> conditions_<X>
    (True if the condition was flagged in any of the three windows.)
    """
    suffixes = ("_pre_dc_any", "_pre_adm_any", "_during_pET_any")
    pat = re.compile(r"^conditions_(.+?)(?:_pre_dc_any|_pre_adm_any|_during_pET_any)$")

    base_names = sorted({m.group(1) for c in df.columns if (m := pat.match(c))})

    new_cols = []
    drop_cols = []
    features_created = []
    for base in base_names:
        variants = [f"conditions_{base}{suf}" for suf in suffixes if f"conditions_{base}{suf}" in df.columns]
        if not variants:
            continue
        feature_name = f"conditions_{base}"
        new_cols.append(pl.any_horizontal([pl.col(c).fill_null(False) for c in variants]).alias(feature_name))
        drop_cols.extend(variants)
        features_created.append(feature_name)

    df = df.with_columns(new_cols)
    df = df.drop([c for c in set(drop_cols) if c in df.columns])
    print(f"  Combined condition columns into {len(new_cols)} feature(s).")
    return df, {"features_created": features_created, "source_columns_dropped": len(set(drop_cols))}


def prefer_first_last_numerics(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Wherever a measurement has _first/_last variants, drop its bare
    column (if any) and its _min/_max/_avg/_stddev siblings."""
    first_last = {c for c in df.columns if c.endswith("_first") or c.endswith("_last")}
    bases = {re.sub(r"_(first|last)$", "", c) for c in first_last}

    drop_cols = []
    for base in bases:
        drop_cols.append(base)
        drop_cols.extend(f"{base}{suf}" for suf in OTHER_NUMERIC_SUFFIXES)
    drop_cols = [c for c in drop_cols if c in df.columns]

    if drop_cols:
        print(f"  Dropping {len(drop_cols)} numeric aggregate column(s) "
              f"(bare/_min/_max/_avg/_stddev) in favor of _first/_last.")
        df = df.drop(drop_cols)
    return df, {"dropped": drop_cols}


def build_nyha_map(var_meta: dict, column: str = NYHA_COLUMN) -> dict:
    """Derives the LOINC-code -> ordinal-severity mapping from
    metadata.json's valueSet ("Class-I".."Class-IV") instead of a
    hardcoded dict, so it stays correct if the underlying codes change."""
    concepts = var_meta.get(column, {}).get("valueSet", {}).get("concept", [])
    roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4}
    mapping = {}
    for c in concepts:
        m = re.search(r"Class-([IV]+)$", c["display"])
        if m and m.group(1) in roman_to_int:
            mapping[c["code"]] = roman_to_int[m.group(1)]
    return mapping


def encode_nyha(df: pl.DataFrame, var_meta: dict) -> tuple[pl.DataFrame, dict]:
    if NYHA_COLUMN not in df.columns:
        print(f"  (skip) NYHA column '{NYHA_COLUMN}' not found.")
        return df, {"skipped": True}
    nyha_map = build_nyha_map(var_meta)
    print(f"  Encoding {NYHA_COLUMN} via metadata valueSet: {nyha_map}")
    df = df.with_columns(pl.col(NYHA_COLUMN).replace(nyha_map, default=None).alias(NYHA_COLUMN))
    return df, {"skipped": False, "map": nyha_map}


# --- temporary dummy imputation (Machteld's placeholder rules) ---

def _sample_right_skewed(rng: np.random.Generator, low: float, high: float, mean: float | None, n: int) -> np.ndarray:
    """Draw n samples in [low, high] with a right-skewed shape hitting the
    requested mean (defaults to the range midpoint)."""
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
    candidates = [base_col, f"{base_col}_first", f"{base_col}_last"]
    return [c for c in candidates if c in df.columns]


def apply_dummy_imputation(df: pl.DataFrame, seed: int = 0) -> tuple[pl.DataFrame, dict]:
    """
    TEMPORARY: fills specific lab/vital columns with placeholder values
    per Machteld's clinically-motivated rules, wherever the given trigger
    column is present but the target is null. Remove/disable
    (config.apply_dummy_imputation = False) once the corrected export
    with real values for these fields arrives.
    """
    rng = np.random.default_rng(seed)
    fills = []
    skipped = []

    for trigger_base, targets in DUMMY_IMPUTATION_RULES:
        trigger_cols = _resolve_variants(df, trigger_base)
        if not trigger_cols:
            print(f"  (skip) trigger column '{trigger_base}' (or _first/_last) not found")
            skipped.append({"reason": "trigger not found", "trigger": trigger_base})
            continue

        trigger_present = df[trigger_cols[0]].is_not_null()
        for c in trigger_cols[1:]:
            trigger_present = trigger_present | df[c].is_not_null()

        for target_base, low, high, mean in targets:
            target_cols = _resolve_variants(df, target_base)
            if not target_cols:
                print(f"  (skip) target column '{target_base}' (or _first/_last) not found")
                skipped.append({"reason": "target not found", "trigger": trigger_base, "target": target_base})
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
                fills.append({"target": target_col, "trigger": trigger_base, "n_filled": n_fill})

    return df, {"fills": fills, "skipped": skipped}


# --- final null cleanup (generic, by declared type) ---

def impute_nyha_missing(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    if NYHA_COLUMN not in df.columns:
        return df, {"filled": 0}
    n_missing = df[NYHA_COLUMN].null_count()
    if n_missing:
        print(f"  Filling {n_missing} missing '{NYHA_COLUMN}' value(s) with sentinel "
              f"{NYHA_MISSING_SENTINEL} ('not assessed', kept distinct from real classes 1-4).")
        df = df.with_columns(pl.col(NYHA_COLUMN).fill_null(NYHA_MISSING_SENTINEL))
    return df, {"filled": n_missing, "sentinel": NYHA_MISSING_SENTINEL}


def impute_numeric_columns(df: pl.DataFrame, var_meta: dict, seed: int = 0) -> tuple[pl.DataFrame, dict]:
    """
    Nulls in NUMERIC columns are imputed by bootstrap-sampling (with
    replacement) from that column's own observed values -- preserves the
    real empirical distribution's shape without assuming a parametric
    form. A companion "<col>_was_missing" boolean flag is added first,
    since missingness itself may be clinically meaningful. Columns with
    fewer than NUMERIC_MIN_NONNULL observed values are dropped entirely.
    """
    rng = np.random.default_rng(seed)
    numeric_cols = [name for name, v in var_meta.items()
                    if v.get("dataType") == "NUMERIC" and name in df.columns]

    flag_cols = []
    drop_cols = []
    imputed_cols = []

    for col in numeric_cols:
        series = df[col]
        null_mask = series.is_null()
        n_null = int(null_mask.sum())
        if n_null == 0:
            continue

        non_null = series.drop_nulls()
        if non_null.len() < NUMERIC_MIN_NONNULL:
            drop_cols.append(col)
            continue

        flag_cols.append(null_mask.alias(f"{col}_was_missing"))

        sampled = rng.choice(non_null.to_numpy(), size=n_null, replace=True)
        filled = series.to_numpy(zero_copy_only=False).astype(float).copy()
        filled[null_mask.to_numpy()] = sampled
        df = df.with_columns(pl.Series(col, filled))
        imputed_cols.append({"column": col, "n_filled": n_null, "n_observed": non_null.len()})

    if flag_cols:
        df = df.with_columns(flag_cols)
    if drop_cols:
        df = df.drop(drop_cols)

    print(f"  Imputed {len(imputed_cols)} numeric column(s), added {len(flag_cols)} '_was_missing' flag(s), "
          f"dropped {len(drop_cols)} column(s) with fewer than {NUMERIC_MIN_NONNULL} observed values.")
    return df, {"imputed": imputed_cols, "was_missing_flags_added": len(flag_cols), "dropped_too_few": drop_cols}


def impute_categorical_and_boolean(df: pl.DataFrame, var_meta: dict) -> tuple[pl.DataFrame, dict]:
    """
    Nulls in BOOLEAN/NOMINAL/ARRAY[NOMINAL] columns become an explicit
    "Missing" category, consistent with how dpSyntGAN.py already treats
    categorical nulls downstream. Boolean columns become 3-valued string
    categories (True/False/Missing). NYHA is handled separately.
    """
    candidate_types = ("BOOLEAN", "NOMINAL", "ARRAY[NOMINAL]")
    cols = [name for name, v in var_meta.items()
            if v.get("dataType") in candidate_types and name in df.columns and name != NYHA_COLUMN]

    filled = []
    for col in cols:
        if df[col].null_count() == 0:
            continue
        filled.append(col)
        df = df.with_columns(pl.col(col).cast(pl.String).fill_null("Missing").alias(col))

    print(f"  Filled {len(filled)} boolean/categorical column(s) with an explicit 'Missing' category.")
    return df, {"filled_columns": filled}

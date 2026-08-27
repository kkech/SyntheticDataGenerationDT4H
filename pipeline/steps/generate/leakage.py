"""
Verbatim-record leakage check.

A generative model can memorise and reproduce its training rows. For a
clinical dataset intended for public release that is the failure mode
that matters most: a synthetic record identical to a real patient's
record is that patient's record, however it was produced, and no amount
of aggregate fidelity makes it acceptable.

This is a necessary check, not a sufficient one. It detects only EXACT
duplicates. It does not detect near-duplicates, nor does it bound
membership-inference risk -- a record differing in one decimal place is
still a re-identification risk and will not be caught here. A full
privacy assessment for publication needs distance-to-closest-record and
membership-inference analysis on top of this.
"""

import hashlib

import pandas as pd


def _normalize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Cast every numeric column to float64 before stringification.

    Without this, the check is defeated by dtype drift alone: a value
    stored as int64 stringifies to "3" while the same value coming back
    from a synthesizer as float64 stringifies to "3.0", so a row that is
    a verbatim copy of a training record hashes differently and is
    reported as clean. That is the worst possible direction for a safety
    check to fail in, and it is invisible -- the count is simply 0.

    Booleans are left alone: they are numeric to pandas, but the real
    frame spells them as strings ("true"/"false") and the case-alignment
    step upstream matches those spellings; turning True into "1.0" here
    would break that match instead of fixing one.

    Precision note: float64 cannot represent integers above 2^53 exactly,
    so two distinct huge integers could collide. That direction is safe
    (it can only over-report leakage, never hide it), and no such column
    survives preprocessing.
    """
    out = {}
    for c in frame.columns:
        col = frame[c]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            try:
                out[c] = pd.Series(col.to_numpy(dtype="float64", na_value=float("nan")),
                                   index=frame.index)
                continue
            except (TypeError, ValueError):
                pass  # exotic numeric dtype: fall back to the raw values
        out[c] = col
    return pd.DataFrame(out, index=frame.index)[list(frame.columns)]


def _row_hashes(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Stable per-row hash over a fixed column order.

    Both frames go through the SAME numeric normalization, so equality of
    the hash means equality of the values, not of the dtypes.
    """
    # Missing values must hash as a VALUE, not blow up the join: pandas
    # leaves NaN/NA as a float under astype(str), which the join rejects.
    # Both frames get the same token, so a missing cell matches a missing
    # cell and nothing else.
    ordered = _normalize_numeric(df[columns]).astype(str).fillna("<NA>")
    joined = ordered.agg("\x1f".join, axis=1)  # unit separator: not valid in the data
    return joined.map(lambda s: hashlib.sha256(s.encode()).hexdigest())


def check_exact_duplicates(synthetic: pd.DataFrame, real: pd.DataFrame) -> dict:
    """
    Counts synthetic rows that exactly reproduce a training row.

    Compares only on columns common to both, in a fixed order, with
    numeric columns normalized to float64 and then stringified, so dtype
    differences between the real frame and a synthesizer's output ("3"
    vs "3.0") cannot mask a genuine match.
    """
    common = [c for c in real.columns if c in synthetic.columns]
    if not common:
        return {"checked": False, "reason": "no overlapping columns"}
    if synthetic.empty:  # .agg("\x1f".join, axis=1) degenerates on 0 rows
        return {"checked": True, "columns_compared": len(common),
                "synthetic_rows": 0, "exact_duplicates_of_training_rows": 0,
                "exact_duplicate_rate": 0.0, "distinct_training_rows_reproduced": 0,
                "synthetic_duplicate_rows_within_output": 0}

    real_hashes = set(_row_hashes(real, common))
    synth_hashes = _row_hashes(synthetic, common)

    leaked_mask = synth_hashes.isin(real_hashes)
    n_leaked = int(leaked_mask.sum())

    return {
        "checked": True,
        "columns_compared": len(common),
        "synthetic_rows": int(len(synthetic)),
        "exact_duplicates_of_training_rows": n_leaked,
        "exact_duplicate_rate": round(n_leaked / max(len(synthetic), 1), 6),
        "distinct_training_rows_reproduced": int(synth_hashes[leaked_mask].nunique()),
        # A model collapsing onto few outputs is its own quality problem,
        # so report it while the hashes are already computed.
        "synthetic_duplicate_rows_within_output": int(len(synthetic) - synth_hashes.nunique()),
    }


def summarize(result: dict) -> str:
    if not result.get("checked"):
        return f"⚠️  Leakage check skipped: {result.get('reason')}"
    n = result["exact_duplicates_of_training_rows"]
    if n == 0:
        return (f"✅ No verbatim training records in the synthetic output "
                f"({result['synthetic_rows']} rows, {result['columns_compared']} columns compared).")
    return (f"🚨 {n} synthetic row(s) ({result['exact_duplicate_rate']:.4%}) exactly reproduce a "
            f"training record, covering {result['distinct_training_rows_reproduced']} distinct real "
            f"patient(s). DO NOT PUBLISH THIS OUTPUT without addressing it.")

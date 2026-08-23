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


def _row_hashes(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Stable per-row hash over a fixed column order."""
    ordered = df[columns].astype(str)
    joined = ordered.agg("\x1f".join, axis=1)  # unit separator: not valid in the data
    return joined.map(lambda s: hashlib.sha256(s.encode()).hexdigest())


def check_exact_duplicates(synthetic: pd.DataFrame, real: pd.DataFrame) -> dict:
    """
    Counts synthetic rows that exactly reproduce a training row.

    Compares only on columns common to both, in a fixed order, with values
    stringified so dtype differences between the real frame and a
    synthesizer's output do not mask a genuine match.
    """
    common = [c for c in real.columns if c in synthetic.columns]
    if not common:
        return {"checked": False, "reason": "no overlapping columns"}

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

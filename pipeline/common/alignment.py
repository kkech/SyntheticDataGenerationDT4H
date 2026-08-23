"""
Categorical-representation alignment between synthetic and real frames.

The generative libraries detect boolean-like string columns
("true"/"false") and emit actual booleans, which serialize as
"True"/"False" -- a different representation from the real data's
lowercase strings. Any comparison that matches category strings exactly
then sees DISJOINT categories on ~150 columns: a classifier separates
real from synthetic perfectly on the spelling alone, distance metrics
mismatch every record pair, and exact-duplicate checks can never fire.
The fidelity metrics that lowercase internally (TVD, coherence rules,
TSTR targets) were unaffected, which is how the bug hid.

`align_categorical_case` maps a synthetic frame onto the REAL frame's
exact category spellings: boolean dtypes become the reference's
"true"/"false" strings, and any category whose lowercase form matches a
reference category's lowercase form adopts the reference spelling.
Genuinely unseen categories are left untouched -- inventing a category
is a real modelling behavior that evaluation must still see.

Applied in the generate step before the leakage check and the released
CSV is written, and defensively at read time by every analysis step, so
files generated before this fix are analyzed correctly without
regeneration.
"""

import pandas as pd


def align_categorical_case(synthetic: pd.DataFrame, reference: pd.DataFrame):
    """Return (aligned copy of synthetic, {column: cells changed})."""
    out = synthetic.copy()
    changed: dict[str, int] = {}
    for c in reference.columns:
        if c not in out.columns:
            continue
        ref = reference[c]
        if pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref):
            continue  # numeric columns have no spelling
        spellings = {str(v).lower(): str(v)
                     for v in pd.unique(ref.dropna().astype("object").astype(str))}

        original = out[c]
        col = original
        if pd.api.types.is_bool_dtype(col):
            col = col.map({True: "true", False: "false"})
        s = col.astype("object").where(col.notna(), None)
        aligned = s.map(lambda v: spellings.get(str(v).lower(), str(v))
                        if v is not None else None)
        # Count changes against the ORIGINAL representation (a bool False
        # becoming the string 'false' is a change even though its
        # post-conversion string compares equal).
        orig_str = original.astype("object").map(
            lambda v: str(v) if v is not None and v == v else "\x00")
        n = int((aligned.fillna("\x00") != orig_str).sum())
        if n:
            changed[c] = n
        out[c] = aligned
    return out, changed


def report(changed: dict) -> str:
    if not changed:
        return "Categorical representation already aligned (0 cells changed)."
    total = sum(changed.values())
    return (f"Aligned categorical representation to the real schema: {total} cells "
            f"in {len(changed)} column(s) re-spelled (e.g. True -> 'true').")

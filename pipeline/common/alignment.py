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


def canonical_str(v) -> str | None:
    """Canonical text form of a category value: integral floats render
    without the trailing '.0' (a numeric-parsed year 2017.0 must match
    the reference category '2017'), and surrounding whitespace is
    stripped. None/NaN stay None."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


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
        spellings = {}
        for v in pd.unique(ref.dropna().astype("object").astype(str)):
            spellings[str(v).strip().lower()] = str(v)

        original = out[c]
        col = original
        if pd.api.types.is_bool_dtype(col):
            col = col.map({True: "true", False: "false"})
        s = col.astype("object").where(col.notna(), None)

        def _align_value(v):
            cs = canonical_str(v)
            if cs is None:
                return None
            return spellings.get(cs.lower(), cs)

        aligned = s.map(_align_value)
        # Count changes against the ORIGINAL representation (a bool False
        # becoming the string 'false' is a change even though its
        # post-conversion string compares equal).
        orig_str = original.astype("object").map(
            lambda v: "\x00" if pd.isna(v) else str(v))
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


def harmonize_dtypes(frame, reference):
    """Coerce columns that are numeric (non-bool) in `reference` but
    landed as object in `frame` back to numeric. In-memory decode sets
    pd.NA into float columns, which pandas upcasts to object; files
    written to CSV re-read numeric so the pipeline never sees it, but
    frames concatenated in memory (top-up sampling, constraint-aware
    sampling) do. Returns (frame, [coerced column names])."""
    out = frame.copy()
    coerced = []
    for c in reference.columns:
        if c not in out.columns:
            continue
        ref_numeric = (pd.api.types.is_numeric_dtype(reference[c])
                       and not pd.api.types.is_bool_dtype(reference[c]))
        if ref_numeric and not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
            coerced.append(c)
    return out, coerced

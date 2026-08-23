"""
Representation audit: the tripwire for the whole class of "the two
frames being compared do not share a representation" bugs.

The categorical-case bug (synthetic 'True' vs real 'true') produced
silently perfect scores -- a classifier separating on spelling, distances
mismatching every record -- instead of an error. This audit runs on
every (synthetic, reference) pair AFTER alignment and turns any residual
mismatch into a loud, structured finding:

  * case_or_format_mismatches: synthetic categories that match a real
    category up to case/whitespace/number-format but not exactly. After
    alignment this must be EMPTY; anything here means a comparison
    upstream is about to be corrupted.
  * dtype_branch_divergences: columns where the synthetic side would take
    a different numeric-vs-categorical code path than the real side.
  * categorical_nulls: NaN cells in categorical columns of a full-width
    synthetic frame (the released representation uses an explicit
    'Missing' category; raw NaN there means an encoding drifted).
  * unseen_categories: genuinely new categories (no real counterpart at
    any spelling). This is a MODELLING behavior, reported not failed.

Used by the evaluate step (per file, results recorded and flagged) and
as a mandatory release-gate check.
"""

import pandas as pd

from pipeline.common.alignment import canonical_str


def audit_representation(synthetic: pd.DataFrame, reference: pd.DataFrame) -> dict:
    case_mismatches: dict[str, list] = {}
    dtype_divergences: list[str] = []
    categorical_nulls: dict[str, float] = {}
    unseen: dict[str, list] = {}

    for c in reference.columns:
        if c not in synthetic.columns:
            continue
        ref, syn = reference[c], synthetic[c]
        ref_is_num = pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref)
        syn_is_num = pd.api.types.is_numeric_dtype(syn) and not pd.api.types.is_bool_dtype(syn)

        if ref_is_num != syn_is_num:
            dtype_divergences.append(c)
        if ref_is_num:
            continue

        ref_exact = {str(v) for v in pd.unique(ref.dropna().astype("object").astype(str))}
        ref_canon = {canonical_str(v).lower() for v in ref_exact}

        nan_frac = float(syn.isna().mean())
        if nan_frac > 0:
            categorical_nulls[c] = round(nan_frac, 4)

        for v in pd.unique(syn.dropna().astype("object")):
            sv = str(v)
            if sv in ref_exact:
                continue
            cv = canonical_str(v)
            if cv is not None and cv.lower() in ref_canon:
                case_mismatches.setdefault(c, []).append(sv)
            else:
                unseen.setdefault(c, []).append(sv)

    return {
        "case_or_format_mismatches": case_mismatches,
        "dtype_branch_divergences": dtype_divergences,
        "categorical_nulls": categorical_nulls,
        "unseen_categories": {k: v[:5] for k, v in unseen.items()},
        "clean": not case_mismatches and not dtype_divergences,
    }


def summarize(audit: dict) -> str:
    if audit["clean"] and not audit["categorical_nulls"]:
        n_unseen = len(audit["unseen_categories"])
        extra = f"; {n_unseen} column(s) with genuinely new categories" if n_unseen else ""
        return f"representation audit clean{extra}"
    parts = []
    if audit["case_or_format_mismatches"]:
        parts.append(f"🚨 {len(audit['case_or_format_mismatches'])} column(s) with "
                     f"case/format category mismatches AFTER alignment")
    if audit["dtype_branch_divergences"]:
        parts.append(f"🚨 {len(audit['dtype_branch_divergences'])} column(s) taking a "
                     f"different numeric/categorical code path than the real side")
    if audit["categorical_nulls"]:
        parts.append(f"⚠️ {len(audit['categorical_nulls'])} categorical column(s) with raw "
                     f"NaN (released representation uses the 'Missing' category)")
    return "; ".join(parts)

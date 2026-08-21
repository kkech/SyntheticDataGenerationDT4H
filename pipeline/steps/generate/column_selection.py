"""
Outcome-relevance column selection for width-limited synthesizers (AIM).

Private-PGM-based AIM cannot handle this dataset's full 211-column
training width (it timed out at 6 hours inside marginal selection), so
its runs train on the K most important columns. "Important" is defined
by the data, not by hand: each candidate column is scored by its mean
absolute association with the metadata-declared clinical outcome
variables (Spearman for numeric-numeric, Cramer's V for
categorical-categorical, correlation ratio for mixed pairs) -- i.e. the
columns most predictive of, or most entangled with, the cohort's
endpoints.

Three guarantees:
  * the TSTR utility targets are force-included, so the reduced-width
    synthetic output remains utility-evaluable;
  * core demographics (age, gender) and NYHA are force-included -- no
    clinical reviewer accepts a heart-failure dataset without them;
  * declared outcome columns are otherwise EXCLUDED from the ranked
    pool: outcome variants correlate near-1.0 with each other and would
    crowd out actual predictors.

Scores are computed on the TRAIN split only and written to a committed
JSON so the selection is auditable.
"""

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.steps.evaluate.associations import (
    MIN_PAIR_OVERLAP,
    _cat_codes,
    _cramers_v,
    _correlation_ratio,
    _numeric_and_categorical,
)

FORCED_CLINICAL_COLUMNS = (
    "patient_demographics_age",
    "patient_demographics_gender",
    "nyha_nyha_pET",
)


def _pair_association(train, a, b, numeric_set, codes) -> float | None:
    a_num, b_num = a in numeric_set, b in numeric_set
    if a_num and b_num:
        pair = train[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < MIN_PAIR_OVERLAP:
            return None
        res = stats.spearmanr(pair[a], pair[b])
        rho = getattr(res, "statistic", getattr(res, "correlation", np.nan))
        return float(rho) if np.isfinite(rho) else None
    if not a_num and not b_num:
        return _cramers_v(codes[a], codes[b])
    num, cat = (a, b) if a_num else (b, a)
    vals = pd.to_numeric(train[num], errors="coerce").to_numpy(dtype=float)
    return _correlation_ratio(vals, codes[cat])


def select_important_columns(train: pd.DataFrame, outcome_cols: set[str],
                             forced: list[str], k: int) -> tuple[list[str], dict]:
    """Top-k columns of `train` by outcome relevance.

    Returns (selected column list in train order, {column: score}).
    `forced` columns count toward k and are included regardless of score.
    """
    numeric, categorical = _numeric_and_categorical(train)
    numeric_set = set(numeric)
    codes = _cat_codes(train, categorical)

    anchors = [c for c in outcome_cols if c in train.columns]
    forced_present = [c for c in dict.fromkeys(forced) if c in train.columns]

    scores: dict[str, float] = {}
    pool = [c for c in train.columns
            if c not in outcome_cols and c not in forced_present]
    for c in pool:
        vals = []
        for a in anchors:
            v = _pair_association(train, c, a, numeric_set, codes)
            if v is not None:
                vals.append(abs(v))
        scores[c] = round(float(np.mean(vals)), 4) if vals else 0.0

    ranked = sorted(pool, key=lambda c: -scores[c])
    n_fill = max(k - len(forced_present), 0)
    chosen = set(forced_present) | set(ranked[:n_fill])
    selected = [c for c in train.columns if c in chosen]  # keep train order
    return selected, scores

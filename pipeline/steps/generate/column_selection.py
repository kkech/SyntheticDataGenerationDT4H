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

Scores are computed on the TRAIN split only. The committed JSON records
the RANKING and the selection parameters -- not the scores themselves:
the scores are exact, unnoised statistics of real patient data, and
publishing them next to a DP-labelled dataset would release precisely
the kind of quantity the DP claim is about. The order is what makes the
selection auditable; the magnitudes were only ever diagnostics.

DISCLOSURE, NOT A FIX: the selection is itself data-dependent. Which
columns a width-limited (AIM) run models at all is chosen by looking at
the real training split without any privacy budget, so a DP run using
this subset is DP *given the column set*, and the column set leaks. That
is a documented limitation of the width-limited runs and must be stated
wherever they are reported; closing it needs either a public column list
(chosen from clinical knowledge / metadata alone) or a DP selection
mechanism (e.g. exponential mechanism over the association scores) with
its own share of the budget. Neither is implemented here.

Iteration order is pinned: the anchors, the ranked pool and ties are all
sorted, so the committed selection is identical across processes rather
than depending on set-iteration order (which varies with PYTHONHASHSEED).
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
                             forced: list[str], k: int) -> tuple[list[str], list[str]]:
    """Top-k columns of `train` by outcome relevance.

    Returns (selected column list in train order, ranked pool columns
    best-first). `forced` columns count toward k and are included
    regardless of score.

    Scores stay internal by design -- see the module docstring: they are
    unnoised statistics of the real training split, so only the RANK
    order leaves this function and reaches the committed JSON.

    `outcome_cols` arrives as a set, so it is sorted before it drives any
    loop: the anchor order affects nothing but floating-point summation
    order, and "affects nothing but floating point" is exactly how an
    irreproducible committed artifact happens.
    """
    numeric, categorical = _numeric_and_categorical(train)
    numeric_set = set(numeric)
    codes = _cat_codes(train, categorical)

    anchors = sorted(c for c in outcome_cols if c in train.columns)
    forced_present = [c for c in dict.fromkeys(forced) if c in train.columns]
    forced_set = set(forced_present)

    scores: dict[str, float] = {}
    pool = [c for c in train.columns
            if c not in outcome_cols and c not in forced_set]
    for c in pool:
        vals = []
        for a in anchors:
            v = _pair_association(train, c, a, numeric_set, codes)
            if v is not None:
                vals.append(abs(v))
        scores[c] = round(float(np.mean(vals)), 4) if vals else 0.0

    # Ties broken by column name, so equal scores (common at 0.0) cannot
    # reorder between runs.
    ranked = sorted(pool, key=lambda c: (-scores[c], c))
    n_fill = max(k - len(forced_present), 0)
    chosen = forced_set | set(ranked[:n_fill])
    selected = [c for c in train.columns if c in chosen]  # keep train order
    return selected, ranked

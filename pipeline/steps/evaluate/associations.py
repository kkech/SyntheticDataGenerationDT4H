"""
Pairwise association structure: does the synthetic data preserve HOW
columns relate, not just each column's marginal distribution?

Marginal metrics (KS/TVD) can all be excellent while the joint structure
is destroyed -- e.g. synthetic potassium and eGFR each distributed
correctly but no longer correlated. For a published dataset the
association structure is the difference between "plausible-looking rows"
and "a usable research dataset", so it is measured explicitly:

  * numeric-numeric:      Spearman rank correlation (robust to the
                          skewed lab distributions), pairwise-complete;
  * categorical-categorical: Cramer's V from the chi-square of the
                          contingency table;
  * numeric-categorical:  correlation ratio (eta): how much of the
                          numeric variance the categories explain.

All three live in [-1,1] or [0,1]; the reported quantity per pair is the
absolute difference between the real and synthetic association, so 0
means the relationship is perfectly preserved.

Pairs with fewer than MIN_PAIR_OVERLAP jointly-observed rows are skipped:
an association estimated on a handful of overlapping observations is
noise, and scoring a generator against noise is meaningless.
"""

import numpy as np
import pandas as pd

MIN_PAIR_OVERLAP = 50


def _numeric_and_categorical(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def _cat_codes(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """Integer codes per categorical column, nulls folded into a 'Missing'
    category (consistent with how the pipeline represents them)."""
    out = {}
    for c in cols:
        s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
        out[c] = pd.factorize(s)[0]
    return out


def _cramers_v(codes_a: np.ndarray, codes_b: np.ndarray) -> float | None:
    na, nb = codes_a.max() + 1, codes_b.max() + 1
    if na < 2 or nb < 2:
        return None  # a constant column has no association to measure
    n = len(codes_a)
    contingency = np.bincount(codes_a * nb + codes_b, minlength=na * nb).reshape(na, nb).astype(float)
    rows = contingency.sum(axis=1, keepdims=True)
    cols = contingency.sum(axis=0, keepdims=True)
    expected = rows @ cols / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (contingency - expected) ** 2 / expected, 0.0))
    denom = n * (min(na, nb) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else None


def _correlation_ratio(values: np.ndarray, codes: np.ndarray) -> float | None:
    mask = ~np.isnan(values)
    if mask.sum() < MIN_PAIR_OVERLAP:
        return None
    v, g = values[mask], codes[mask]
    grand = v.mean()
    ss_total = ((v - grand) ** 2).sum()
    if ss_total <= 0:
        return None
    ss_between = 0.0
    for code in np.unique(g):
        grp = v[g == code]
        ss_between += len(grp) * (grp.mean() - grand) ** 2
    return float(np.sqrt(ss_between / ss_total))


def association_profile(df: pd.DataFrame) -> dict:
    """All pairwise associations for one frame, as {pair_key: value}."""
    numeric, categorical = _numeric_and_categorical(df)

    num_num = {}
    if len(numeric) >= 2:
        corr = df[numeric].corr(method="spearman", min_periods=MIN_PAIR_OVERLAP)
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v):
                    num_num[f"{a}|{b}"] = float(v)

    codes = _cat_codes(df, categorical)
    cat_cat = {}
    for i, a in enumerate(categorical):
        for b in categorical[i + 1:]:
            v = _cramers_v(codes[a], codes[b])
            if v is not None:
                cat_cat[f"{a}|{b}"] = v

    num_cat = {}
    for a in numeric:
        vals = pd.to_numeric(df[a], errors="coerce").to_numpy(dtype=float)
        for b in categorical:
            v = _correlation_ratio(vals, codes[b])
            if v is not None:
                num_cat[f"{a}|{b}"] = v

    return {"num_num": num_num, "cat_cat": cat_cat, "num_cat": num_cat}


FABRICATED_REAL_MAX = 0.1   # essentially no real association ...
FABRICATED_SYNTH_MIN = 0.5  # ... rendered as a strong synthetic one


def compare_association_profiles(real: dict, synth: dict) -> dict:
    """Absolute association difference per pair, aggregated per pair type.

    Also counts FABRICATED associations -- pairs nearly independent in
    the real data (|assoc| < 0.1) but strongly associated in the
    synthetic data (|assoc| > 0.5). Fabrication is worse than
    attenuation: a user of the released data would 'discover' a
    relationship that does not exist.
    """
    result = {}
    for kind in ("num_num", "cat_cat", "num_cat"):
        common = set(real[kind]) & set(synth[kind])
        deltas = {k: abs(real[kind][k] - synth[kind][k]) for k in common}
        fabricated = [k for k in common
                      if abs(real[kind][k]) < FABRICATED_REAL_MAX
                      and abs(synth[kind][k]) > FABRICATED_SYNTH_MIN]
        if deltas:
            s = pd.Series(deltas)
            worst = [
                {"pair": k, "delta": round(v, 4),
                 "real": round(real[kind][k], 4), "synthetic": round(synth[kind][k], 4)}
                for k, v in s.sort_values(ascending=False).head(5).items()
            ]
            result[kind] = {
                "pairs": len(deltas),
                "mean_abs_delta": round(float(s.mean()), 4),
                "median_abs_delta": round(float(s.median()), 4),
                "max_abs_delta": round(float(s.max()), 4),
                "frac_below_0.1": round(float((s < 0.1).mean()), 4),
                "fabricated_pairs": len(fabricated),
                "fabricated_rate": round(len(fabricated) / len(deltas), 4),
                "fabricated_examples": sorted(
                    fabricated, key=lambda k: -abs(synth[kind][k]))[:3],
                "worst": worst,
            }
        else:
            result[kind] = {"pairs": 0}
        result[kind]["pairs_only_in_real"] = len(set(real[kind]) - common)
    return result

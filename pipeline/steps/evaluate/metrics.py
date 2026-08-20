"""
Distribution-distance metrics for original vs preprocessed vs synthetic.

Deliberately basic, per-column statistics to start with -- enough to see
where a synthesizer is faithful and where it drifts, and to compare
synthesizers against each other on equal terms:

  * numeric columns: two-sample Kolmogorov-Smirnov statistic (max CDF
    gap, scale-free, in [0,1]) and Wasserstein distance standardized by
    the reference std (so "0.1" means "earth moved ~0.1 reference
    standard deviations" regardless of the column's units), plus
    relative mean/std differences;
  * categorical columns: total variation distance between category
    frequency distributions (in [0,1]; the fraction of probability mass
    that would have to move);
  * every column: the missingness rate on each side. Missingness is
    modelled, not imputed, in this pipeline -- so how well the
    synthesizer reproduces WHICH cells are empty is itself a fidelity
    result, not a nuisance.

All numeric comparisons are over OBSERVED values only (nulls excluded);
missingness is compared separately rather than being allowed to distort
the value distributions.
"""

import pandas as pd


def _as_category_probs(series: pd.Series) -> pd.Series:
    """Category frequency distribution over a normalized string space.

    Lowercased so raw booleans (True) and synthesized strings ('true')
    land on the same category; nulls become an explicit 'Missing'
    category so missingness differences show up in the TVD as well.
    """
    def _norm(v):
        # Robust to non-scalar cells (e.g. list/array values in the raw
        # frame): anything unhashable or odd becomes its string form.
        if v is None:
            return "missing"
        if isinstance(v, float) and v != v:  # NaN
            return "missing"
        try:
            if pd.isna(v):
                return "missing"
        except (TypeError, ValueError):
            pass
        return str(v).lower()

    s = series.map(_norm)
    return s.value_counts(normalize=True)


def total_variation_distance(a: pd.Series, b: pd.Series) -> float:
    pa, pb = _as_category_probs(a), _as_category_probs(b)
    cats = pa.index.union(pb.index)
    return float(0.5 * sum(abs(pa.get(c, 0.0) - pb.get(c, 0.0)) for c in cats))


def numeric_metrics(a: pd.Series, b: pd.Series) -> dict | None:
    """KS + standardized Wasserstein over observed values. None if either
    side has no observed values to compare."""
    from scipy import stats

    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(av) == 0 or len(bv) == 0:
        return None

    ks = stats.ks_2samp(av, bv)
    ref_std = av.std()
    w = stats.wasserstein_distance(av, bv)
    return {
        "n_a": int(len(av)),
        "n_b": int(len(bv)),
        "ks_statistic": round(float(ks.statistic), 4),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": round(float(w), 6),
        "wasserstein_std": round(float(w / ref_std), 4) if ref_std > 0 else None,
        "mean_a": round(float(av.mean()), 4),
        "mean_b": round(float(bv.mean()), 4),
        "std_a": round(float(av.std()), 4),
        "std_b": round(float(bv.std()), 4),
    }


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def compare_frames(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> dict:
    """Per-column comparison over the columns common to both frames."""
    common = [c for c in df_a.columns if c in df_b.columns]
    numeric_rows, categorical_rows = [], []

    for col in common:
        a, b = df_a[col], df_b[col]
        miss_a = round(float(a.isna().mean()), 4)
        miss_b = round(float(b.isna().mean()), 4)

        if _is_numeric(a) and _is_numeric(b):
            m = numeric_metrics(a, b)
            if m is None:
                continue
            m.update({"column": col, "missing_rate_a": miss_a, "missing_rate_b": miss_b})
            numeric_rows.append(m)
        else:
            categorical_rows.append({
                "column": col,
                "tvd": round(total_variation_distance(a, b), 4),
                "n_categories_a": int(a.astype("object").where(a.notna(), "Missing").nunique()),
                "n_categories_b": int(b.astype("object").where(b.notna(), "Missing").nunique()),
                "missing_rate_a": miss_a,
                "missing_rate_b": miss_b,
            })

    def _agg(values):
        if not values:
            return {}
        s = pd.Series(values)
        return {"mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
                "max": round(float(s.max()), 4)}

    ks_vals = [r["ks_statistic"] for r in numeric_rows]
    w_vals = [r["wasserstein_std"] for r in numeric_rows if r["wasserstein_std"] is not None]
    tvd_vals = [r["tvd"] for r in categorical_rows]
    # Numeric columns only: categorical missingness is an explicit
    # 'Missing' category on the synthesized side, so it is already part
    # of the TVD; mixing representations here would double-count it and
    # make the aggregate read as drift where none exists.
    miss_diffs = [abs(r["missing_rate_a"] - r["missing_rate_b"]) for r in numeric_rows]

    return {
        "pair": f"{label_a} vs {label_b}",
        "columns_compared": len(numeric_rows) + len(categorical_rows),
        "columns_only_in_a": len(df_a.columns) - len(common),
        "columns_only_in_b": len(df_b.columns) - len(common),
        "aggregates": {
            "numeric_columns": len(numeric_rows),
            "categorical_columns": len(categorical_rows),
            "ks": _agg(ks_vals),
            "ks_frac_below_0.1": round(sum(v < 0.1 for v in ks_vals) / len(ks_vals), 4) if ks_vals else None,
            "wasserstein_std": _agg(w_vals),
            "tvd": _agg(tvd_vals),
            "tvd_frac_below_0.05": round(sum(v < 0.05 for v in tvd_vals) / len(tvd_vals), 4) if tvd_vals else None,
            "missing_rate_mean_abs_diff": round(sum(miss_diffs) / len(miss_diffs), 4) if miss_diffs else None,
        },
        "worst_numeric": sorted(numeric_rows, key=lambda r: -r["ks_statistic"])[:5],
        "worst_categorical": sorted(categorical_rows, key=lambda r: -r["tvd"])[:5],
        "numeric": numeric_rows,
        "categorical": categorical_rows,
    }

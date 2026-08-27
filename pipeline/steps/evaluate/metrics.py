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

All numeric value comparisons are over OBSERVED values only (nulls
excluded); missingness is compared separately rather than being allowed
to distort the value distributions. A numeric column that one side
observes and the other leaves entirely empty is NOT dropped: that is the
maximal possible divergence, and dropping it would erase precisely the
worst failures (it scores KS=1.0 and its missing-rate gap stays in the
missingness aggregate).

`compare_frames_subsampled` exists because two-sample statistics like KS
concentrate toward zero as sample sizes grow: a noise floor computed at
(n_train, n_holdout) is only exchangeable with comparisons run at the
same (n_a, n_b), so the larger frame is drawn down to the floor's row
count before scoring.
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
    """KS + standardized Wasserstein over observed values.

    Returns None only when BOTH sides have no observed values -- then
    there is no value distribution at all (the caller still compares
    missingness). When exactly one side is empty the divergence is
    maximal, not unmeasurable: silently dropping such a column would
    hide the worst possible failure mode, so it scores ks=1.0 with
    `unscoreable: True` (Wasserstein needs values on both sides and
    stays None).
    """
    from scipy import stats

    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(av) == 0 and len(bv) == 0:
        return None
    if len(av) == 0 or len(bv) == 0:
        return {
            "n_a": int(len(av)),
            "n_b": int(len(bv)),
            # Every observed value sits where the other side has no mass
            # at all: the CDF gap is 1 by definition.
            "ks_statistic": 1.0,
            "ks_pvalue": None,
            "wasserstein": None,
            "wasserstein_std": None,
            "mean_a": round(float(av.mean()), 4) if len(av) else None,
            "mean_b": round(float(bv.mean()), 4) if len(bv) else None,
            "std_a": round(float(av.std()), 4) if len(av) else None,
            "std_b": round(float(bv.std()), 4) if len(bv) else None,
            "unscoreable": True,
        }

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


def _build_result(pair: str, numeric_rows: list, categorical_rows: list,
                  missing_only_rows: list, columns_only_in_a: int,
                  columns_only_in_b: int) -> dict:
    """Aggregates + worst-offender lists from already-built per-column
    rows. Shared between the direct comparison and the averaged
    subsampled comparison so both report the exact same structure."""

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
    # make the aggregate read as drift where none exists. Columns with
    # no observed values on one or both sides still belong here -- their
    # missingness gap is exactly what there is to compare.
    miss_diffs = [abs(r["missing_rate_a"] - r["missing_rate_b"])
                  for r in numeric_rows + missing_only_rows]

    return {
        "pair": pair,
        "columns_compared": len(numeric_rows) + len(categorical_rows) + len(missing_only_rows),
        "columns_unscoreable": sum(1 for r in numeric_rows if r.get("unscoreable")),
        "columns_only_in_a": columns_only_in_a,
        "columns_only_in_b": columns_only_in_b,
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
        "numeric_missing_only": missing_only_rows,
    }


def compare_frames(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                   exclude_columns: set | None = None) -> dict:
    """Per-column comparison over the columns common to both frames.

    `exclude_columns` removes columns from the comparison entirely --
    used to keep the 38 constant columns (re-attached verbatim by the
    generate step, trivially perfect in every metric) from flattering
    the aggregates: they are copies, not modelling successes.
    """
    exclude_columns = exclude_columns or set()
    b_cols = set(df_b.columns)
    common = [c for c in df_a.columns if c in b_cols and c not in exclude_columns]
    numeric_rows, categorical_rows, missing_only_rows = [], [], []

    for col in common:
        a, b = df_a[col], df_b[col]
        miss_a = round(float(a.isna().mean()), 4)
        miss_b = round(float(b.isna().mean()), 4)

        if _is_numeric(a) and _is_numeric(b):
            m = numeric_metrics(a, b)
            if m is None:
                # Both sides all-missing: no value distribution to
                # score, but the missingness comparison is still real
                # (zero only when both missing rates agree), so it stays
                # in the missing-rate aggregate instead of vanishing.
                missing_only_rows.append({"column": col, "missing_rate_a": miss_a,
                                          "missing_rate_b": miss_b})
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

    # Counted against the OTHER frame's actual columns, not against
    # `common`: `common` also drops the excluded constants, which are
    # present in both frames and must not be misreported as schema gaps.
    a_cols = set(df_a.columns)
    return _build_result(
        f"{label_a} vs {label_b}", numeric_rows, categorical_rows, missing_only_rows,
        columns_only_in_a=sum(1 for c in df_a.columns if c not in b_cols),
        columns_only_in_b=sum(1 for c in df_b.columns if c not in a_cols),
    )


# Per-column fields that vary between subsample draws (everything about
# side b, plus the two-sample statistics); side-a fields are identical
# across draws and are carried through from the first draw. The value is
# the rounding used for the mean (None = keep full float precision,
# 0 = round to int).
_NUMERIC_AVG_FIELDS = {"ks_statistic": 4, "ks_pvalue": None, "wasserstein": 6,
                       "wasserstein_std": 4, "mean_b": 4, "std_b": 4,
                       "missing_rate_b": 4, "n_b": 0}
_CATEGORICAL_AVG_FIELDS = {"tvd": 4, "n_categories_b": 0, "missing_rate_b": 4}
_MISSING_ONLY_AVG_FIELDS = {"missing_rate_b": 4}


def _mean_field(rows: list, field: str, ndigits: int | None):
    vals = [r.get(field) for r in rows if r.get(field) is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if ndigits is None:
        return float(m)
    if ndigits == 0:
        return int(round(m))
    return round(float(m), ndigits)


def _merge_rows(runs: list, key: str, fields: dict) -> list:
    """Per-column mean across draws, matched by column name. A column's
    row is averaged over the draws where it was measurable (a draw can,
    rarely, leave a sparse column with no observed values)."""
    by_col = [{r["column"]: r for r in run[key]} for run in runs]
    merged = []
    for col in by_col[0]:
        rows = [d[col] for d in by_col if col in d]
        row = dict(rows[0])
        for f, nd in fields.items():
            row[f] = _mean_field(rows, f, nd)
        if any(r.get("unscoreable") for r in rows):
            row["unscoreable"] = True
        merged.append(row)
    return merged


def compare_frames_subsampled(df_a: pd.DataFrame, df_b: pd.DataFrame,
                              label_a: str, label_b: str, n_rows: int, base_seed: int,
                              repeats: int = 3, exclude_columns: set | None = None) -> dict:
    """compare_frames with df_b subsampled (without replacement) to
    n_rows, averaged over `repeats` distinctly-seeded draws.

    WHY: KS-type statistics concentrate toward zero as sample sizes
    grow, so the train-vs-holdout noise floor at (n_train, n_holdout) is
    only exchangeable with comparisons run at the same (n_a, n_b).
    Scoring a full-size synthetic frame against that floor makes the
    floor read ~40% too loose at this dataset's sizes. Drawing df_b down
    to the floor's row count (seeds base_seed, +1, +2, ...) puts both on
    the same geometry; averaging the draws keeps the extra subsampling
    noise out of the headline numbers. Per-column rows in the result are
    means across draws; aggregates and worst lists are recomputed from
    those means.
    """
    if len(df_b) <= n_rows:
        # Already at (or below) the target geometry -- nothing to draw.
        out = compare_frames(df_a, df_b, label_a, label_b, exclude_columns=exclude_columns)
        out["subsample_rows"] = int(len(df_b))
        out["subsample_repeats"] = 1
        return out

    runs = [compare_frames(df_a, df_b.sample(n=n_rows, random_state=base_seed + i),
                           label_a, label_b, exclude_columns=exclude_columns)
            for i in range(max(1, repeats))]
    if len(runs) == 1:
        out = runs[0]
    else:
        out = _build_result(
            runs[0]["pair"],
            _merge_rows(runs, "numeric", _NUMERIC_AVG_FIELDS),
            _merge_rows(runs, "categorical", _CATEGORICAL_AVG_FIELDS),
            _merge_rows(runs, "numeric_missing_only", _MISSING_ONLY_AVG_FIELDS),
            runs[0]["columns_only_in_a"],
            runs[0]["columns_only_in_b"],
        )
    out["subsample_rows"] = int(n_rows)
    out["subsample_repeats"] = len(runs)
    return out

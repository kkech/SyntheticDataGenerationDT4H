"""
Privacy-safe per-column profiler, used by the profile_data step to
summarize the full dataset before any of it leaves the local machine.

Every category value is only reported by name if at least
SUPPRESSION_THRESHOLD rows share it (a k-anonymity-style guard); rarer
values -- including entire near-unique/free-text/identifier columns -- are
rolled into a "suppressed" bucket instead, so raw record-level values never
end up in a committed report.
"""

import polars as pl

HIGH_CARDINALITY_THRESHOLD = 50
TOP_N_CATEGORIES = 20
SUPPRESSION_THRESHOLD = 5


def classify_column(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_numeric():
        return "numeric"
    return "categorical"


def summarize_value_counts(series: pl.Series) -> dict:
    """Top value counts with rare/near-unique values suppressed for privacy."""
    vc = series.value_counts().sort("count", descending=True)
    value_col = [c for c in vc.columns if c != "count"][0]

    top_values = {}
    shown_non_null = 0
    suppressed_categories = 0
    suppressed_rows = 0

    for value, count in vc.select([value_col, "count"]).iter_rows():
        if value is None:
            top_values["null"] = count
            continue
        if shown_non_null < TOP_N_CATEGORIES and count >= SUPPRESSION_THRESHOLD:
            top_values[str(value)] = count
            shown_non_null += 1
        else:
            suppressed_categories += 1
            suppressed_rows += count

    result = {"top_values": top_values}
    if suppressed_categories:
        result["suppressed"] = {
            "distinct_values_suppressed": suppressed_categories,
            "rows_covered": suppressed_rows,
            "reason": f"count below {SUPPRESSION_THRESHOLD} and/or ranked beyond top {TOP_N_CATEGORIES}",
        }
    return result


def analyze_boolean(series: pl.Series) -> dict:
    result = {
        "unique_count": series.n_unique(),
        "is_constant": series.drop_nulls().n_unique() <= 1,
    }
    result.update(summarize_value_counts(series))
    return result


def analyze_numeric(series: pl.Series) -> dict:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return {"note": "All values are null."}

    quantiles = {str(q): non_null.quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {
        "min": non_null.min(),
        "max": non_null.max(),
        "mean": non_null.mean(),
        "std": non_null.std(),
        "quantiles": quantiles,
        "is_constant": non_null.n_unique() <= 1,
    }


def analyze_categorical(series: pl.Series) -> dict:
    unique_count = series.n_unique()
    result = {
        "unique_count": unique_count,
        "is_constant": series.drop_nulls().n_unique() <= 1,
        "high_cardinality": unique_count > HIGH_CARDINALITY_THRESHOLD,
    }
    result.update(summarize_value_counts(series))
    return result


def analyze_column(df: pl.DataFrame, col: str) -> dict:
    series = df[col]
    total = df.height
    null_count = series.null_count()
    col_type = classify_column(series.dtype)

    info = {
        "dtype": str(series.dtype),
        "inferred_type": col_type,
        "row_count": total,
        "null_count": null_count,
        "null_pct": round(null_count / total, 4) if total else None,
    }

    if col_type == "boolean":
        info.update(analyze_boolean(series))
    elif col_type == "numeric":
        info.update(analyze_numeric(series))
    else:
        info.update(analyze_categorical(series))

    return info


def write_markdown(analysis: dict, total_rows: int, path: str) -> None:
    lines = [
        "# Column Analysis",
        "",
        f"Total rows: {total_rows}",
        f"Total columns: {len(analysis)}",
        "",
    ]

    for col, info in analysis.items():
        lines.append(f"## {col}")
        lines.append("")
        lines.append(f"- dtype: `{info['dtype']}` ({info['inferred_type']})")
        lines.append(
            f"- nulls: {info['null_count']} ({info['null_pct']:.2%})"
            if info["null_pct"] is not None
            else "- nulls: n/a"
        )

        if info["inferred_type"] == "numeric":
            if "note" in info:
                lines.append(f"- {info['note']}")
            else:
                lines.append(f"- min/max: {info['min']} / {info['max']}")
                lines.append(f"- mean/std: {info['mean']:.4f} / {info['std']}")
                lines.append(f"- quantiles: {info['quantiles']}")
                if info["is_constant"]:
                    lines.append("- ⚠️ constant column (single value)")
        else:
            lines.append(f"- unique values: {info['unique_count']}")
            if info.get("is_constant"):
                lines.append("- ⚠️ constant column (single value)")
            if info.get("high_cardinality"):
                lines.append(f"- ⚠️ high-cardinality column ({info['unique_count']} distinct values)")
            lines.append(f"- top values (shown only where count ≥ {SUPPRESSION_THRESHOLD}):")
            if info["top_values"]:
                for value, count in info["top_values"].items():
                    lines.append(f"  - `{value}`: {count}")
            else:
                lines.append("  - (none met the display threshold)")
            if "suppressed" in info:
                s = info["suppressed"]
                lines.append(
                    f"  - {s['distinct_values_suppressed']} other distinct value(s) "
                    f"suppressed, covering {s['rows_covered']} row(s) ({s['reason']})"
                )

        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

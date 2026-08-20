import polars as pl
import json
import os

# --- CONFIGURATION ---
INPUT_PATH = "/mnt/data/DT4Hnew/DT4H_Cleaned_Data.parquet"
OUTPUT_JSON = "/mnt/data/DT4Hnew/DT4H_Column_Analysis.json"
OUTPUT_MD = "/mnt/data/DT4Hnew/DT4H_Column_Analysis.md"

# Columns with more distinct values than this are treated as high-cardinality:
# only the top N values are reported (matches the cutoff used later in
# metaDataMergedAndClean.py to drop noisy high-cardinality string columns).
HIGH_CARDINALITY_THRESHOLD = 50
TOP_N_CATEGORIES = 20
ID_NAME_PATTERNS = ["id_", "contactid", "metingid", "prescriptionid", "pseudo_id"]

# Privacy guard: a category value is only reported by name if at least this
# many rows share it. Rarer values (including near-unique / free-text /
# identifier columns) are rolled into a "suppressed" bucket instead, so raw
# record-level values never end up in the committed report.
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

    quantiles = {
        str(q): non_null.quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }
    std = non_null.std()
    return {
        "min": non_null.min(),
        "max": non_null.max(),
        "mean": non_null.mean(),
        "std": std,
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
    col_lower = col.lower()

    info = {
        "dtype": str(series.dtype),
        "inferred_type": col_type,
        "row_count": total,
        "null_count": null_count,
        "null_pct": round(null_count / total, 4) if total else None,
        "looks_like_id": any(pat in col_lower for pat in ID_NAME_PATTERNS),
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
        lines.append(f"- nulls: {info['null_count']} ({info['null_pct']:.2%})" if info["null_pct"] is not None else "- nulls: n/a")
        if info.get("looks_like_id"):
            lines.append("- ⚠️ looks like an identifier column (by name)")

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


def explore_data():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Could not find {INPUT_PATH}")
        return

    df = pl.read_parquet(INPUT_PATH)
    total_rows = df.height
    print(f"--- 📊 ANALYZING: {total_rows} rows, {df.width} columns ---")

    analysis = {}
    for col in df.columns:
        print(f"  -> profiling '{col}'")
        analysis[col] = analyze_column(df, col)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {"total_rows": total_rows, "total_columns": df.width, "columns": analysis},
            f,
            indent=2,
            default=str,
        )
    print(f"\n✅ JSON analysis saved to: {OUTPUT_JSON}")

    write_markdown(analysis, total_rows, OUTPUT_MD)
    print(f"✅ Markdown summary saved to: {OUTPUT_MD}")


if __name__ == "__main__":
    explore_data()

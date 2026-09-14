"""
Privacy-safe profiler: the regressions this pipeline actually hit, locked
in as tests -- nested-dtype value counts (the List(String) ECG columns
that crashed the first extended campaign) and outward coarsening of
disclosed extremes.
"""

import polars as pl

from pipeline.common.profiling import (
    SUPPRESSION_THRESHOLD,
    analyze_column,
    coarsen_extreme,
)


def test_list_string_column_profiles_without_casting():
    # Regression: the deterministic tie-break sort cast the value column
    # to String, which polars refuses for List(String) -- the raw
    # extract's ARRAY[NOMINAL] ECG-rhythm columns.
    df = pl.DataFrame({"ecg": pl.Series(
        [["afib", "sinus"]] * 6 + [["sinus"]] * 6 + [None] * 3,
        dtype=pl.List(pl.String))})
    info = analyze_column(df, "ecg")
    assert info["inferred_type"] == "categorical"
    assert info["top_values"]["__null__"] == 3
    assert sum(v for k, v in info["top_values"].items() if k != "__null__") == 12


def test_tie_break_is_deterministic_across_row_orders():
    rows = ["b"] * SUPPRESSION_THRESHOLD + ["a"] * SUPPRESSION_THRESHOLD
    a = analyze_column(pl.DataFrame({"c": rows}), "c")
    b = analyze_column(pl.DataFrame({"c": rows[::-1]}), "c")
    assert list(a["top_values"]) == list(b["top_values"]) == ["a", "b"]


def test_coarsen_extreme_rounds_outward():
    assert coarsen_extreme(31.75, "floor") <= 31.75
    assert coarsen_extreme(807.3, "ceil") >= 807.3
    # 2 significant figures: the exact value never survives
    assert coarsen_extreme(31.75, "floor") == 31.0
    assert coarsen_extreme(807.3, "ceil") == 810.0
    assert coarsen_extreme(-4.31, "floor") == -4.4
    assert coarsen_extreme(0.0, "ceil") == 0.0

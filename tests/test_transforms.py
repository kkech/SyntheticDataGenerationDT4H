"""
Preprocessing transforms: the medication/condition combiners, the
pinned-schema column drop, and NYHA ordinal encoding -- including the
feature-set 2.2 shapes (requests-only medication indicators, the second
NYHA column) that motivated the fixes these tests lock in.
"""

import polars as pl
import pytest

from pipeline.steps.preprocess import transforms as t


def _bool(vals):
    return pl.Series(vals, dtype=pl.Boolean)


# --- combine_medications ---

def test_combine_medications_symmetric_pair_ors_and_drops_sources():
    df = pl.DataFrame({
        "med_admins_acei_any": _bool([True, False, None]),
        "med_requests_acei_any": _bool([False, False, True]),
        "other": pl.Series([1, 2, 3]),
    })
    out, info = t.combine_medications(df)
    assert out["med_acei"].to_list() == [True, False, True]  # null -> False, OR'd
    assert "med_admins_acei_any" not in out.columns
    assert "med_requests_acei_any" not in out.columns
    assert "other" in out.columns
    assert info["features_created"] == ["med_acei"]


def test_combine_medications_requests_only_column_is_combined():
    # Feature-set 2.2: med_requests_activeDuringEncounter_*_any has no
    # admins counterpart. It must still become a med_* feature (and its
    # raw variant must be dropped), not survive as an unmapped raw column.
    df = pl.DataFrame({
        "med_requests_activeDuringEncounter_bb_any": _bool([True, None, False]),
        "med_admins_acei_any": _bool([False, True, False]),
        "med_requests_acei_any": _bool([False, False, False]),
    })
    out, info = t.combine_medications(df)
    assert "med_activeDuringEncounter_bb" in out.columns
    assert out["med_activeDuringEncounter_bb"].to_list() == [True, False, False]
    assert "med_requests_activeDuringEncounter_bb_any" not in out.columns
    assert set(info["features_created"]) == {"med_acei", "med_activeDuringEncounter_bb"}


def test_combine_medications_admins_only_column_is_combined():
    df = pl.DataFrame({"med_admins_loop_diuretic_any": _bool([True, False])})
    out, _ = t.combine_medications(df)
    assert out["med_loop_diuretic"].to_list() == [True, False]
    assert "med_admins_loop_diuretic_any" not in out.columns


def test_combine_medications_history_variants():
    df = pl.DataFrame({
        "med_admins_history_sglt2i_any": _bool([None, True]),
        "med_requests_history_sglt2i_any": _bool([True, None]),
        "med_admins_sglt2i_any": _bool([False, False]),
        "med_requests_sglt2i_any": _bool([True, False]),
    })
    out, info = t.combine_medications(df)
    assert out["med_sglt2i"].to_list() == [True, False]
    assert out["med_sglt2i_history"].to_list() == [True, True]
    assert set(info["features_created"]) == {"med_sglt2i", "med_sglt2i_history"}


def test_combine_medications_noop_on_symmetric_14_schema_shape():
    # On the 1.4/2.1 schema every med has both variants; keying groups on
    # the admins/requests UNION must produce exactly the same features as
    # the old admins-only keying did there.
    df = pl.DataFrame({
        f"med_{kind}_{med}_any": _bool([True, False])
        for med in ("acei", "bb")
        for kind in ("admins", "requests")
    })
    out, info = t.combine_medications(df)
    assert sorted(info["features_created"]) == ["med_acei", "med_bb"]
    assert sorted(c for c in out.columns) == ["med_acei", "med_bb"]


# --- combine_conditions ---

def test_combine_conditions_ors_windows_and_drops_sources():
    df = pl.DataFrame({
        "conditions_ckd_pre_dc_any": _bool([True, None, False]),
        "conditions_ckd_pre_adm_any": _bool([False, False, False]),
        "conditions_ckd_during_pET_any": _bool([None, True, False]),
        "conditions_ckd_timeFromEarliest_first": pl.Series([1.0, 2.0, None]),
    })
    out, info = t.combine_conditions(df)
    assert out["conditions_ckd"].to_list() == [True, True, False]
    assert info["features_created"] == ["conditions_ckd"]
    # non-window condition columns are untouched
    assert "conditions_ckd_timeFromEarliest_first" in out.columns
    for c in ("conditions_ckd_pre_dc_any", "conditions_ckd_pre_adm_any",
              "conditions_ckd_during_pET_any"):
        assert c not in out.columns


# --- drop_undeclared_columns ---
# (pinned-schema mode exists on the 2.2-support line; skip cleanly where
# a checkout does not carry it, so this file runs identically in both repos)

pinned_schema = pytest.mark.skipif(
    not hasattr(t, "drop_undeclared_columns"),
    reason="pinned-schema mode not in this checkout")


@pinned_schema
def test_drop_undeclared_columns_pins_to_metadata_schema():
    df = pl.DataFrame({"a": [1], "b": [2], "extra_22_col": [3]})
    out, info = t.drop_undeclared_columns(df, {"a": {}, "b": {}})
    assert out.columns == ["a", "b"]
    assert info["dropped"] == ["extra_22_col"]


@pinned_schema
def test_drop_undeclared_runs_before_combining_matters():
    # An undeclared med variant must be droppable BEFORE the combiner
    # would fold it into a feature the pinned schema never described.
    df = pl.DataFrame({
        "med_admins_acei_any": _bool([True]),
        "med_requests_acei_any": _bool([False]),
        "med_requests_newdrug_any": _bool([True]),  # not in pinned metadata
    })
    pinned = {"med_admins_acei_any": {}, "med_requests_acei_any": {}}
    df2, _ = t.drop_undeclared_columns(df, pinned)
    out, info = t.combine_medications(df2)
    assert info["features_created"] == ["med_acei"]
    assert "med_newdrug" not in out.columns


# --- NYHA ---

def _nyha_meta(*columns):
    concept = [
        {"code": "LA28404-4", "display": "NYHA-Class-I"},
        {"code": "LA28405-1", "display": "NYHA-Class-II"},
        {"code": "LA28406-9", "display": "NYHA-Class-III"},
        {"code": "LA28407-7", "display": "NYHA-Class-IV"},
    ]
    return {c: {"dataType": "NOMINAL", "valueSet": {"concept": concept}} for c in columns}


def test_encode_nyha_primary_column():
    df = pl.DataFrame({t.NYHA_COLUMN: ["LA28404-4", "LA28407-7", None]})
    out, info = t.encode_nyha(df, _nyha_meta(t.NYHA_COLUMN))
    assert out[t.NYHA_COLUMN].to_list() == [1, 4, None]
    assert not info["skipped"]


def test_encode_nyha_encodes_both_22_columns():
    df = pl.DataFrame({
        t.NYHA_COLUMN: ["LA28404-4", "LA28405-1"],
        "nyha_nyha": ["LA28406-9", None],
    })
    out, info = t.encode_nyha(df, _nyha_meta(t.NYHA_COLUMN, "nyha_nyha"))
    assert out[t.NYHA_COLUMN].to_list() == [1, 2]
    assert out["nyha_nyha"].to_list() == [3, None]
    assert set(info["maps"]) == {t.NYHA_COLUMN, "nyha_nyha"}


def test_encode_nyha_base_column_falls_back_to_primary_map():
    # 2.2 metadata may declare nyha_nyha without repeating the valueSet;
    # the primary column's map is the correct fallback.
    df = pl.DataFrame({
        t.NYHA_COLUMN: ["LA28404-4"],
        "nyha_nyha": ["LA28407-7"],
    })
    meta = _nyha_meta(t.NYHA_COLUMN)
    meta["nyha_nyha"] = {"dataType": "NOMINAL"}  # no valueSet of its own
    out, _ = t.encode_nyha(df, meta)
    assert out["nyha_nyha"].to_list() == [4]


def test_encode_nyha_unmapped_code_raises():
    df = pl.DataFrame({t.NYHA_COLUMN: ["LA28404-4", "NOT-A-CODE"]})
    with pytest.raises(ValueError, match="NOT-A-CODE"):
        t.encode_nyha(df, _nyha_meta(t.NYHA_COLUMN))


def test_impute_nyha_missing_fills_every_nyha_column():
    df = pl.DataFrame({
        t.NYHA_COLUMN: pl.Series([1, None], dtype=pl.Int64),
        "nyha_nyha": pl.Series([None, 4], dtype=pl.Int64),
    })
    out, info = t.impute_nyha_missing(df)
    assert out[t.NYHA_COLUMN].to_list() == [1, t.NYHA_MISSING_SENTINEL]
    assert out["nyha_nyha"].to_list() == [t.NYHA_MISSING_SENTINEL, 4]
    assert info["filled"] == 2

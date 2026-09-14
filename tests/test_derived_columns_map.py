"""
The derived-columns semantic map must mirror the preprocessing combiners
exactly -- including the 2.2 requests-only medication indicators.
"""

from make_derived_columns_map import condition_groups, medication_groups


def test_medication_groups_union_of_admins_and_requests():
    cols = [
        "med_admins_acei_any", "med_requests_acei_any",
        "med_requests_activeDuringEncounter_bb_any",          # requests-only (2.2)
        "med_admins_history_sglt2i_any", "med_requests_history_sglt2i_any",
    ]
    g = medication_groups(cols)
    assert g["med_acei"] == ["med_admins_acei_any", "med_requests_acei_any"]
    assert g["med_activeDuringEncounter_bb"] == ["med_requests_activeDuringEncounter_bb_any"]
    assert g["med_sglt2i_history"] == [
        "med_admins_history_sglt2i_any", "med_requests_history_sglt2i_any"]


def test_condition_groups_all_windows():
    cols = ["conditions_ckd_pre_dc_any", "conditions_ckd_during_pET_any",
            "conditions_ckd_timeFromEarliest_first"]
    g = condition_groups(cols)
    assert g == {"conditions_ckd": ["conditions_ckd_pre_dc_any",
                                    "conditions_ckd_during_pET_any"]}


def test_groups_mirror_the_actual_combiner():
    # The map must never drift from what preprocessing actually does.
    import polars as pl

    from pipeline.steps.preprocess.transforms import combine_medications

    cols = ["med_admins_acei_any", "med_requests_acei_any",
            "med_requests_activeDuringEncounter_bb_any"]
    df = pl.DataFrame({c: pl.Series([True], dtype=pl.Boolean) for c in cols})
    _, info = combine_medications(df)
    assert sorted(info["features_created"]) == sorted(medication_groups(cols))

"""
Post-processed candidates drop (and optionally top up) rows, so anything
linked to the source file by record index -- the generated clinical notes
-- needs the provenance of every output row. The row map is that link.
"""

import pytest

postprocess_candidate = pytest.importorskip(
    "postprocess_candidate", reason="post-processor imports the generation stack")


def _map(source_rows, n=6):
    return postprocess_candidate.build_row_map(
        "output/generate/DT4H_Synthetic_tvae_seed0.csv", n, source_rows,
        snapped=False, filtered=True, threshold=0.07, columns_used=254)


def test_filter_only_map_lists_kept_and_dropped_rows():
    m = _map([0, 2, 3, 5])
    assert m["input_file"] == "DT4H_Synthetic_tvae_seed0.csv"   # basename only
    assert m["input_rows"] == 6 and m["output_rows"] == 4
    assert m["source_row_of_output_row"] == [0, 2, 3, 5]
    assert m["dropped_input_rows"] == [1, 4]
    assert m["rows_kept_from_input"] == 4 and m["rows_topped_up"] == 0


def test_top_up_rows_have_no_source():
    m = _map([0, 2, 3, 5, None, None])
    assert m["output_rows"] == 6
    assert m["source_row_of_output_row"][-2:] == [None, None]
    assert m["rows_topped_up"] == 2
    assert m["dropped_input_rows"] == [1, 4]


def test_no_filter_is_the_identity_map():
    m = postprocess_candidate.build_row_map(
        "x.csv", 3, [0, 1, 2], snapped=True, filtered=False,
        threshold=None, columns_used=10)
    assert m["source_row_of_output_row"] == [0, 1, 2]
    assert m["dropped_input_rows"] == []
    assert m["distance_threshold"] is None

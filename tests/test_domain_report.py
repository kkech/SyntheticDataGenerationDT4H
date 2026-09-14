"""
Domain-violation reporting and --clip-to-domain: sentinel-aware violation
counting, the coarsened disclosure of observed extremes, and (where
snsynth is installed) the clip path itself.
"""

import pandas as pd
import pytest

from pipeline.steps.generate.synthesizers.smartnoise_models import (
    coarse_observed_span,
    compute_domain_report,
    sentinel_public_bound,
)

DOMAINS = {"lab": {"lo": 0.0, "hi": 100.0}}


def test_values_inside_domain_do_not_violate():
    df = pd.DataFrame({"lab": [1.0, 50.0, 100.0]})
    r = compute_domain_report(df, ["lab"], DOMAINS, encoding={})
    assert r["columns"]["lab"]["violates"] is False
    assert r["missing"] == [] and r["degenerate"] == []


def test_artifact_above_hi_counts():
    df = pd.DataFrame({"lab": [50.0, 430.0]})  # BMI-430-style entry artifact
    r = compute_domain_report(df, ["lab"], DOMAINS, encoding={})
    c = r["columns"]["lab"]
    assert c["violates"] and c["n_above"] == 1 and c["n_below"] == 0


def test_sentinel_cells_are_not_violations():
    # Sentinels sit BELOW pub_lo by design; for an encoded column the
    # violation bound is the public-rule sentinel bound, so encoding
    # missingness must never read as a domain violation.
    s_pub = sentinel_public_bound(0.0, 100.0)
    df = pd.DataFrame({"lab": [s_pub, 50.0]})
    r = compute_domain_report(df, ["lab"], DOMAINS,
                              encoding={"lab": {"sentinel": s_pub}})
    assert r["columns"]["lab"]["violates"] is False
    # ...but WITHOUT the encoding entry the same value is below pub_lo:
    r2 = compute_domain_report(df, ["lab"], DOMAINS, encoding={})
    assert r2["columns"]["lab"]["n_below"] == 1


def test_inflated_data_sentinel_below_public_bound_counts():
    # An extreme artifact inflates the OBSERVED range, so the data-side
    # sentinel (min - 0.25*range) can undershoot the public-rule bound;
    # those cells must be reported (and are what --clip-to-domain pulls
    # back up onto the public sentinel bound, still decoding to null).
    s_pub = sentinel_public_bound(0.0, 100.0)
    inflated_sentinel = s_pub - 500.0
    df = pd.DataFrame({"lab": [inflated_sentinel, 50.0]})
    r = compute_domain_report(df, ["lab"], DOMAINS,
                              encoding={"lab": {"sentinel": inflated_sentinel}})
    c = r["columns"]["lab"]
    assert c["violates"] and c["n_below"] == 1
    assert c["lower"] == pytest.approx(s_pub)


def test_missing_and_degenerate_domains_reported():
    df = pd.DataFrame({"nodomain": [1.0], "bad": [1.0]})
    r = compute_domain_report(df, ["nodomain", "bad"],
                              {"bad": {"lo": 5.0, "hi": 5.0}}, encoding={})
    assert r["missing"] == ["nodomain"]
    assert r["degenerate"] and "bad" in r["degenerate"][0]
    assert r["columns"] == {}


def test_sentinel_public_bound_is_at_or_below_any_contained_data_sentinel():
    # data sentinel = obs_min - max(0.25*obs_range, floor); containment
    # (pub_lo <= obs_min, obs_range <= pub_range) implies the public-rule
    # bound is always <= the data sentinel, so real sentinels never violate.
    pub_lo, pub_hi = 0.0, 100.0
    for obs_min, obs_max in [(0.0, 100.0), (10.0, 90.0), (40.0, 41.0)]:
        data_sentinel = obs_min - max((obs_max - obs_min) * 0.25, 1.0)
        assert sentinel_public_bound(pub_lo, pub_hi) <= data_sentinel


def test_coarse_observed_span_never_discloses_exact_extremes():
    r = {"col_min": 31.75, "col_max": 807.3}
    span = coarse_observed_span(r)
    assert "31.75" not in span and "807.3" not in span
    # outward: reported low <= true min, reported high >= true max
    lo, hi = span.strip("[]").replace("~", "").split(",")
    assert float(lo) <= 31.75 and float(hi) >= 807.3


def test_clip_pulls_cells_onto_declared_bounds_and_spares_caller_frame():
    pytest.importorskip("snsynth")
    from pipeline.steps.generate.synthesizers.smartnoise_models import _SmartNoiseBase

    inst = _SmartNoiseBase.__new__(_SmartNoiseBase)
    inst.params = {}
    df = pd.DataFrame({"lab": [-5.0, 50.0, 430.0]})
    original = df.copy()
    work = df.copy()  # fit() hands _bound_constraints a private copy when clipping
    constraints, clip_report = inst._bound_constraints(
        work, ["lab"], DOMAINS, encoding={}, clip=True)
    assert clip_report == {"lab": 2}
    assert work["lab"].tolist() == [0.0, 50.0, 100.0]
    pd.testing.assert_frame_equal(df, original)  # caller's frame untouched


def test_no_clip_raises_before_fitting():
    pytest.importorskip("snsynth")
    from pipeline.steps.generate.synthesizers.smartnoise_models import _SmartNoiseBase

    inst = _SmartNoiseBase.__new__(_SmartNoiseBase)
    inst.params = {}
    df = pd.DataFrame({"lab": [430.0]})
    with pytest.raises(ValueError, match="clip-to-domain"):
        inst._bound_constraints(df, ["lab"], DOMAINS, encoding={}, clip=False)

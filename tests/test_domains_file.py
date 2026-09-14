"""
Consistency between the committed public-domain declaration and what the
pipeline actually models: every NYHA column is ordinal-encoded into a
numeric column, so each needs its own domain entry -- a gap here fails
every DP run at a site with that column in its schema.
"""

import json
import os

from pipeline.steps.preprocess.transforms import NYHA_COLUMNS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_nyha_column_has_a_domain_entry():
    with open(os.path.join(REPO, "public_domains.json")) as f:
        domains = json.load(f)["domains"]
    missing = [c for c in NYHA_COLUMNS if c not in domains]
    assert not missing, (
        f"NYHA column(s) without a public-domain entry: {missing}. "
        f"Ordinal-encoded NYHA is numeric in the DP models' continuous "
        f"split, so each encoded column needs its own [0, 4] entry.")


def test_nyha_domain_entries_are_the_ordinal_scale():
    with open(os.path.join(REPO, "public_domains.json")) as f:
        domains = json.load(f)["domains"]
    for c in NYHA_COLUMNS:
        spec = domains.get(c)
        if spec is None:
            continue  # absence is the other test's failure
        assert spec["lo"] == 0.0 and spec["hi"] == 4.0, (
            f"{c}: expected the ordinal scale [0, 4] "
            f"(0 = not-assessed sentinel), got [{spec['lo']}, {spec['hi']}]")

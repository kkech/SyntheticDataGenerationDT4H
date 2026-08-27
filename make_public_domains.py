"""
Build the TEMPLATE public numeric-domain declaration that DP runs require.

    python make_public_domains.py            # write public_domains.json (template)
    python make_public_domains.py --force    # overwrite even a reviewed file

WHY THIS FILE EXISTS
--------------------
A differentially private synthesizer needs a per-column domain [lo, hi]
to bound each numeric column's sensitivity. Deriving that domain from
the training data -- which is what this pipeline used to do -- is a
formal DP violation: the bounds themselves are an unnoised function of
private records, so the released mechanism is not (ε, δ)-DP no matter
what ε the synthesizer was given. The standard fix is the one taken
here: the domain is declared A PRIORI, by a human, from public /
clinical knowledge, and released alongside the data as part of the
mechanism's specification.

This tool does NOT produce that declaration. It produces a starting
point: proposals rounded OUTWARD from ranges that are already public
(the committed preprocessing profile and the committed sentinel
encoding map), so a reviewer edits plausible numbers rather than typing
329 ranges from scratch. The proposals are deliberately generous --
rounded to one significant figure away from the observed range -- so
that a reviewer who accepts one is accepting a range wider than the
cohort, not a range fitted to it.

The file is written with `"reviewed": false`. The generate step REFUSES
to run any DP synthesizer until a human has gone through every range,
edited what is not clinically plausible, and set `"reviewed": true`.
That flag is the human's signature on the released domain declaration;
this tool never sets it, and never overwrites a file that carries it
unless --force is passed.

HONEST LIMITATION: seeding the proposals from observed ranges means a
reviewer who rubber-stamps the file has effectively released a
one-significant-figure function of the data. Rounding outward blunts
that (a 1-sf bound is coarse, and the observed min/max of every encoded
numeric column is already published in the encoding map this repo
commits), but it does not make review optional -- it makes it cheap.
Ranges that a clinician would not defend from domain knowledge alone
must be replaced, not approved.
"""

import argparse
import hashlib
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.steps.preprocess.transforms import (  # noqa: E402
    NUMERIC_ENCODING_FILENAME,
)

PROFILE_FILENAME = "DT4H_Preprocessed_Column_Analysis.json"

NOTE = (
    "TEMPLATE -- NOT YET A PRIVACY GUARANTEE. Each entry is the a-priori PUBLIC "
    "domain [lo, hi] declared for a numeric column: the range a value could take "
    "for any patient who might have been in the cohort, decided from clinical / "
    "measurement knowledge, NOT from this dataset. DP runs bound each column to "
    "this domain, so a range fitted to the observed data leaks the data. "
    "REVIEW WORKFLOW: (1) read every range and replace anything you would not "
    "defend without looking at the cohort -- the proposals here were seeded from "
    "already-published observed ranges, rounded outward to one significant figure, "
    "purely to save typing; (2) widen anything clinically implausible (a domain "
    "wider than the cohort costs utility, a domain narrower than the cohort is "
    "both a privacy leak and a fit-time error); (3) set \"reviewed\": true. "
    "The generate step refuses to fit any DP synthesizer while reviewed is false, "
    "and fails at fit time (before hours of training) if any training value falls "
    "outside the declared domain. This file is released with the data: it is part "
    "of the mechanism specification a reader needs to check the (epsilon, delta) "
    "claim. BEFORE RELEASING the file, delete the observed_min/observed_max keys "
    "from every entry: they are exact values of real patients, kept here only so "
    "the reviewer can see that a proposal covers the cohort. Only lo/hi (and "
    "sentinel_encoded) are part of the public declaration."
)


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _floor_sig(x: float, sig: int = 1) -> float:
    """Round DOWN to `sig` significant figures (towards -inf)."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    scale = 10.0 ** (math.floor(math.log10(abs(x))) - (sig - 1))
    return math.floor(x / scale) * scale


def _ceil_sig(x: float, sig: int = 1) -> float:
    """Round UP to `sig` significant figures (towards +inf)."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    scale = 10.0 ** (math.floor(math.log10(abs(x))) - (sig - 1))
    return math.ceil(x / scale) * scale


def propose_domain(lo: float, hi: float) -> tuple[float, float]:
    """
    Propose a public [lo, hi] that strictly CONTAINS the observed range.

    Outward rounding to one significant figure: 33.9 -> 30, 241 -> 300.
    A positive lower bound that is small relative to the upper bound is
    proposed as 0 instead, because for the measurement columns in this
    cohort "could be near zero" is the clinically defensible statement,
    not "never below the smallest value we happened to observe".

    Returns floats cleaned of binary-representation dust (0.30000000004),
    and is asserted by the caller to contain [lo, hi] -- if float
    rounding ever went the wrong way we fall back to the raw observed
    value, which is no worse than today's behaviour.
    """
    lo_pub = _floor_sig(lo)
    hi_pub = _ceil_sig(hi)
    if lo > 0 and lo <= 0.05 * hi_pub:
        # A positive minimum that sits in the bottom few percent of the
        # range says more about who got measured than about what the
        # quantity can be; propose the natural floor instead.
        lo_pub = 0.0

    lo_pub = round(lo_pub, 12)
    hi_pub = round(hi_pub, 12)
    if lo_pub > lo:  # float dust; never propose a bound inside the data
        lo_pub = lo
    if hi_pub < hi:
        hi_pub = hi
    if hi_pub <= lo_pub:  # degenerate/constant column
        hi_pub = lo_pub + 1.0
    return float(lo_pub), float(hi_pub)


def build_template(profile: dict, encoding: dict) -> dict:
    """
    One proposal per numeric, non-constant column of the preprocessed
    frame. For sentinel-encoded columns the OBSERVED range comes from the
    encoding map (the profile's min is the sentinel, not a real value);
    for the rest it is the profile's own min/max.
    """
    domains = {}
    for col, info in profile.get("columns", {}).items():
        if info.get("inferred_type") != "numeric" or info.get("is_constant"):
            continue
        enc = encoding.get(col)
        if enc is not None:
            obs_lo, obs_hi = float(enc["min_observed"]), float(enc["max_observed"])
        elif info.get("min") is None or info.get("max") is None:
            continue
        else:
            obs_lo, obs_hi = float(info["min"]), float(info["max"])

        lo_pub, hi_pub = propose_domain(obs_lo, obs_hi)
        assert lo_pub <= obs_lo and hi_pub >= obs_hi, col  # outward by construction
        domains[col] = {
            "lo": lo_pub,
            "hi": hi_pub,
            # Review context only -- the DP bound uses lo/hi alone. These
            # two numbers are already published (encoding map / profile),
            # so carrying them here discloses nothing new; delete them if
            # you would rather the released declaration name no observed
            # quantity at all.
            "observed_min": obs_lo,
            "observed_max": obs_hi,
            "sentinel_encoded": enc is not None,
        }
    return domains


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the template public numeric-domain declaration for DP runs.")
    parser.add_argument("--out", help="Output path (default: config.public_domains_path).")
    parser.add_argument("--profile", help=f"Path to {PROFILE_FILENAME}.")
    parser.add_argument("--encoding", help=f"Path to {NUMERIC_ENCODING_FILENAME}.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output even if it is already marked reviewed. "
                             "Doing so DISCARDS a human review; the new file comes back "
                             "as reviewed:false and must be reviewed again.")
    args = parser.parse_args()

    config = PipelineConfig()
    out_path = args.out or config.public_domains_path
    profile_path = args.profile or os.path.join(
        config.step_dir("profile_preprocessed_data"), PROFILE_FILENAME)
    encoding_path = args.encoding or os.path.join(
        config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)

    for path, what in ((profile_path, "preprocessed profile"),
                       (encoding_path, "numeric encoding map")):
        if not os.path.exists(path):
            print(f"❌ {what} not found: {path}\n"
                  f"   Run the preprocess and profile_preprocessed_data steps first.")
            return 2

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if existing.get("reviewed") is True and not args.force:
            print(f"❌ {out_path} is already marked reviewed:true -- refusing to "
                  f"overwrite a human-reviewed domain declaration.\n"
                  f"   Pass --force if you really mean to discard that review "
                  f"(the regenerated file comes back reviewed:false).")
            return 3
        print(f"⚠️  Overwriting existing {os.path.basename(out_path)} "
              f"(reviewed={existing.get('reviewed')}).")

    with open(profile_path) as f:
        profile = json.load(f)
    with open(encoding_path) as f:
        encoding = json.load(f)

    domains = build_template(profile, encoding)
    doc = {
        "reviewed": False,
        "note": NOTE,
        "generated_by": "make_public_domains.py",
        "proposal_rule": "observed range rounded outward to 1 significant figure "
                         "(a small positive floor is proposed as 0)",
        "sources": {
            "profile": {"path": profile_path, "sha256": _sha256(profile_path)},
            "encoding_map": {"path": encoding_path, "sha256": _sha256(encoding_path)},
        },
        "domains": domains,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    n_sent = sum(1 for d in domains.values() if d["sentinel_encoded"])
    print(f"✅ Wrote {len(domains)} proposed public domain(s) -> {out_path}")
    print(f"   {n_sent} sentinel-encoded column(s); {len(domains) - n_sent} without "
          f"missingness encoding.")
    print("   NEXT: review every range for clinical plausibility, edit as needed, "
          "then set \"reviewed\": true. DP runs refuse to start until you do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

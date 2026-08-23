"""
Release gate: the automated go/no-go check a candidate synthetic file
must pass before it is distributed.

    python release_gate.py --file output/generate/DT4H_Synthetic_<run>.csv

Checks, each mandatory:
  1. schema        -- columns are a subset of the released schema, no extras;
  2. freshness     -- the file re-decodes to itself (no undecoded sentinels
                      from an older pipeline version);
  3. leakage       -- zero verbatim reproductions of training records;
  4. coherence     -- rule-violation rate within tolerance of the real
                      holdout baseline (rules from the committed rule set);
  5. distance      -- the share of sampled records closer to a training
                      record than the holdout p5 threshold must not exceed
                      twice the natural rate (5% of real unseen patients
                      fall below their own p5 by construction, so zero
                      tolerance would fail real data too). Spot check on
                      a sample for speed.

Writes DT4H_Release_Gate_<name>.md next to the file and exits non-zero
on FAIL, so it can gate a scripted publishing flow.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case
from pipeline.steps.coherence import rules as R
from pipeline.steps.generate import leakage
from pipeline.steps.generate.step import GenerateStep
from pipeline.steps.privacy.distance import build_encoder, nearest_two_distances

# Coherence policy: a candidate passes if its rule-violation rate is
# within one order of magnitude of the real holdout's own rate, or
# within 1 percentage point absolute -- whichever is more permissive.
# Stated as policy so it can be tightened per release decision.
COHERENCE_MULTIPLIER = 10.0
COHERENCE_ABSOLUTE = 0.01
DCR_SAMPLE = 500
# Distance policy: 5% of real unseen patients fall below the holdout p5
# threshold BY CONSTRUCTION, so a perfect generator would land at ~5% too
# and zero tolerance would reject real data itself. A candidate passes if
# its share below the threshold is at most this multiple of the natural 5%.
DISTANCE_NATURAL_SHARE = 0.05
DISTANCE_SHARE_MULTIPLIER = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic-file release gate.")
    parser.add_argument("--file", required=True, help="Candidate DT4H_Synthetic_*.csv")
    args = parser.parse_args()
    config = PipelineConfig()

    candidate = pd.read_csv(args.file, low_memory=False)
    train = pl.read_parquet(config.train_output_path).to_pandas()
    holdout = pl.read_parquet(config.holdout_output_path).to_pandas()
    # The candidate is in DECODED (released) space, so the leakage and
    # coherence comparisons must use decoded frames too; the sentinel-
    # space train frame is kept for the distance encoder, which
    # re-encodes decoded nulls itself.
    train_decoded, _ = GenerateStep._decode_numeric_missing(train.copy(), config)
    holdout_decoded, _ = GenerateStep._decode_numeric_missing(holdout.copy(), config)
    candidate, respelled = align_categorical_case(candidate, train_decoded)
    if respelled:
        print(f"  note: {sum(respelled.values())} categorical cell(s) normalized to the "
              f"real schema's spellings after CSV parsing (expected for boolean "
              f"columns, which pandas parses to bool dtype)")
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  {'✅' if passed else '❌'} {name}: {detail}")

    print(f"Release gate for {args.file} ({candidate.shape[0]} x {candidate.shape[1]}):")

    extra = [c for c in candidate.columns if c not in train.columns]
    check("schema", not extra and len(candidate) > 0,
          f"{len(candidate.columns)} columns, {len(extra)} not in released schema")

    from pipeline.common.representation_audit import audit_representation, summarize as rep_summary

    rep = audit_representation(candidate, train_decoded)
    check("representation", rep["clean"] and not rep["categorical_nulls"], rep_summary(rep))

    _, would_change = GenerateStep._decode_numeric_missing(candidate.copy(), config)
    stale = sum(would_change.values())
    check("freshness", stale == 0,
          f"{stale} cell(s) an up-to-date decode would change")

    leak = leakage.check_exact_duplicates(candidate, train_decoded)
    check("leakage", leak.get("exact_duplicates_of_training_rows", 1) == 0,
          f"{leak.get('exact_duplicates_of_training_rows')} verbatim training row(s) "
          f"(compared in released/decoded space)")

    rules_path = os.path.join(config.step_dir("coherence"), "DT4H_Coherence_Rules.json")
    if os.path.exists(rules_path):
        with open(rules_path) as f:
            ruleset = json.load(f)["rules"]
        cand_summary = R.summarize_rule_results(R.evaluate_rules(candidate, ruleset))
        hold_summary = R.summarize_rule_results(R.evaluate_rules(holdout_decoded, ruleset))
        rate = cand_summary["overall_violation_rate"] or 0.0
        base = hold_summary["overall_violation_rate"] or 0.0
        threshold = max(COHERENCE_MULTIPLIER * base, base + COHERENCE_ABSOLUTE)
        check("coherence", rate <= threshold,
              f"violation rate {rate} vs holdout baseline {base} "
              f"(threshold {round(threshold, 5)} = max({COHERENCE_MULTIPLIER:g}x baseline, "
              f"baseline+{COHERENCE_ABSOLUTE}))")
    else:
        check("coherence", False, "no committed rule set -- run the coherence step first")

    priv_path = os.path.join(config.step_dir("privacy"), "DT4H_Privacy_Assessment.json")
    if os.path.exists(priv_path):
        with open(priv_path) as f:
            p5 = json.load(f)["holdout_baseline"]["dcr_p5"]
        enc_path = os.path.join(config.step_dir("preprocess"), "DT4H_Numeric_Missing_Encoding.json")
        encoding = json.load(open(enc_path)) if os.path.exists(enc_path) else {}
        encode, _, _ = build_encoder(train, encoding)
        rng = np.random.default_rng(0)
        sample = candidate.iloc[rng.choice(len(candidate), min(DCR_SAMPLE, len(candidate)),
                                           replace=False)].copy()
        missing = [c for c in train.columns if c not in sample.columns]
        if missing:
            sample = pd.concat([sample, pd.DataFrame(pd.NA, index=sample.index,
                                                     columns=missing)], axis=1)
        t_num, t_cat = encode(train)
        s_num, s_cat = encode(sample)
        d1, _ = nearest_two_distances(s_num, s_cat, t_num, t_cat)
        too_close = int((d1 < p5).sum())
        share = too_close / len(sample)
        limit = DISTANCE_SHARE_MULTIPLIER * DISTANCE_NATURAL_SHARE
        check("distance", share <= limit,
              f"{too_close}/{len(sample)} sampled record(s) ({share:.1%}) closer than "
              f"the holdout p5 threshold ({p5}); policy limit {limit:.0%} = "
              f"{DISTANCE_SHARE_MULTIPLIER:g}x the natural 5% share")
    else:
        check("distance", False, "no committed privacy assessment -- run the privacy step first")

    passed = all(c["passed"] for c in checks)
    verdict = "PASS -- cleared for release" if passed else "FAIL -- DO NOT RELEASE"
    print(f"\n{'✅' if passed else '🚫'} {verdict}")

    name = os.path.basename(args.file).replace(".csv", "")
    report = os.path.join(os.path.dirname(args.file), f"DT4H_Release_Gate_{name}.md")
    with open(report, "w") as f:
        f.write(f"# Release Gate: {name}\n\n"
                f"Evaluated {datetime.now(timezone.utc).isoformat()}\n\n"
                f"**{verdict}**\n\n| check | result | detail |\n|---|---|---|\n")
        for c in checks:
            f.write(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |\n")
    print(f"Report -> {report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

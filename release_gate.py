"""
Release gate: the automated go/no-go check a candidate synthetic file
must pass before it is distributed.

    python release_gate.py --file output/generate/DT4H_Synthetic_<run>.csv
    python release_gate.py --file <...> --policy controlled --note "consortium 2026-08-27"

Checks, each mandatory:
  1. schema        -- columns are a subset of the released schema, no extras;
  2. freshness     -- the file re-decodes to itself (no undecoded sentinels
                      from an older pipeline version);
  3. leakage       -- zero verbatim reproductions of training records;
  4. coherence     -- rule-violation rate within tolerance of the real
                      holdout baseline (rules from the committed rule set);
  5. distance      -- the share of sampled records closer to a training
                      record than the holdout p5 threshold must not exceed
                      the policy's multiple of the natural rate (5% of real
                      unseen patients fall below their own p5 by
                      construction, so zero tolerance would fail real data
                      too). Spot check on a sample for speed. Width-limited
                      candidates are measured on the shared column subset
                      against a p5 recomputed on that same subset -- never
                      against the full-width threshold, which NA-padding
                      would let them pass vacuously.

Checks 4 and 5 are thresholded by the selected POLICY (see POLICIES
below); the others are absolute facts. Every report states the verdict
under ALL policies, so a relaxed pass is never readable without its
stricter counterpart.

Writes DT4H_Release_Gate_<name>.md and DT4H_Release_Gate_<name>.json
next to the file, and exits non-zero on FAIL under the selected policy,
so it can gate a scripted publishing flow.
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

# --- policy ---
#
# Two of the six checks are thresholded rather than absolute (schema,
# freshness, leakage and representation are pass/fail facts). Their
# thresholds are POLICY, not measurement, so they are named, versioned
# and stamped into every report rather than edited in place:
#
#   coherence -- how far above the real holdout's own rule-violation rate
#     a synthetic file may sit. The holdout is real patient data and
#     still violates the rule set (mostly rare, one-off inconsistencies),
#     so this is a multiple of a measured baseline, not of zero.
#   distance -- the share of records closer to a training record than the
#     holdout p5 threshold. 5% of real unseen patients fall below that
#     threshold BY CONSTRUCTION, so a perfect generator lands at ~5% too
#     and zero tolerance would reject real data itself.
#
# The distance limit is the privacy-protective one; the coherence limit
# is a data-quality one. They are deliberately separated so a release
# decision can accept lower clinical coherence WITHOUT touching the
# memorization margin -- which is what the 'controlled' policy does.
#
# Whichever policy is selected, every report states the verdict under
# ALL policies, so a relaxed pass is never readable without its stricter
# counterpart. Changing these numbers is a consortium decision; record it
# with --note so the report carries the authority for the change.
POLICIES = {
    "release": {
        "coherence_multiplier": 10.0,
        "coherence_absolute": 0.01,
        "distance_multiplier": 2.0,
        "intent": "open or brokered release of a file that leaves the enclave",
    },
    "controlled": {
        "coherence_multiplier": 20.0,
        "coherence_absolute": 0.03,
        "distance_multiplier": 2.0,
        "intent": "controlled-access sharing under a data-use agreement, where a "
                  "recipient is bound by contract and the file is not public",
    },
}
DEFAULT_POLICY = "release"

DCR_SAMPLE = 500
DISTANCE_NATURAL_SHARE = 0.05



def _require_csv(path: str) -> None:
    """Fail with a plain message when the file is not a text CSV -- the
    classic slip is handing the model .pkl to --file (pickle starts with
    byte 0x80) or a parquet (starts with PAR1), which pandas otherwise
    reports as an opaque UnicodeDecodeError from the C parser."""
    import os as _os
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head[:1] == b"\x80":
        raise SystemExit(f"❌ {path} is a PICKLE, not a CSV. --file takes the synthetic "
                         f"CSV; the fitted model .pkl goes to --model.")
    if head == b"PAR1":
        raise SystemExit(f"❌ {path} is a PARQUET file, not a CSV. --file takes the "
                         f"synthetic DT4H_Synthetic_*.csv.")
    if _os.path.basename(path).endswith(".pkl"):
        raise SystemExit(f"❌ {path} looks like a model pickle. --file takes the synthetic CSV.")

def _coherence_threshold(policy: dict, baseline: float) -> float:
    return max(policy["coherence_multiplier"] * baseline,
               baseline + policy["coherence_absolute"])


def _distance_limit(policy: dict) -> float:
    return policy["distance_multiplier"] * DISTANCE_NATURAL_SHARE


def _verdicts_by_policy(checks: list, measured: dict) -> dict:
    """The verdict this same measurement produces under every named
    policy. The absolute checks (schema, representation, freshness,
    leakage) are policy-independent, so only coherence and distance are
    re-thresholded -- and a file failing an absolute check fails
    everywhere, which is the point of stating them all."""
    absolute = [c for c in checks if c["check"] not in ("coherence", "distance")]
    absolute_ok = all(c["passed"] for c in absolute)
    out = {}
    for name, policy in POLICIES.items():
        per_check = {}
        if "coherence_rate" in measured:
            threshold = _coherence_threshold(policy, measured["coherence_baseline"])
            per_check["coherence"] = {
                "passed": measured["coherence_rate"] <= threshold,
                "threshold": round(threshold, 5),
            }
        if "distance_share" in measured:
            limit = _distance_limit(policy)
            per_check["distance"] = {
                # A degenerate subset threshold (p5 == 0) fails under every
                # policy: no share can evidence safety against it.
                "passed": (not measured.get("distance_degenerate")
                           and measured["distance_share"] <= limit),
                "threshold": limit,
            }
        # A missing measurement means the check could not be evaluated at
        # all (no rule set, no privacy assessment); that is a FAIL under
        # every policy, exactly as the check itself recorded.
        complete = len(per_check) == 2
        out[name] = {
            "verdict": "PASS" if (absolute_ok and complete
                                  and all(v["passed"] for v in per_check.values()))
                       else "FAIL",
            "checks": per_check,
            "intent": policy["intent"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic-file release gate.")
    parser.add_argument("--file", help="Candidate DT4H_Synthetic_*.csv")
    parser.add_argument("--all", action="store_true",
                        help="Gate every DT4H_Synthetic_*.csv and DT4H_Candidate_*.csv in "
                             "output/generate/ and print a both-policies summary table. "
                             "Per-file reports are written as usual; exit code is 0 when "
                             "the sweep completes (verdicts live in the reports).")
    parser.add_argument("--policy", choices=sorted(POLICIES), default=DEFAULT_POLICY,
                        help=f"Threshold policy for the coherence and distance checks "
                             f"(default: {DEFAULT_POLICY}). Every report states the verdict "
                             f"under all policies regardless.")
    parser.add_argument("--coherence-multiplier", type=float,
                        help="Override the policy's coherence multiple of the holdout "
                             "baseline. Recorded in the report as a custom policy.")
    parser.add_argument("--distance-multiplier", type=float,
                        help="Override the policy's multiple of the natural 5%% distance "
                             "share. THIS IS THE PRIVACY MARGIN -- recorded in the report "
                             "as a custom policy and flagged in the verdict.")
    parser.add_argument("--note",
                        help="Free text recorded in the report: who authorized a "
                             "non-default policy, and when (e.g. a consortium decision).")
    args = parser.parse_args()

    policy_name = args.policy
    policy = dict(POLICIES[policy_name])
    overrides = {}
    if args.coherence_multiplier is not None:
        overrides["coherence_multiplier"] = args.coherence_multiplier
    if args.distance_multiplier is not None:
        overrides["distance_multiplier"] = args.distance_multiplier
    if overrides:
        policy.update(overrides)
        policy["intent"] = (f"custom thresholds over the '{policy_name}' policy "
                            f"({', '.join(f'{k}={v:g}' for k, v in overrides.items())})")
        policy_name = f"custom (from {policy_name})"
    print(f"Policy: {policy_name} -- coherence "
          f"{policy['coherence_multiplier']:g}x holdout baseline "
          f"(or +{policy['coherence_absolute']:.0%} absolute), distance "
          f"{policy['distance_multiplier']:g}x the natural "
          f"{DISTANCE_NATURAL_SHARE:.0%} share")
    if args.note:
        print(f"Note: {args.note}")
    config = PipelineConfig()

    if args.all:
        import glob as _glob
        gen = config.step_dir("generate")
        paths = sorted(_glob.glob(os.path.join(gen, "DT4H_Synthetic_*.csv"))) + \
                sorted(_glob.glob(os.path.join(gen, "DT4H_Candidate_*.csv")))
        if not paths:
            print(f"No candidate CSVs found in {gen}.")
            return 1
        print(f"Gating {len(paths)} file(s) under every policy...\n")
        rows = []
        for p in paths:
            print("\n" + "#" * 70 + f"\n# {os.path.basename(p)}\n" + "#" * 70)
            try:
                _gate(p, policy, policy_name, args, config)
            except SystemExit as e:
                print(f"  skipped: {e}")
            name = os.path.basename(p).replace(".csv", "")
            sidecar = os.path.join(os.path.dirname(p), f"DT4H_Release_Gate_{name}.json")
            try:
                with open(sidecar) as fh:
                    d = json.load(fh)
                v = d.get("verdict_by_policy", {})
                fails = [c["check"] for c in d.get("checks", []) if not c.get("passed")]
                rows.append((name, v.get("release", {}).get("verdict", "?"),
                             v.get("controlled", {}).get("verdict", "?"),
                             ",".join(fails) or "-"))
            except Exception as e:
                rows.append((name, "ERROR", "ERROR", str(e)[:40]))
        print("\n" + "=" * 70 + "\nGATE SUMMARY (same measurement, both policies)\n" + "=" * 70)
        print(f"{'file':44s} {'release':8s} {'controlled':10s} failing checks")
        for name, rel, con, fails in rows:
            print(f"{name:44s} {rel:8s} {con:10s} {fails}")
        n_rel = sum(1 for r in rows if r[1] == "PASS")
        n_con = sum(1 for r in rows if r[2] == "PASS")
        print(f"\n{n_rel}/{len(rows)} pass 'release'; {n_con}/{len(rows)} pass 'controlled'. "
              f"Reports written next to each file.")
        return 0

    if not args.file:
        parser.error("--file <csv> or --all is required")
    return _gate(args.file, policy, policy_name, args, config)


def _gate(path, policy, policy_name, args, config) -> int:
    _require_csv(path)
    candidate = pd.read_csv(path, low_memory=False)
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
    # Values the thresholded checks are computed from, kept so the report
    # can state the verdict under every policy from one measurement.
    measured = {}

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  {'✅' if passed else '❌'} {name}: {detail}")

    print(f"Release gate for {path} ({candidate.shape[0]} x {candidate.shape[1]}):")

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
        measured["coherence_rate"] = rate
        measured["coherence_baseline"] = base
        measured["coherence_rules_violated"] = cand_summary.get("rules_violated")
        measured["coherence_baseline_rules_violated"] = hold_summary.get("rules_violated")
        # Reported, not thresholded: the share of RECORDS carrying at
        # least one violation. The threshold above is per rule-check,
        # which understates how many released patients are affected --
        # a reader deciding whether to release needs both, against the
        # real holdout's own share.
        cand_rows = R.row_violation_mask(candidate, ruleset)
        hold_rows = R.row_violation_mask(holdout_decoded, ruleset)
        measured["coherence_row_share"] = round(float(cand_rows.mean()), 5)
        measured["coherence_row_share_baseline"] = round(float(hold_rows.mean()), 5)
        print(f"     rows carrying at least one violation: "
              f"{measured['coherence_row_share']:.1%} "
              f"(real holdout: {measured['coherence_row_share_baseline']:.1%}) -- "
              f"reported, not thresholded")
        threshold = _coherence_threshold(policy, base)
        check("coherence", rate <= threshold,
              f"violation rate {rate} vs holdout baseline {base} "
              f"(threshold {round(threshold, 5)} = "
              f"max({policy['coherence_multiplier']:g}x baseline, "
              f"baseline+{policy['coherence_absolute']}))")
    else:
        check("coherence", False, "no committed rule set -- run the coherence step first")

    priv_path = os.path.join(config.step_dir("privacy"), "DT4H_Privacy_Assessment.json")
    if os.path.exists(priv_path):
        with open(priv_path) as f:
            _priv = json.load(f)
        committed_p5 = _priv["holdout_baseline"]["dcr_p5"]
        # The privacy step already measured EVERY row of each standard run
        # against the same encoder and threshold. For those files, use the
        # full-sample share instead of a 500-record spot check: at the 10%
        # limit the spot check's sampling error (~1.3pp) can flip a
        # borderline verdict in either direction -- it did, once.
        _run_id = os.path.basename(path).replace("DT4H_Synthetic_", "").replace(".csv", "")
        full_share = next((r.get("share_closer_than_holdout_p5") for r in _priv.get("runs", [])
                           if r.get("run_id") == _run_id), None)
        enc_path = os.path.join(config.step_dir("preprocess"), "DT4H_Numeric_Missing_Encoding.json")
        encoding = json.load(open(enc_path)) if os.path.exists(enc_path) else {}
        # SUBSET-AWARE: the check runs on the intersection of candidate
        # and train columns. NA-padding a width-limited candidate and
        # measuring it against the FULL-width holdout p5 guaranteed
        # large distances -- a file missing half its columns passed
        # vacuously. Candidate distances and the p5 baseline they are
        # compared against must live in the same column subspace.
        subset = [c for c in train.columns if c in candidate.columns]
        full_width = len(subset) == len(train.columns)
        encode, _, _ = build_encoder(train[subset], encoding)
        t_num, t_cat = encode(train[subset])
        if full_width:
            # committed value: the privacy step computed it with this
            # exact full-width encoder, so no recompute is needed
            p5 = committed_p5
            p5_source = "committed full-width holdout p5 (privacy step)"
        else:
            h_num, h_cat = encode(holdout[subset])
            dh, _ = nearest_two_distances(h_num, h_cat, t_num, t_cat)
            p5 = round(float(np.percentile(dh, 5)), 6)
            p5_source = (f"holdout-vs-train p5 recomputed on the {len(subset)}-column "
                         f"subset (candidate missing "
                         f"{len(train.columns) - len(subset)} of "
                         f"{len(train.columns)} columns)")
            print(f"  ⚠️  width-limited candidate: distance check restricted to the "
                  f"{len(subset)} shared column(s); subset p5 = {p5} "
                  f"(committed full-width p5 {committed_p5} does not apply)")
        if full_width and full_share is not None:
            share = float(full_share)
            n_measured = len(candidate)
            too_close = int(round(share * n_measured))
            share_source = "privacy step (all rows, same encoder)"
        else:
            rng = np.random.default_rng(0)
            sample = candidate.iloc[rng.choice(len(candidate), min(DCR_SAMPLE, len(candidate)),
                                               replace=False)]
            s_num, s_cat = encode(sample[subset])
            d1, _ = nearest_two_distances(s_num, s_cat, t_num, t_cat)
            too_close = int((d1 < p5).sum())
            n_measured = len(sample)
            share = too_close / n_measured
            share_source = f"spot check ({n_measured} sampled rows)"
        measured["distance_share"] = share
        measured["distance_sampled"] = int(n_measured)
        measured["distance_closer"] = too_close
        measured["distance_share_source"] = share_source
        measured["holdout_p5"] = p5
        measured["holdout_p5_committed"] = committed_p5
        measured["holdout_p5_source"] = p5_source
        measured["distance_columns_used"] = len(subset)
        measured["distance_full_width"] = full_width
        limit = _distance_limit(policy)
        measured["distance_degenerate"] = bool(p5 <= 0)
        if p5 <= 0:
            # Degenerate threshold: on a narrow column subset, unseen real
            # patients themselves collide with training records at distance
            # zero, so "closer than p5" is unsatisfiable and the check would
            # pass vacuously. That is not evidence of safety -- refuse it.
            check("distance", False,
                  f"holdout p5 threshold is 0 over the {len(subset)} shared "
                  f"column(s): real unseen patients already collide with "
                  f"training records at this width, so the distance check "
                  f"cannot discriminate and a pass here would be vacuous")
        else:
            check("distance", share <= limit,
                  f"{too_close}/{n_measured} record(s) ({share:.1%}, {share_source}) closer than "
                  f"the holdout p5 threshold ({p5}, over {len(subset)} column(s)"
                  + ("" if full_width else ", SUBSET of the full schema")
                  + f"); policy limit {limit:.0%} = "
                  f"{policy['distance_multiplier']:g}x the natural "
                  f"{DISTANCE_NATURAL_SHARE:.0%} share")
    else:
        check("distance", False, "no committed privacy assessment -- run the privacy step first")

    passed = all(c["passed"] for c in checks)
    default_policy = policy_name == DEFAULT_POLICY
    if passed and default_policy:
        verdict = "PASS -- cleared for release"
    elif passed:
        verdict = (f"PASS under the '{policy_name}' policy -- NOT cleared for open "
                   f"release; cleared only for: {policy['intent']}")
    else:
        verdict = "FAIL -- DO NOT RELEASE"
    print(f"\n{'✅' if passed else '🚫'} {verdict}")

    by_policy = _verdicts_by_policy(checks, measured)
    print("\nSame measurement under every policy:")
    for pname, v in by_policy.items():
        marker = " <- selected" if pname == args.policy and not policy_name.startswith("custom") else ""
        detail = ", ".join(f"{k} {'ok' if d['passed'] else 'over'} "
                           f"(limit {d['threshold']:g})"
                           for k, d in v["checks"].items())
        print(f"  {pname:<12} {v['verdict']:<5} {detail}{marker}")
    if passed and not default_policy:
        print(f"⚠️  This file does NOT pass the default '{DEFAULT_POLICY}' policy. "
              f"Relaxed-policy clearance is a governance decision, not a technical one: "
              f"it needs the note recorded above and a consortium ruling behind it.")

    timestamp = datetime.now(timezone.utc).isoformat()
    name = os.path.basename(path).replace(".csv", "")
    report = os.path.join(os.path.dirname(path), f"DT4H_Release_Gate_{name}.md")
    with open(report, "w") as f:
        f.write(f"# Release Gate: {name}\n\n"
                f"Evaluated {timestamp}\n\n"
                f"**{verdict}**\n\n"
                f"Policy: `{policy_name}` -- coherence "
                f"{policy['coherence_multiplier']:g}x the holdout baseline "
                f"(or +{policy['coherence_absolute']:.0%} absolute), distance "
                f"{policy['distance_multiplier']:g}x the natural "
                f"{DISTANCE_NATURAL_SHARE:.0%} share. "
                f"Intent: {policy['intent']}.\n\n")
        if args.note:
            f.write(f"Note: {args.note}\n\n")
        if passed and not default_policy:
            f.write(f"> This file does not pass the default `{DEFAULT_POLICY}` policy. "
                    f"Clearance under a relaxed policy is a governance decision and "
                    f"carries only the scope stated above.\n\n")
        f.write("| check | result | detail |\n|---|---|---|\n")
        for c in checks:
            f.write(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |\n")

        if "distance_columns_used" in measured:
            f.write(f"\nDistance check computed over "
                    f"**{measured['distance_columns_used']}** column(s); "
                    f"p5 threshold {measured['holdout_p5']} from: "
                    f"{measured['holdout_p5_source']}.")
            if not measured.get("distance_full_width", True):
                f.write(" **The candidate is narrower than the full schema: the "
                        "committed full-width p5 does not apply and the baseline was "
                        "recomputed on the shared column subset.**")
            f.write("\n")

        if "coherence_row_share" in measured:
            f.write(f"\nReported, not thresholded: "
                    f"**{measured['coherence_row_share']:.1%}** of this file's records "
                    f"carry at least one rule violation, against "
                    f"**{measured['coherence_row_share_baseline']:.1%}** of the real "
                    f"holdout's. The coherence check above is per applicable rule-check, "
                    f"which is the smaller number; a release decision should be taken on "
                    f"both.\n")

        f.write("\n## Verdict under each policy\n\n"
                "The same measurement, re-thresholded. Absolute checks (schema, "
                "representation, freshness, leakage) are policy-independent.\n\n"
                "| policy | verdict | coherence limit | distance limit | intent |\n"
                "|---|---|---|---|---|\n")
        for pname, v in by_policy.items():
            coh = v["checks"].get("coherence", {})
            dist = v["checks"].get("distance", {})
            f.write(f"| {pname} | {v['verdict']} "
                    f"| {coh.get('threshold', 'n/a')} "
                    f"{'✅' if coh.get('passed') else '❌' if coh else ''} "
                    f"| {dist.get('threshold', 'n/a')} "
                    f"{'✅' if dist.get('passed') else '❌' if dist else ''} "
                    f"| {v['intent']} |\n")
    print(f"Report -> {report}")

    sidecar = report.replace(".md", ".json")
    with open(sidecar, "w") as f:
        json.dump({
            "file": os.path.basename(path),
            "evaluated": timestamp,
            "policy": policy_name,
            "policy_thresholds": policy,
            "note": args.note,
            "verdict": "PASS" if passed else "FAIL",
            "cleared_for_open_release": bool(passed and default_policy),
            "checks": checks,
            "measured": measured,
            "verdict_by_policy": by_policy,
        }, f, indent=2, default=str)
    print(f"Report -> {sidecar}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

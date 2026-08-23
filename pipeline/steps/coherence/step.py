"""
Step: coherence

Row-level clinical coherence audit. Marginal and pairwise metrics say
the COLUMNS look right; this step asks whether each synthetic ROW is a
coherent patient: rules are mined/learned from the TRAIN split (boolean
implications, category-range consistency, survival logic -- see
rules.py), validated on the HOLDOUT split (real patients are the fair
baseline for how often clinical data violates its own logic), and then
every synthetic dataset is audited against them.

The mined rule set is committed (DT4H_Coherence_Rules.json) so the
audit is reproducible and criticizable. Aggregate statistics only.
"""

import glob
import json
import os

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.coherence import rules as R


class CoherenceStep(PipelineStep):
    name = "coherence"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd

        from pipeline.steps.generate.step import GenerateStep

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError("No synthetic files -- run the generate step first.")

        train = self._load_decoded(config.train_output_path, config)
        holdout = self._load_decoded(config.holdout_output_path, config)

        print("Mining coherence rules from the train split...")
        ruleset = R.build_rules(train)
        by_type = {}
        for r in ruleset:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        print(f"  {len(ruleset)} rules: {by_type}")

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)
        rules_path = os.path.join(out_dir, "DT4H_Coherence_Rules.json")
        with open(rules_path, "w") as f:
            json.dump({"parameters": {
                "min_support": R.MIN_SUPPORT,
                "max_train_violation": R.MAX_TRAIN_VIOLATION,
                "range_margin_fraction": R.RANGE_MARGIN_FRACTION,
            }, "rules": ruleset}, f, indent=2)
        print(f"  Saved rule set -> {rules_path}")

        results = {"n_rules": len(ruleset), "rules_by_type": by_type, "frames": []}

        for label, frame in (("train (real)", train), ("holdout (real, unseen)", holdout)):
            res = R.evaluate_rules(frame, ruleset)
            summary = R.summarize_rule_results(res)
            summary["frame"] = label
            results["frames"].append(summary)
            print(f"  {label}: overall violation rate "
                  f"{summary['overall_violation_rate']} over {summary['rule_checks_applicable']} checks")

        holdout_rate = results["frames"][1]["overall_violation_rate"] or 0.0

        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            res = R.evaluate_rules(synth, ruleset)
            summary = R.summarize_rule_results(res)
            summary["frame"] = f"synthetic[{run_id}]"
            summary["run_id"] = run_id
            worst = sorted((r for r in res if r["violation_rate"]),
                           key=lambda r: -r["violation_rate"])[:5]
            summary["worst_rules"] = [
                {k: r.get(k) for k in ("type", "if_true", "then_true", "categorical",
                                        "numeric", "days", "flag", "violation_rate", "applicable")}
                for r in worst]
            results["frames"].append(summary)
            flag = "✅" if (summary["overall_violation_rate"] or 0) <= max(holdout_rate * 3, 0.01) else "⚠️ "
            print(f"  {flag} {run_id}: violation rate {summary['overall_violation_rate']} "
                  f"({summary['rules_violated']}/{summary['rules']} rules violated)")

        json_path = os.path.join(out_dir, "DT4H_Coherence_Audit.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        md_path = os.path.join(out_dir, "DT4H_Coherence_Audit.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"Saved coherence audit -> {json_path} / {md_path}")

    def _load_decoded(self, path: str, config: PipelineConfig):
        import polars as pl

        from pipeline.steps.generate.step import GenerateStep

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run the preprocess step first.")
        df = pl.read_parquet(path).to_pandas()
        df, _ = GenerateStep._decode_numeric_missing(df, config)
        return df

    @staticmethod
    def _render_markdown(r: dict) -> str:
        lines = [
            "# Row-Coherence Audit",
            "",
            f"{r['n_rules']} rules ({r['rules_by_type']}) mined/learned from the TRAIN split "
            "and validated on real data. The holdout row is the fair baseline: real, unseen "
            "patients violating the same rules. A synthetic dataset far above it produces "
            "rows that are individually implausible patients even when every column's "
            "distribution is correct.",
            "",
            "| frame | applicable checks | violations | violation rate | rules violated |",
            "|---|---|---|---|---|",
        ]
        for f in r["frames"]:
            lines.append(f"| {f['frame']} | {f['rule_checks_applicable']} | {f['violations']} "
                         f"| {f['overall_violation_rate']} | {f['rules_violated']}/{f['rules']} |")
        lines += ["", "## Worst rules per synthetic dataset", ""]
        for f in r["frames"]:
            if not f.get("worst_rules"):
                continue
            lines.append(f"**{f['frame']}**")
            for w in f["worst_rules"]:
                desc = (f"{w.get('if_true')} => {w.get('then_true')}" if w["type"] == "implication"
                        else f"{w.get('categorical')} vs {w.get('numeric')}" if w["type"] == "category_range"
                        else f"{w.get('days')} / {w.get('flag') or 'bounds'}")
                lines.append(f"- `{desc}` ({w['type']}): rate {w['violation_rate']} "
                             f"over {w['applicable']} rows")
            lines.append("")
        return "\n".join(lines) + "\n"

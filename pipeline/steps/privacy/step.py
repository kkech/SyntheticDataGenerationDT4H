"""
Step: privacy

Record-level privacy assessment of every synthetic dataset, beyond the
exact-duplicate check the generate step already performs:

  * DCR (distance to closest record): for each synthetic record, the
    Gower-style distance to its nearest real record. The reference point
    is the real data's own leave-one-out DCR distribution -- how close
    real patients are to each other. A synthetic dataset whose records
    sit systematically closer to real records than real records sit to
    each other is echoing individuals rather than the population.
  * NNDR (nearest-neighbor distance ratio): d1/d2 per synthetic record.
    Values near 1 mean the record is "between" real records (population
    structure); values near 0 mean it is locked onto one specific real
    record (memorization).

Headline number per synthesizer: the share of synthetic records closer
to a real record than the 5th percentile of the real-to-real baseline.
Under no memorization this hovers around 5%; far above that is a red
flag for release.

HONEST LIMITATION, stated here and in the report: a proper membership-
inference attack evaluation requires a holdout set excluded from
training, and this pipeline currently trains on all rows. DCR/NNDR
bound record-copying, not membership inference. For the DP synthesizers
the epsilon guarantee covers membership inference by construction; for
the non-DP ones this remains an open item for the paper's limitations
section.

Aggregate statistics only are written -- safe to commit.
"""

import glob
import json
import os
import time

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.privacy.distance import (
    build_encoder,
    nearest_two_distances,
    summarize_dcr,
)


class PrivacyStep(PipelineStep):
    name = "privacy"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd
        import polars as pl

        from pipeline.steps.preprocess.transforms import NUMERIC_ENCODING_FILENAME

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError(
                f"No DT4H_Synthetic_*.csv in {config.step_dir('generate')} -- run the generate step first."
            )

        real = pl.read_parquet(config.preprocessed_output_path).to_pandas()
        encoding_path = os.path.join(config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)
        encoding = {}
        if os.path.exists(encoding_path):
            with open(encoding_path) as f:
                encoding = json.load(f)

        print(f"Real reference: {real.shape[0]} x {real.shape[1]} (sentinel space)")
        encode, numeric_cols, cat_cols = build_encoder(real, encoding)
        print(f"Distance space: {len(numeric_cols)} numeric + {len(cat_cols)} categorical columns "
              f"(constants excluded -- they cannot separate records)")

        real_num, real_cat = encode(real)

        print("Computing real-to-real baseline (leave-one-out nearest neighbors)...")
        t0 = time.time()
        base_d1, base_d2 = nearest_two_distances(real_num, real_cat, real_num, real_cat,
                                                 exclude_self=True)
        baseline = summarize_dcr(base_d1, base_d2)
        print(f"  baseline DCR: p5={baseline['dcr_p5']}, median={baseline['dcr_median']} "
              f"({time.time() - t0:.0f}s)")

        results = {
            "distance_space": {"numeric_columns": len(numeric_cols),
                               "categorical_columns": len(cat_cols)},
            "real_baseline": baseline,
            "synthesizers": [],
        }

        import numpy as np

        for path in synthetic_files:
            synth_name = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synthetic = pd.read_csv(path, low_memory=False)
            print(f"\nAssessing '{synth_name}' ({synthetic.shape[0]} rows)...")
            t0 = time.time()

            missing_cols = [c for c in real.columns if c not in synthetic.columns]
            if missing_cols:
                print(f"  ⚠️  {len(missing_cols)} real column(s) absent from this synthetic file; "
                      f"distances computed over the common columns only.")
                for c in missing_cols:
                    synthetic[c] = pd.NA

            synth_num, synth_cat = encode(synthetic)
            d1, d2 = nearest_two_distances(synth_num, synth_cat, real_num, real_cat)
            stats = summarize_dcr(d1, d2)

            share_too_close = float((d1 < baseline["dcr_p5"]).mean())
            stats.update({
                "synthesizer": synth_name,
                "share_closer_than_real_p5": round(share_too_close, 4),
                "duration_seconds": round(time.time() - t0, 1),
            })
            results["synthesizers"].append(stats)

            verdict = "✅" if share_too_close <= 0.10 and stats["exact_matches"] == 0 else "🚨"
            print(f"  {verdict} DCR p5={stats['dcr_p5']} median={stats['dcr_median']} | "
                  f"exact matches={stats['exact_matches']} | NNDR median={stats['nndr_median']} | "
                  f"closer-than-real-p5: {share_too_close:.1%} (no-memorization expectation ~5%) "
                  f"({stats['duration_seconds']}s)")

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, "DT4H_Privacy_Assessment.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved privacy assessment (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Privacy_Assessment.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"Saved privacy assessment (Markdown) -> {md_path}")

    @staticmethod
    def _render_markdown(r: dict) -> str:
        b = r["real_baseline"]
        lines = [
            "# Privacy Assessment: distance to closest record",
            "",
            f"Distances are Gower-style mixed-type distances in [0,1] over "
            f"{r['distance_space']['numeric_columns']} numeric and "
            f"{r['distance_space']['categorical_columns']} categorical columns, computed in "
            "sentinel space (a synthetic record is only close to a real one if it matches its "
            "values AND its missingness pattern).",
            "",
            f"**Real-to-real baseline** (leave-one-out): DCR p5 = `{b['dcr_p5']}`, "
            f"median = `{b['dcr_median']}`, NNDR median = `{b['nndr_median']}`.",
            "",
            "| synthesizer | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than real p5 |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in r["synthesizers"]:
            flag = "" if s["share_closer_than_real_p5"] <= 0.10 and s["exact_matches"] == 0 else " 🚨"
            lines.append(
                f"| {s['synthesizer']}{flag} | {s['dcr_min']} | {s['dcr_p5']} | {s['dcr_median']} "
                f"| {s['exact_matches']} | {s['nndr_median']} "
                f"| {s['share_closer_than_real_p5']:.1%} |")

        lines += [
            "",
            "Reading the table: `closer than real p5` is the share of synthetic records nearer "
            "to some real record than the closest 5% of real-to-real neighbor distances -- "
            "~5% is the no-memorization expectation; well above that suggests the model echoes "
            "individuals. `exact matches` must be 0 for any release. NNDR near 1 means records "
            "sit between real records (population structure), near 0 means they lock onto one "
            "real record.",
            "",
            "## Limitations",
            "- DCR/NNDR bound record-copying, not membership inference. A proper membership-",
            "  inference evaluation requires a holdout excluded from training; this pipeline",
            "  currently trains on all rows. For DP synthesizers the epsilon guarantee covers",
            "  membership inference by construction; for non-DP synthesizers this is an open",
            "  item for the limitations section.",
        ]
        return "\n".join(lines) + "\n"

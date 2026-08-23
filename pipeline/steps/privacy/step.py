"""
Step: privacy

Record-level privacy assessment of every synthetic dataset, beyond the
exact-duplicate check the generate step already performs:

  * DCR (distance to closest record): for each synthetic record, the
    Gower-style distance to its nearest TRAINING record. The reference
    is the HOLDOUT baseline: how close real patients the model never saw
    sit to the training records. That is exactly what "synthetic record
    at a plausible distance" should look like -- a synthetic dataset
    whose records sit systematically closer to the training data than
    unseen real patients do is echoing the individuals it was trained
    on, not the population.
  * NNDR (nearest-neighbor distance ratio): d1/d2 per synthetic record.
    Values near 1 mean the record is "between" real records (population
    structure); values near 0 mean it is locked onto one specific real
    record (memorization).

Headline number per run: the share of synthetic records closer to a
training record than the 5th percentile of the holdout-to-train
baseline. Under no memorization this hovers around 5%; far above that
is a red flag for release.

Because the generators trained only on the training split, this is a
genuine unseen-data comparison. A full adversarial membership-inference
attack (shadow models, per-record scores) remains future work and is
noted in the report; for DP synthesizers the epsilon guarantee bounds
membership inference by construction.

Aggregate statistics only are written -- safe to commit.
"""

import glob
import json
import os
import time

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case
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

        for path in (config.train_output_path, config.holdout_output_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found -- run the preprocess step first.")

        train = pl.read_parquet(config.train_output_path).to_pandas()
        holdout = pl.read_parquet(config.holdout_output_path).to_pandas()
        encoding_path = os.path.join(config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)
        encoding = {}
        if os.path.exists(encoding_path):
            with open(encoding_path) as f:
                encoding = json.load(f)

        print(f"Training reference: {train.shape[0]} x {train.shape[1]} (sentinel space) | "
              f"holdout: {holdout.shape[0]} rows (real patients the models never saw)")
        encode, numeric_cols, cat_cols = build_encoder(train, encoding)
        print(f"Distance space: {len(numeric_cols)} numeric + {len(cat_cols)} categorical columns "
              f"(constants excluded -- they cannot separate records)")

        train_num, train_cat = encode(train)
        hold_num, hold_cat = encode(holdout)

        import numpy as np

        def _dcr_hist(d1):
            counts, edges = np.histogram(d1, bins=50, range=(0.0, 1.0))
            return {"bin_edges": [round(float(e), 3) for e in edges],
                    "counts": [int(c) for c in counts]}

        print("Computing holdout-to-train baseline (unseen real patients vs training records)...")
        t0 = time.time()
        base_d1, base_d2 = nearest_two_distances(hold_num, hold_cat, train_num, train_cat)
        baseline = summarize_dcr(base_d1, base_d2)
        baseline["dcr_histogram"] = _dcr_hist(base_d1)
        print(f"  holdout baseline DCR: p5={baseline['dcr_p5']}, median={baseline['dcr_median']} "
              f"({time.time() - t0:.0f}s)")

        run_meta = self._load_run_metadata(config)
        results = {
            "distance_space": {"numeric_columns": len(numeric_cols),
                               "categorical_columns": len(cat_cols)},
            "holdout_baseline": baseline,
            "runs": [],
        }

        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synthetic = pd.read_csv(path, low_memory=False)
            synthetic, _ = align_categorical_case(synthetic, train)
            print(f"\nAssessing '{run_id}' ({synthetic.shape[0]} rows)...")
            t0 = time.time()

            missing_cols = [c for c in train.columns if c not in synthetic.columns]
            if missing_cols:
                print(f"  ⚠️  {len(missing_cols)} training column(s) absent from this synthetic "
                      f"file (width-limited run); distances computed over the common columns only.")
                # One concat, not a column-at-a-time loop: inserting ~160
                # columns individually fragments the frame.
                pad = pd.DataFrame(pd.NA, index=synthetic.index, columns=missing_cols)
                synthetic = pd.concat([synthetic, pad], axis=1)

            synth_num, synth_cat = encode(synthetic)
            d1, d2 = nearest_two_distances(synth_num, synth_cat, train_num, train_cat)
            stats = summarize_dcr(d1, d2)
            stats["dcr_histogram"] = _dcr_hist(d1)

            share_too_close = float((d1 < baseline["dcr_p5"]).mean())
            stats.update({
                "run_id": run_id,
                **run_meta.get(run_id, {}),
                "share_closer_than_holdout_p5": round(share_too_close, 4),
                "duration_seconds": round(time.time() - t0, 1),
                "columns_padded_as_missing": len(missing_cols),
            })
            results["runs"].append(stats)

            verdict = "✅" if share_too_close <= 0.10 and stats["exact_matches"] == 0 else "🚨"
            print(f"  {verdict} DCR p5={stats['dcr_p5']} median={stats['dcr_median']} | "
                  f"exact matches={stats['exact_matches']} | NNDR median={stats['nndr_median']} | "
                  f"closer-than-holdout-p5: {share_too_close:.1%} (no-memorization expectation ~5%) "
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

    def _load_run_metadata(self, config: PipelineConfig) -> dict:
        path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            summary = json.load(f)
        return {
            r.get("run_id", r.get("synthesizer")): {
                "synthesizer": r.get("synthesizer"),
                "epsilon": r.get("epsilon"),
                "seed": r.get("seed"),
            }
            for r in summary.get("runs", [])
        }

    @staticmethod
    def _render_markdown(r: dict) -> str:
        b = r["holdout_baseline"]
        lines = [
            "# Privacy Assessment: distance to closest training record",
            "",
            f"Distances are Gower-style mixed-type distances in [0,1] over "
            f"{r['distance_space']['numeric_columns']} numeric and "
            f"{r['distance_space']['categorical_columns']} categorical columns, computed in "
            "sentinel space (a synthetic record is only close to a real one if it matches its "
            "values AND its missingness pattern). The baseline is the HOLDOUT distribution: "
            "real patients the generators never saw, measured against the training records -- "
            "exactly what an innocent 'new' record's distance profile looks like.",
            "",
            f"**Holdout-to-train baseline**: DCR p5 = `{b['dcr_p5']}`, "
            f"median = `{b['dcr_median']}`, NNDR median = `{b['nndr_median']}`.",
            "",
            "| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in r["runs"]:
            flag = "" if s["share_closer_than_holdout_p5"] <= 0.10 and s["exact_matches"] == 0 else " 🚨"
            lines.append(
                f"| {s['run_id']}{flag} | {s['dcr_min']} | {s['dcr_p5']} | {s['dcr_median']} "
                f"| {s['exact_matches']} | {s['nndr_median']} "
                f"| {s['share_closer_than_holdout_p5']:.1%} |")

        lines += [
            "",
            "Reading the table: `closer than holdout p5` is the share of synthetic records "
            "nearer to some training record than the closest 5% of unseen-real-patient "
            "distances -- ~5% is the no-memorization expectation; well above that suggests "
            "the model echoes the individuals it trained on. `exact matches` must be 0 for "
            "any release. NNDR near 1 means records sit between real records (population "
            "structure), near 0 means they lock onto one real record.",
            "",
            "## Limitations",
            "- DCR/NNDR against the holdout baseline bound record-copying with a genuine",
            "  unseen-data reference. A full adversarial membership-inference evaluation",
            "  (shadow models, per-record attack scores) remains future work; for DP",
            "  synthesizers the epsilon guarantee bounds membership inference by",
            "  construction.",
            "- Width-limited (AIM) runs generate a column subset; their absent columns are",
            "  padded as missing on the synthetic side before encoding. Their DCR values",
            "  are therefore NOT directly comparable to full-width runs -- compare",
            "  width-limited runs only against each other and against the shared baseline.",
        ]
        return "\n".join(lines) + "\n"

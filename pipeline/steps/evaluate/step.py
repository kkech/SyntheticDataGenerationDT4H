"""
Step: evaluate

Measures distribution distance between the stages of the pipeline:

  * original      -- the raw loaded dataset (output/load_data/);
  * preprocessed  -- the full training frame, sentinels decoded;
  * train/holdout -- the 75/25 split of the preprocessed frame; the
                     generators only ever saw the train side;
  * synthetic     -- every DT4H_Synthetic_*.csv the generate step wrote.

Comparisons, each answering a different question:

  original vs preprocessed  did preprocessing distort the data it kept?
                            (should be ~zero on every untouched column)
  train vs holdout          the SAMPLING-NOISE FLOOR: two disjoint
                            samples of real patients differ by this much
                            purely by chance. No synthesizer can honestly
                            beat this; how close each gets to it is the
                            calibrated fidelity result.
  train vs synthetic        fidelity to what the generator was trained on
                            (the headline number, read against the floor)
  holdout vs synthetic      generalization: distance to real data the
                            generator never saw.

The 38 constant columns the generate step re-attaches verbatim are
EXCLUDED from all aggregates: they are copies, not modelling successes,
and would flatter every model equally.

Runs sharing a (model, epsilon) are aggregated across seeds as
mean +/- sd, and DP models get an epsilon-sweep view.

Writes JSON detail plus one Markdown overview to output/evaluate/.
Aggregate statistics only -- safe to commit.
"""

import glob
import json
import os
import statistics

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case
from pipeline.steps.base import PipelineStep
from pipeline.steps.evaluate.associations import association_profile, compare_association_profiles
from pipeline.steps.evaluate.metrics import compare_frames


class EvaluateStep(PipelineStep):
    name = "evaluate"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd
        import polars as pl

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError(
                f"No DT4H_Synthetic_*.csv in {config.step_dir('generate')} -- run the generate step first."
            )

        train = self._load_decoded(config.train_output_path, config)
        holdout = self._load_decoded(config.holdout_output_path, config)
        print(f"Train (decoded): {train.shape[0]} x {train.shape[1]} | "
              f"Holdout (decoded): {holdout.shape[0]} x {holdout.shape[1]}")

        # Constants match the generate step's definition (computed on the
        # train frame): re-attached verbatim, so trivially perfect in
        # every metric -- excluded from all aggregates.
        constants = {c for c in train.columns if train[c].nunique(dropna=False) <= 1}
        print(f"Excluding {len(constants)} constant column(s) from all aggregates "
              f"(re-attached verbatim by generate -- copies, not modelling successes).")

        run_meta = self._load_run_metadata(config)

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        results = {"constant_columns_excluded": sorted(constants), "comparisons": []}

        # Lossless-preprocessing proof (full frames, nothing excluded).
        if os.path.exists(config.local_full_dataset_path):
            original = self._prepare_original(config)
            preprocessed = self._load_decoded(config.preprocessed_output_path, config)
            print(f"\nComparing original vs preprocessed (lossless-preprocessing proof)...")
            results["comparisons"].append(
                compare_frames(original, preprocessed, "original", "preprocessed")
            )
        else:
            print(f"⚠️  Original dataset not found at {config.local_full_dataset_path} -- "
                  f"lossless-preprocessing proof skipped this run.")

        print("\nComputing the sampling-noise floor (train vs holdout)...")
        noise_floor = compare_frames(train, holdout, "train", "holdout",
                                     exclude_columns=constants)
        results["comparisons"].append(noise_floor)
        nf = noise_floor["aggregates"]
        print(f"  Noise floor: KS mean={nf['ks'].get('mean')}, TVD mean={nf['tvd'].get('mean')}, "
              f"missing-rate MAD={nf['missing_rate_mean_abs_diff']} -- no synthesizer can "
              f"honestly do better than this.")

        print("\nProfiling association structure of the training data...")
        real_assoc = association_profile(train)
        print(f"  {sum(len(v) for v in real_assoc.values())} measurable column pairs "
              f"(numeric-numeric {len(real_assoc['num_num'])}, "
              f"categorical-categorical {len(real_assoc['cat_cat'])}, "
              f"numeric-categorical {len(real_assoc['num_cat'])})")

        # Association noise floor: how much associations differ between
        # two real samples -- calibrates every synthetic |delta| below.
        print("Computing the association noise floor (train vs holdout profiles)...")
        assoc_floor = compare_association_profiles(real_assoc, association_profile(holdout))
        results["association_noise_floor"] = assoc_floor
        for kind in ("num_num", "cat_cat", "num_cat"):
            a = assoc_floor.get(kind, {})
            if a.get("pairs"):
                print(f"  {kind}: mean |Δ| floor = {a['mean_abs_delta']} over {a['pairs']} pairs")

        # C2ST floor: a classifier separating two real samples should
        # land at ~0.5; synthetic AUCs are read against this.
        from pipeline.steps.evaluate.c2st import c2st_auc

        modelled_cols = [c for c in train.columns if c not in constants]
        print("Computing the C2ST floor (train vs holdout)...")
        c2st_floor = c2st_auc(train, holdout, modelled_cols, seed=config.seed)
        results["c2st_floor_train_vs_holdout"] = c2st_floor
        print(f"  C2ST floor AUC = {c2st_floor} (0.5 = indistinguishable)")

        strata = self._strata(train)
        subgroup_floor = {}
        for sname, mask in strata.items():
            h_mask = self._strata(holdout).get(sname)
            if mask.sum() < 150 or h_mask is None or h_mask.sum() < 50:
                continue
            agg = compare_frames(train[mask], holdout[h_mask], f"train[{sname}]",
                                 f"holdout[{sname}]", exclude_columns=constants)["aggregates"]
            subgroup_floor[sname] = {"n_train": int(mask.sum()), "n_holdout": int(h_mask.sum()),
                                     "ks_mean": agg["ks"].get("mean"),
                                     "tvd_mean": agg["tvd"].get("mean")}
        results["subgroup_noise_floor"] = subgroup_floor
        print(f"Subgroup strata with sufficient support: {list(subgroup_floor)}")

        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synthetic = pd.read_csv(path, low_memory=False)
            synthetic, _ = align_categorical_case(synthetic, train)
            print(f"\nComparing against '{run_id}' ({synthetic.shape[0]} rows x "
                  f"{synthetic.shape[1]} cols)...")

            # Files are scored EXACTLY as they are on disk -- never
            # silently repaired. This check only detects staleness: a
            # correctly generated file re-decodes to itself, so any cell
            # the current decoder would change means the file was written
            # by an older pipeline version and the generate step needs a
            # full rerun. The mismatch is scored as-is and flagged loudly,
            # so the report can never look better than the actual files.
            from pipeline.steps.generate.step import GenerateStep

            _, would_change = GenerateStep._decode_numeric_missing(synthetic.copy(), config)
            stale_cells = sum(would_change.values())
            if stale_cells:
                print(f"🚨 STALE FILE: {os.path.basename(path)} contains {stale_cells} cell(s) "
                      f"across {len(would_change)} column(s) that the current pipeline would not "
                      f"produce (undecoded sentinel-region values). Scores below describe the "
                      f"stale file. Re-run the generate step to refresh it.")

            entry = {
                "run_id": run_id,
                **run_meta.get(run_id, {}),
                "stale_file": bool(stale_cells),
                "stale_cells": stale_cells,
                "train_vs_synthetic": compare_frames(
                    train, synthetic, "train", f"synthetic[{run_id}]",
                    exclude_columns=constants),
                "holdout_vs_synthetic": compare_frames(
                    holdout, synthetic, "holdout", f"synthetic[{run_id}]",
                    exclude_columns=constants),
                # Full-schema view (constants included) for transparency;
                # aggregates only -- the modelled-columns view stays the
                # headline because constants are copies, not modelling.
                "train_vs_synthetic_full_schema_aggregates": compare_frames(
                    train, synthetic, "train", f"synthetic[{run_id}] (full schema)"
                )["aggregates"],
            }

            entry["c2st_auc"] = c2st_auc(train, synthetic, modelled_cols, seed=config.seed)
            print(f"  C2ST AUC = {entry['c2st_auc']} (floor {c2st_floor})")

            entry["subgroup_fidelity"] = {}
            s_strata = self._strata(synthetic)
            for sname, floor in subgroup_floor.items():
                sm = s_strata.get(sname)
                if sm is None or sm.sum() < 50:
                    continue
                agg = compare_frames(train[strata[sname]], synthetic[sm],
                                     f"train[{sname}]", f"synthetic[{sname}]",
                                     exclude_columns=constants)["aggregates"]
                entry["subgroup_fidelity"][sname] = {
                    "n_synthetic": int(sm.sum()),
                    "ks_mean": agg["ks"].get("mean"),
                    "tvd_mean": agg["tvd"].get("mean"),
                }

            print(f"  Profiling association structure of '{run_id}'...")
            entry["associations"] = compare_association_profiles(
                real_assoc, association_profile(synthetic)
            )
            fab = sum(entry["associations"].get(k, {}).get("fabricated_pairs", 0)
                      for k in ("num_num", "cat_cat", "num_cat"))
            if fab:
                print(f"  ⚠️  {fab} fabricated association(s) (near-zero real, strong synthetic)")
            results["comparisons"].append(entry)

            agg = entry["train_vs_synthetic"]["aggregates"]
            print(f"  train vs {run_id}: "
                  f"KS mean={agg['ks'].get('mean')} (floor {nf['ks'].get('mean')}), "
                  f"TVD mean={agg['tvd'].get('mean')} (floor {nf['tvd'].get('mean')}), "
                  f"missing-rate MAD={agg['missing_rate_mean_abs_diff']}")

        results["groups"] = self._group_runs(results["comparisons"])

        json_path = os.path.join(out_dir, "DT4H_Evaluation.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved evaluation detail (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Evaluation.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"Saved evaluation overview (Markdown) -> {md_path}")

    # --- frame preparation ---

    def _load_decoded(self, path: str, config: PipelineConfig):
        """A sentinel-space parquet with sentinels decoded back to null,
        so comparisons see observed distributions, not sentinel spikes."""
        import polars as pl

        from pipeline.steps.generate.step import GenerateStep

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run the preprocess step first.")
        df = pl.read_parquet(path).to_pandas()
        df, _ = GenerateStep._decode_numeric_missing(df, config)
        return df

    def _prepare_original(self, config: PipelineConfig):
        """The raw frame aligned only enough that comparisons measure real
        drift rather than encoding artifacts: ARRAY columns flattened,
        Decimals cast, NYHA LOINC codes mapped to the same 1-4 ordinals."""
        import polars as pl

        from pipeline.steps.preprocess.transforms import (
            NYHA_COLUMN,
            build_nyha_map,
            flatten_array_columns,
            load_variable_metadata,
            normalize_numeric_dtypes,
        )

        orig_pl = pl.read_parquet(config.local_full_dataset_path)
        if os.path.exists(config.metadata_path):
            orig_pl, _ = flatten_array_columns(orig_pl, load_variable_metadata(config.metadata_path))
        orig_pl, _ = normalize_numeric_dtypes(orig_pl)
        df = orig_pl.to_pandas()
        if NYHA_COLUMN in df.columns and os.path.exists(config.metadata_path):
            nyha_map = build_nyha_map(load_variable_metadata(config.metadata_path))
            df = df.copy()
            df[NYHA_COLUMN] = df[NYHA_COLUMN].map(nyha_map)
        return df

    @staticmethod
    def _strata(df):
        """Clinically meaningful subgroups for per-stratum fidelity.
        Returns {name: boolean mask}; missing stratifier columns simply
        produce no stratum."""
        import pandas as pd

        strata = {}
        g = "patient_demographics_gender"
        if g in df.columns:
            low = df[g].astype("object").where(df[g].notna(), "").astype(str).str.lower()
            strata["female"] = (low == "female").to_numpy()
            strata["male"] = (low == "male").to_numpy()
        a = "patient_demographics_age"
        if a in df.columns:
            age = pd.to_numeric(df[a], errors="coerce")
            strata["age_under_65"] = (age < 65).fillna(False).to_numpy()
            strata["age_65_79"] = ((age >= 65) & (age < 80)).fillna(False).to_numpy()
            strata["age_80_plus"] = (age >= 80).fillna(False).to_numpy()
        return strata

    def _load_run_metadata(self, config: PipelineConfig) -> dict:
        """run_id -> {synthesizer, epsilon, seed} from the generation
        summary, so results can be grouped across seeds and epsilons."""
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

    # --- cross-run aggregation ---

    @staticmethod
    def _group_runs(comparisons: list) -> list:
        """Mean +/- sd of the headline aggregates per (model, epsilon)
        group, across seeds."""
        groups: dict[tuple, list] = {}
        for c in comparisons:
            if "train_vs_synthetic" not in c:
                continue
            key = (c.get("synthesizer") or c["run_id"], c.get("epsilon"))
            groups.setdefault(key, []).append(c["train_vs_synthetic"]["aggregates"])

        def _mean_sd(values):
            values = [v for v in values if v is not None]
            if not values:
                return None
            return {"mean": round(statistics.mean(values), 4),
                    "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
                    "n": len(values)}

        out = []
        for (model, eps), aggs in sorted(groups.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            out.append({
                "synthesizer": model,
                "epsilon": eps,
                "n_runs": len(aggs),
                "ks_mean": _mean_sd([a["ks"].get("mean") for a in aggs]),
                "tvd_mean": _mean_sd([a["tvd"].get("mean") for a in aggs]),
                "missing_rate_mad": _mean_sd([a["missing_rate_mean_abs_diff"] for a in aggs]),
            })
        return out

    # --- reporting ---

    @staticmethod
    def _render_markdown(results: dict) -> str:
        lines = [
            "# Evaluation: fidelity against the sampling-noise floor",
            "",
            "Metrics are computed per column over observed values (nulls excluded); "
            "missingness rates are compared separately. KS and TVD are in [0,1], "
            "lower is closer; `W/std` is the Wasserstein distance in units of the "
            "reference standard deviation. The `train vs holdout` row is the "
            "sampling-noise floor: two disjoint samples of real patients differ by "
            "this much purely by chance, so read every synthetic row against it. "
            f"{len(results.get('constant_columns_excluded', []))} constant columns "
            "(re-attached verbatim, trivially perfect) are excluded from all aggregates.",
            "",
            "| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        def _row(c):
            a = c["aggregates"]
            return (f"| {c['pair']} | {c['columns_compared']} "
                    f"| {a['ks'].get('mean', '-')} | {a['ks'].get('median', '-')} "
                    f"| {a['ks_frac_below_0.1'] if a['ks_frac_below_0.1'] is not None else '-'} "
                    f"| {a['wasserstein_std'].get('mean', '-')} "
                    f"| {a['tvd'].get('mean', '-')} "
                    f"| {a['tvd_frac_below_0.05'] if a['tvd_frac_below_0.05'] is not None else '-'} "
                    f"| {a['missing_rate_mean_abs_diff']} |")

        detail_sections = []
        for c in results["comparisons"]:
            if "pair" in c:  # original-vs-preprocessed or train-vs-holdout
                lines.append(_row(c))
                detail_sections.append(c)
            else:
                if c.get("stale_file"):
                    lines.append(f"| 🚨 synthetic[{c['run_id']}] IS STALE "
                                 f"({c['stale_cells']} undecoded cells) -- rerun generate | | | | | | | | |")
                lines.append(_row(c["train_vs_synthetic"]))
                detail_sections.append(c["train_vs_synthetic"])

        if results.get("groups"):
            lines += [
                "",
                "## Per (model, ε) across seeds (train vs synthetic)",
                "",
                "| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |",
                "|---|---|---|---|---|---|",
            ]

            def _pm(m):
                if not m:
                    return "-"
                return f"{m['mean']}" + (f" ± {m['sd']}" if m.get("sd") is not None else "")

            for g in results["groups"]:
                eps = f"{g['epsilon']:g}" if g.get("epsilon") is not None else "-"
                lines.append(f"| {g['synthesizer']} | {eps} | {g['n_runs']} "
                             f"| {_pm(g['ks_mean'])} | {_pm(g['tvd_mean'])} "
                             f"| {_pm(g['missing_rate_mad'])} |")

        lines += [
            "",
            "## Full-joint distinguishability (C2ST)",
            "",
            "AUC of a classifier separating real from synthetic rows; 0.5 = joints "
            f"indistinguishable. Floor (train vs holdout): **{results.get('c2st_floor_train_vs_holdout', '-')}**.",
            "",
            "| run | C2ST AUC |", "|---|---|",
        ]
        for c in results["comparisons"]:
            if "c2st_auc" in c:
                lines.append(f"| {c['run_id']} | {c['c2st_auc']} |")

        if results.get("subgroup_noise_floor"):
            floor = results["subgroup_noise_floor"]
            names = list(floor)
            lines += ["", "## Subgroup fidelity (KS mean per stratum, train vs synthetic)",
                      "",
                      "Does the synthetic cohort represent every subgroup as faithfully as the "
                      "majority? Each cell is read against its stratum's own noise floor.",
                      "",
                      "| run | " + " | ".join(names) + " |",
                      "|---|" + "|".join(["---"] * len(names)) + "|",
                      "| *noise floor* | " + " | ".join(str(floor[n]["ks_mean"]) for n in names) + " |"]
            for c in results["comparisons"]:
                if "subgroup_fidelity" not in c:
                    continue
                cells = [str(c["subgroup_fidelity"].get(n, {}).get("ks_mean", "-")) for n in names]
                lines.append(f"| {c['run_id']} | " + " | ".join(cells) + " |")

        lines += [
            "",
            "## Generalization (holdout vs synthetic)",
            "",
            "Distance to real records the generator NEVER saw. A model that is much "
            "closer to train than to holdout is fitting its training sample, not the "
            "population.",
            "",
            "| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |",
            "|---|---|---|---|---|",
        ]
        for c in results["comparisons"]:
            if "train_vs_synthetic" not in c:
                continue
            t, h = c["train_vs_synthetic"]["aggregates"], c["holdout_vs_synthetic"]["aggregates"]
            lines.append(f"| {c['run_id']} | {t['ks'].get('mean', '-')} | {h['ks'].get('mean', '-')} "
                         f"| {t['tvd'].get('mean', '-')} | {h['tvd'].get('mean', '-')} |")

        lines += [
            "",
            "## Association structure (train vs synthetic)",
            "",
            "Absolute change in pairwise association; 0 = relationship perfectly preserved. "
            "`fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered "
            "strongly associated (>0.5) in the synthetic data. Noise floor rows show how much "
            "two real samples differ.",
            "",
            "| run | pair type | pairs | mean \\|Δ\\| | median \\|Δ\\| | \\|Δ\\|<0.1 | fabricated | worst pair |",
            "|---|---|---|---|---|---|---|---|",
        ]
        floor_assoc = results.get("association_noise_floor", {})
        for kind, label in (("num_num", "Spearman (num-num)"), ("cat_cat", "Cramer's V (cat-cat)"),
                            ("num_cat", "corr-ratio (num-cat)")):
            a = floor_assoc.get(kind, {})
            if a.get("pairs"):
                lines.append(f"| *noise floor* | {label} | {a['pairs']} | {a['mean_abs_delta']} "
                             f"| {a['median_abs_delta']} | {a['frac_below_0.1']} "
                             f"| {a.get('fabricated_pairs', 0)} | - |")
        for c in results["comparisons"]:
            if "associations" not in c:
                continue
            for kind, label in (("num_num", "Spearman (num-num)"), ("cat_cat", "Cramer's V (cat-cat)"),
                                ("num_cat", "corr-ratio (num-cat)")):
                a = c["associations"].get(kind, {})
                if not a.get("pairs"):
                    lines.append(f"| {c['run_id']} | {label} | 0 | - | - | - | - | - |")
                    continue
                w = a["worst"][0]
                lines.append(
                    f"| {c['run_id']} | {label} | {a['pairs']} | {a['mean_abs_delta']} "
                    f"| {a['median_abs_delta']} | {a['frac_below_0.1']} "
                    f"| {a.get('fabricated_pairs', 0)} "
                    f"| `{w['pair']}` ({w['real']} -> {w['synthetic']}) |")

        for c in detail_sections:
            lines += ["", f"## {c['pair']}", ""]
            if c["worst_numeric"]:
                lines.append("Worst numeric columns (by KS):")
                for r in c["worst_numeric"]:
                    lines.append(f"- `{r['column']}`: KS={r['ks_statistic']}, W/std={r['wasserstein_std']}, "
                                 f"mean {r['mean_a']} -> {r['mean_b']}, "
                                 f"missing {r['missing_rate_a']:.0%} -> {r['missing_rate_b']:.0%}")
            if c["worst_categorical"]:
                lines.append("Worst categorical columns (by TVD):")
                for r in c["worst_categorical"]:
                    lines.append(f"- `{r['column']}`: TVD={r['tvd']}, "
                                 f"{r['n_categories_a']} -> {r['n_categories_b']} categories, "
                                 f"missing {r['missing_rate_a']:.0%} -> {r['missing_rate_b']:.0%}")

        return "\n".join(lines) + "\n"

"""
Step: utility

Train-Synthetic-Test-Real (TSTR): the standard machine-learning utility
check for a released synthetic dataset. For each clinical outcome target,
a gradient-boosting classifier is trained twice --

  * baseline: on the real TRAINING split;
  * TSTR:     on each synthetic dataset;

-- and both are scored on the HOLDOUT split: real patients that neither
the generators nor either classifier ever saw during training. A
synthetic dataset with good utility yields nearly the baseline AUC:
models trained on it transfer to real patients. Distribution metrics
(evaluate step) say the data LOOKS right; this says the data WORKS.

Targets are selected from the metadata's declared outcome variables,
diversified by outcome family (five time-windows of one endpoint are not
five results) with a mortality endpoint force-included -- see
targets.py. Every declared outcome column is excluded from the feature
set: outcomes correlate with each other, and leaking one outcome into
the features of another would inflate every AUC.

Everything runs in the decoded (released) representation: nulls are
nulls, exactly as a user of the published dataset would see them.
Gradient boosting handles missing values natively, so missingness
patterns participate in prediction the same way for real and synthetic.

Aggregate statistics only are written -- safe to commit.
"""

import glob
import json
import os
import statistics

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.utility.targets import (
    MIN_CLASS_TEST,
    MIN_CLASS_TRAIN,
    declared_outcomes,
    select_targets,
    to_binary,
)


class UtilityStep(PipelineStep):
    name = "utility"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError(
                f"No DT4H_Synthetic_*.csv in {config.step_dir('generate')} -- run the generate step first."
            )

        train = self._load_decoded(config.train_output_path, config)
        holdout = self._load_decoded(config.holdout_output_path, config)
        print(f"Real train: {train.shape[0]} rows | holdout (test set, unseen by "
              f"generators AND classifiers): {holdout.shape[0]} rows")

        outcome_cols = declared_outcomes(config.metadata_path)
        targets = select_targets(train, outcome_cols, config.utility_max_targets,
                                 explicit=config.utility_targets)

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        if not targets:
            print("⚠️  No eligible outcome targets (need a BOOLEAN outcome with at least "
                  f"{MIN_CLASS_TRAIN} records of each class). Writing an empty report.")
            self._write({"targets": [], "note": "no eligible targets"}, out_dir)
            return

        feature_cols = [c for c in train.columns if c not in outcome_cols]
        print(f"Targets ({len(targets)}, family-diversified): {targets}")
        print(f"Features: {len(feature_cols)} columns (all declared outcome columns "
              f"excluded to prevent cross-outcome leakage)")

        run_meta = self._load_run_metadata(config)
        results = {"seed": config.seed, "targets": [],
                   "n_train": int(train.shape[0]), "n_holdout": int(holdout.shape[0])}

        for target in targets:
            print(f"\nTarget: {target}")
            entry = self._evaluate_target(train, holdout, synthetic_files, target,
                                          feature_cols, config)
            results["targets"].append(entry)

        # Per-run aggregates across targets, then grouped per (model, eps).
        metrics = {"auc_gap": {}, "auc_gap_logreg": {}, "augmentation_delta": {}}
        for t in results["targets"]:
            for r in t["tstr"]:
                if r.get("auc") is not None and t.get("baseline_auc") is not None:
                    metrics["auc_gap"].setdefault(r["run_id"], []).append(
                        t["baseline_auc"] - r["auc"])
                if r.get("auc_gap_logreg") is not None:
                    metrics["auc_gap_logreg"].setdefault(r["run_id"], []).append(
                        r["auc_gap_logreg"])
                if r.get("augmentation_delta") is not None:
                    metrics["augmentation_delta"].setdefault(r["run_id"], []).append(
                        r["augmentation_delta"])
        results["aggregate_auc_gap"] = {
            run_id: round(sum(v) / len(v), 4)
            for run_id, v in sorted(metrics["auc_gap"].items())
        }

        groups: dict[tuple, dict] = {}
        for metric, by_run in metrics.items():
            for run_id, vals in by_run.items():
                meta = run_meta.get(run_id, {})
                key = (meta.get("synthesizer") or run_id, meta.get("epsilon"))
                groups.setdefault(key, {}).setdefault(metric, []).append(
                    sum(vals) / len(vals))
        results["grouped_auc_gap"] = []
        for (model, eps), m in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            gaps = m.get("auc_gap", [])
            if not gaps:
                continue
            results["grouped_auc_gap"].append({
                "synthesizer": model, "epsilon": eps, "n_runs": len(gaps),
                "mean_gap": round(statistics.mean(gaps), 4),
                "sd_gap": round(statistics.stdev(gaps), 4) if len(gaps) > 1 else None,
                "mean_gap_logreg": round(statistics.mean(m["auc_gap_logreg"]), 4)
                    if m.get("auc_gap_logreg") else None,
                "mean_augmentation_delta": round(statistics.mean(m["augmentation_delta"]), 4)
                    if m.get("augmentation_delta") else None,
            })

        print("\nMean AUC gap vs baseline per (model, ε), lower is better "
              "(augmentation: positive = synthetic data ADDS value):")
        for g in results["grouped_auc_gap"]:
            eps = f" ε={g['epsilon']:g}" if g.get("epsilon") is not None else ""
            sd = f" ± {g['sd_gap']}" if g.get("sd_gap") is not None else ""
            print(f"  {g['synthesizer']}{eps}: HistGB {g['mean_gap']:+.4f}{sd} | "
                  f"LogReg {g.get('mean_gap_logreg')} | augment {g.get('mean_augmentation_delta')} "
                  f"({g['n_runs']} run(s))")

        self._write(results, out_dir)

    # --- data loading ---

    def _load_decoded(self, path: str, config: PipelineConfig):
        import polars as pl

        from pipeline.steps.generate.step import GenerateStep

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run the preprocess step first.")
        df = pl.read_parquet(path).to_pandas()
        df, _ = GenerateStep._decode_numeric_missing(df, config)
        return df

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

    # --- model fitting ---

    def _evaluate_target(self, train, holdout, synthetic_files, target, feature_cols,
                         config) -> dict:
        import pandas as pd

        y_train_all = to_binary(train[target])
        y_test_all = to_binary(holdout[target])

        entry = {
            "target": target,
            "n_train_labelled": int(len(y_train_all)) if y_train_all is not None else 0,
            "n_holdout_labelled": int(len(y_test_all)) if y_test_all is not None else 0,
            "baseline_auc": None,
            "tstr": [],
        }
        if y_train_all is None or y_test_all is None:
            entry["note"] = "target not two-class in train or holdout; skipped"
            print(f"  (skipped: {entry['note']})")
            return entry
        if min(int(y_test_all.sum()), int((1 - y_test_all).sum())) < MIN_CLASS_TEST:
            entry["note"] = "holdout too imbalanced for this target; skipped"
            print(f"  (skipped: {entry['note']})")
            return entry

        categories = self._fit_categories(train, feature_cols)
        x_train = self._encode(train.loc[y_train_all.index], feature_cols, categories)
        x_test = self._encode(holdout.loc[y_test_all.index], feature_cols, categories)

        baseline = self._fit_score(x_train, y_train_all, x_test, y_test_all, config.seed)
        baseline_lr = self._fit_score_logreg(x_train, y_train_all, x_test, y_test_all, config.seed)
        entry["baseline_auc"] = baseline
        entry["baseline_auc_logreg"] = baseline_lr
        entry["positives_train"] = int(y_train_all.sum())
        entry["positives_holdout"] = int(y_test_all.sum())
        print(f"  baseline (train real -> test holdout): AUC={baseline} (HistGB) / {baseline_lr} (LogReg)")

        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            record = {"run_id": run_id, "auc": None}
            y_s = to_binary(synth[target]) if target in synth.columns else None
            if y_s is None or min(int(y_s.sum()), int((1 - y_s).sum())) < MIN_CLASS_TRAIN:
                record["note"] = "target missing or single-class in synthetic data"
                print(f"  {run_id}: not evaluable ({record['note']}) -- itself a utility finding")
            else:
                x_s = self._encode(synth.loc[y_s.index], feature_cols, categories)
                record["auc"] = self._fit_score(x_s, y_s, x_test, y_test_all, config.seed)
                record["auc_gap"] = round(baseline - record["auc"], 4)
                # Second learner: the utility claim should not be an
                # artifact of one model class.
                record["auc_logreg"] = self._fit_score_logreg(x_s, y_s, x_test, y_test_all, config.seed)
                record["auc_gap_logreg"] = round(baseline_lr - record["auc_logreg"], 4)
                # Augmentation: does real+synthetic beat real alone?
                x_aug = pd.concat([x_train, x_s], ignore_index=True)
                y_aug = pd.concat([y_train_all.reset_index(drop=True),
                                   y_s.reset_index(drop=True)], ignore_index=True)
                record["auc_augmented"] = self._fit_score(x_aug, y_aug, x_test, y_test_all, config.seed)
                record["augmentation_delta"] = round(record["auc_augmented"] - baseline, 4)
                print(f"  {run_id}: TSTR AUC={record['auc']} (gap {record['auc_gap']:+.4f}) | "
                      f"LogReg gap {record['auc_gap_logreg']:+.4f} | "
                      f"augmentation {record['augmentation_delta']:+.4f}")
            entry["tstr"].append(record)
        return entry

    @staticmethod
    def _fit_categories(df, feature_cols) -> dict:
        import pandas as pd

        cats = {}
        for c in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
                cats[c] = pd.Index(sorted(s.unique()))
        return cats

    @staticmethod
    def _encode(df, feature_cols, categories):
        """Numeric passthrough (NaN preserved -- the trees handle it),
        categoricals as integer codes from the REAL data's category list;
        unseen synthetic categories become NaN. Columns absent from a
        frame (width-limited runs) become all-NaN, which the trees treat
        as uninformative rather than crashing."""
        import numpy as np
        import pandas as pd

        # Built as a dict and materialized once -- inserting ~200 columns
        # one at a time fragments the frame.
        out = {}
        for c in feature_cols:
            if c not in df.columns:
                out[c] = np.full(len(df), np.nan)
            elif c in categories:
                s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
                codes = categories[c].get_indexer(s).astype(float)
                codes[codes < 0] = np.nan
                out[c] = codes
            else:
                out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        return pd.DataFrame(out, index=df.index)

    @staticmethod
    def _fit_score(x_train, y_train, x_test, y_test, seed) -> float:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        # A feature with fewer than 2 distinct observed values in THIS
        # training matrix carries nothing to split on -- and sklearn's
        # histogram binning crashes outright on it (sliding_window_view
        # of size 2 over <2 distinct values). Columns can be degenerate
        # in a subset (one synthesizer's output, a width-limited run)
        # while varying in the full data, so this is decided per training
        # matrix, and the test matrix is subset to the same columns.
        usable = [c for c in x_train.columns if x_train[c].nunique(dropna=True) >= 2]
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(x_train[usable], y_train)
        return round(float(roc_auc_score(y_test, model.predict_proba(x_test[usable])[:, 1])), 4)

    @staticmethod
    def _fit_score_logreg(x_train, y_train, x_test, y_test, seed) -> float:
        """Second model class for TSTR robustness: median-imputed,
        standardized logistic regression."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        usable = [c for c in x_train.columns if x_train[c].nunique(dropna=True) >= 2]
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(x_train[usable], y_train)
        return round(float(roc_auc_score(y_test, clf.predict_proba(x_test[usable])[:, 1])), 4)

    # --- reporting ---

    def _write(self, results: dict, out_dir: str) -> None:
        json_path = os.path.join(out_dir, "DT4H_Utility_TSTR.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved utility results (JSON) -> {json_path}")

        md_path = os.path.join(out_dir, "DT4H_Utility_TSTR.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"Saved utility results (Markdown) -> {md_path}")

    @staticmethod
    def _render_markdown(r: dict) -> str:
        lines = [
            "# Utility: Train-Synthetic, Test-Real (TSTR)",
            "",
            "A gradient-boosting classifier is trained on the real TRAINING split "
            "(baseline) and on each synthetic dataset, then both are scored on the "
            "HOLDOUT split -- real patients that neither the generators nor either "
            "classifier ever saw. The closer the TSTR AUC is to the baseline, the more "
            "useful the synthetic data is for actual modelling work.",
            "",
        ]
        if not r.get("targets"):
            lines.append(f"_No eligible targets: {r.get('note', '')}_")
            return "\n".join(lines) + "\n"

        lines.append(f"Real train: {r.get('n_train')} rows | holdout test: "
                     f"{r.get('n_holdout')} rows")

        for t in r["targets"]:
            lines += ["", f"## `{t['target']}`"]
            if t.get("note"):
                lines.append(f"_{t['note']}_")
                continue
            lines += [
                f"train {t['n_train_labelled']} labelled ({t.get('positives_train')} positive), "
                f"holdout {t['n_holdout_labelled']} labelled ({t.get('positives_holdout')} positive) | "
                f"baseline AUC **{t['baseline_auc']}** (HistGB) / {t.get('baseline_auc_logreg')} (LogReg)", "",
                "| run | TSTR AUC | gap | LogReg gap | augmentation Δ |", "|---|---|---|---|---|"]
            for s in t["tstr"]:
                if s.get("auc") is None:
                    lines.append(f"| {s['run_id']} | - | {s.get('note', '')} | - | - |")
                else:
                    lr = f"{s['auc_gap_logreg']:+.4f}" if s.get("auc_gap_logreg") is not None else "-"
                    ag = f"{s['augmentation_delta']:+.4f}" if s.get("augmentation_delta") is not None else "-"
                    lines.append(f"| {s['run_id']} | {s['auc']} | {s['auc_gap']:+.4f} | {lr} | {ag} |")

        if r.get("grouped_auc_gap"):
            lines += ["", "## Per (model, ε) across seeds and targets",
                      "",
                      "Gaps vs baseline, lower is better; augmentation Δ is the AUC change from "
                      "training on real+synthetic vs real alone (positive = synthetic data adds value).",
                      "", "| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ |",
                      "|---|---|---|---|---|---|"]
            for g in r["grouped_auc_gap"]:
                eps = f"{g['epsilon']:g}" if g.get("epsilon") is not None else "-"
                sd = f" ± {g['sd_gap']}" if g.get("sd_gap") is not None else ""
                lines.append(f"| {g['synthesizer']} | {eps} | {g['n_runs']} "
                             f"| {g['mean_gap']:+.4f}{sd} | {g.get('mean_gap_logreg')} "
                             f"| {g.get('mean_augmentation_delta')} |")
        return "\n".join(lines) + "\n"

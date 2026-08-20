"""
Step: utility

Train-Synthetic-Test-Real (TSTR): the standard machine-learning utility
check for a released synthetic dataset. For each clinical outcome target,
a gradient-boosting classifier is trained twice --

  * baseline: on a real training split;
  * TSTR:     on each synthetic dataset (full);

-- and both are scored on the SAME held-out real test split. A synthetic
dataset with good utility yields nearly the baseline AUC: models trained
on it transfer to real patients. Distribution metrics (evaluate step) say
the data LOOKS right; this says the data WORKS.

Targets are selected from the feature-set metadata's declared outcome
variables (BOOLEAN ones with enough of both classes), and every declared
outcome column is excluded from the feature set -- outcomes correlate
with each other, and leaking one outcome into the features of another
would inflate every AUC.

Everything runs in the decoded (released) representation: nulls are
nulls, exactly as a user of the published dataset would see them.
Gradient boosting handles missing values natively, so missingness
patterns participate in prediction the same way for real and synthetic.

Aggregate statistics only are written -- safe to commit.
"""

import glob
import json
import os

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep

MIN_CLASS_TRAIN = 10   # each class must appear at least this often in training data
MIN_CLASS_TEST = 5     # and this often in the real test split
TEST_FRACTION = 0.25


class UtilityStep(PipelineStep):
    name = "utility"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd

        from pipeline.steps.generate.step import GenerateStep
        from pipeline.steps.preprocess.transforms import load_variable_metadata

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError(
                f"No DT4H_Synthetic_*.csv in {config.step_dir('generate')} -- run the generate step first."
            )

        import polars as pl

        real = pl.read_parquet(config.preprocessed_output_path).to_pandas()
        real, _ = GenerateStep._decode_numeric_missing(real, config)

        var_meta = load_variable_metadata(config.metadata_path)
        outcome_cols = self._declared_outcomes(config)
        targets = self._select_targets(real, outcome_cols, config)

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        if not targets:
            print("⚠️  No eligible outcome targets (need a BOOLEAN outcome with at least "
                  f"{MIN_CLASS_TRAIN} records of each class). Writing an empty report.")
            results = {"targets": [], "note": "no eligible targets"}
            self._write(results, out_dir)
            return

        feature_cols = [c for c in real.columns if c not in outcome_cols]
        print(f"Targets: {[t for t in targets]}")
        print(f"Features: {len(feature_cols)} columns (all {len([c for c in outcome_cols if c in real.columns])} "
              f"declared outcome columns excluded to prevent cross-outcome leakage)")

        results = {"test_fraction": TEST_FRACTION, "seed": config.seed, "targets": []}

        for target in targets:
            print(f"\nTarget: {target}")
            entry = self._evaluate_target(real, synthetic_files, target, feature_cols, config)
            results["targets"].append(entry)

        # per-synthesizer aggregate across targets
        by_synth: dict[str, list] = {}
        for t in results["targets"]:
            for r in t["tstr"]:
                if r.get("auc") is not None and t.get("baseline_auc") is not None:
                    by_synth.setdefault(r["synthesizer"], []).append(t["baseline_auc"] - r["auc"])
        results["aggregate_auc_gap"] = {
            name: round(sum(v) / len(v), 4) for name, v in sorted(by_synth.items())
        }
        print("\nMean AUC gap vs baseline (lower is better):")
        for name, gap in results["aggregate_auc_gap"].items():
            print(f"  {name}: {gap:+.4f}")

        self._write(results, out_dir)

    # --- target selection ---

    def _declared_outcomes(self, config: PipelineConfig) -> set[str]:
        with open(config.metadata_path) as f:
            raw = json.load(f)
        return {v["name"] for v in raw["entries"][0]["outcomes"]}

    def _select_targets(self, real, outcome_cols: set[str], config: PipelineConfig) -> list[str]:
        import pandas as pd

        if config.utility_targets:
            picked = []
            for t in config.utility_targets:
                if t in real.columns and self._to_binary(real[t]) is not None:
                    picked.append(t)
                else:
                    print(f"⚠️  Requested utility target '{t}' is absent or not a "
                          "two-class boolean column; skipped.")
            return picked

        candidates = []
        for col in outcome_cols:
            if col not in real.columns:
                continue
            y = self._to_binary(real[col])
            if y is None:
                continue
            pos, neg = int(y.sum()), int((1 - y).sum())
            if min(pos, neg) < MIN_CLASS_TRAIN + MIN_CLASS_TEST:
                continue
            balance = min(pos, neg) / max(pos, neg)
            candidates.append((balance, col))
        candidates.sort(reverse=True)
        return [col for _, col in candidates[: config.utility_max_targets]]

    @staticmethod
    def _to_binary(series):
        """Boolean-ish column -> 0/1 with Missing rows dropped; None if it
        is not a two-class column."""
        import pandas as pd

        s = series.astype("object").where(series.notna(), "missing").astype(str).str.lower()
        s = s[s != "missing"]
        values = set(s.unique())
        if not values or not values <= {"true", "false"}:
            return None
        return (s == "true").astype(int)

    # --- model fitting ---

    def _evaluate_target(self, real, synthetic_files, target, feature_cols, config) -> dict:
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        y_real = self._to_binary(real[target])
        real_rows = real.loc[y_real.index]

        train_idx, test_idx = train_test_split(
            y_real.index, test_size=TEST_FRACTION, random_state=config.seed,
            stratify=y_real,
        )
        categories = self._fit_categories(real_rows, feature_cols)
        x_train = self._encode(real_rows.loc[train_idx], feature_cols, categories)
        x_test = self._encode(real_rows.loc[test_idx], feature_cols, categories)
        y_train, y_test = y_real.loc[train_idx], y_real.loc[test_idx]

        entry = {
            "target": target,
            "n_real": int(len(y_real)),
            "positives": int(y_real.sum()),
            "baseline_auc": None,
            "tstr": [],
        }
        if min(y_test.sum(), (1 - y_test).sum()) < MIN_CLASS_TEST:
            entry["note"] = "test split too imbalanced; skipped"
            print("  (skipped: test split too imbalanced)")
            return entry

        baseline = self._fit_score(x_train, y_train, x_test, y_test, config.seed)
        entry["baseline_auc"] = baseline
        print(f"  baseline (train real -> test real): AUC={baseline}")

        for path in synthetic_files:
            name = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            record = {"synthesizer": name, "auc": None}
            y_s = self._to_binary(synth[target]) if target in synth.columns else None
            if y_s is None or min(int(y_s.sum()), int((1 - y_s).sum())) < MIN_CLASS_TRAIN:
                record["note"] = "target missing or single-class in synthetic data"
                print(f"  {name}: not evaluable ({record['note']}) -- itself a utility finding")
            else:
                x_s = self._encode(synth.loc[y_s.index], feature_cols, categories)
                record["auc"] = self._fit_score(x_s, y_s, x_test, y_test, config.seed)
                record["auc_gap"] = round(baseline - record["auc"], 4)
                print(f"  {name}: TSTR AUC={record['auc']} (gap {record['auc_gap']:+.4f})")
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
        unseen synthetic categories become NaN."""
        import numpy as np
        import pandas as pd

        out = pd.DataFrame(index=df.index)
        for c in feature_cols:
            if c not in df.columns:
                out[c] = np.nan
            elif c in categories:
                s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
                codes = categories[c].get_indexer(s).astype(float)
                codes[codes < 0] = np.nan
                out[c] = codes
            else:
                out[c] = pd.to_numeric(df[c], errors="coerce")
        return out

    @staticmethod
    def _fit_score(x_train, y_train, x_test, y_test, seed) -> float:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        # A feature with fewer than 2 distinct observed values in THIS
        # training matrix carries nothing to split on -- and sklearn's
        # histogram binning crashes outright on it (sliding_window_view
        # of size 2 over <2 distinct values). Columns can be degenerate
        # in a subset (a train split, or one synthesizer's output) while
        # varying in the full data, so this is decided per training
        # matrix, and the test matrix is subset to the same columns.
        usable = [c for c in x_train.columns if x_train[c].nunique(dropna=True) >= 2]
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(x_train[usable], y_train)
        return round(float(roc_auc_score(y_test, model.predict_proba(x_test[usable])[:, 1])), 4)

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
            "A gradient-boosting classifier is trained on real data (baseline) and on each "
            "synthetic dataset, then both are scored on the same held-out real test split. "
            "The closer the TSTR AUC is to the baseline, the more useful the synthetic data "
            "is for actual modelling work.",
            "",
        ]
        if not r.get("targets"):
            lines.append(f"_No eligible targets: {r.get('note', '')}_")
            return "\n".join(lines) + "\n"

        for t in r["targets"]:
            lines += ["", f"## `{t['target']}`",
                      f"{t['n_real']} labelled real records ({t['positives']} positive) | "
                      f"baseline AUC **{t['baseline_auc']}**", "",
                      "| synthesizer | TSTR AUC | gap vs baseline |", "|---|---|---|"]
            for s in t["tstr"]:
                if s.get("auc") is None:
                    lines.append(f"| {s['synthesizer']} | - | {s.get('note', '')} |")
                else:
                    lines.append(f"| {s['synthesizer']} | {s['auc']} | {s['auc_gap']:+.4f} |")

        if r.get("aggregate_auc_gap"):
            lines += ["", "## Mean AUC gap across targets (lower is better)", "",
                      "| synthesizer | mean gap |", "|---|---|"]
            for name, gap in r["aggregate_auc_gap"].items():
                lines.append(f"| {name} | {gap:+.4f} |")
        return "\n".join(lines) + "\n"

"""
Step: attacks

Adversarial privacy evaluation -- the DCR analysis (privacy step) bounds
record-copying; this step actively ATTACKS each synthetic dataset and
reports how well the attacks do. The holdout split makes both attacks
honest: training members and unseen non-members are real patients from
the same distribution.

  * MEMBERSHIP INFERENCE (distance attack): the attacker scores each
    real record by its distance to the nearest synthetic record and
    predicts "was in the training set" for close ones. Reported as the
    attack's AUC over train (members) vs holdout (non-members); 0.5 =
    the synthetic data reveals nothing about who was trained on. A
    bootstrap interval says whether an AUC above 0.5 is even
    distinguishable from noise.
  * ATTRIBUTE INFERENCE: an attacker who knows a patient's
    quasi-identifiers (age, gender, admission year) looks up the most
    similar synthetic records and infers a sensitive attribute. The
    honest headline is the MEMBERSHIP ADVANTAGE: attack accuracy on
    training members minus accuracy on unseen holdout patients. Any
    population-level inference (both accuracies above baseline, equally)
    is the intended purpose of releasing data; only member-specific
    advantage is leakage.

If the `anonymeter` package is installed, its singling-out and
linkability evaluators run as well; otherwise they are skipped with a
note (the two native attacks above are the primary evidence).

Aggregate statistics only -- safe to commit.
"""

import glob
import json
import os
import time

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.steps.base import PipelineStep
from pipeline.steps.privacy.distance import build_encoder, nearest_two_distances

QUASI_IDENTIFIERS = ("patient_demographics_age", "patient_demographics_gender",
                     "encounters_admissionYear")
SENSITIVE_CANDIDATES = ("cause_of_death_isAllCause_f5a_w5a_first",
                        "ckd_severity_from_calculated_egfr",
                        "nyha_nyha_pET",
                        "conditions_dm")
N_BOOTSTRAP = 500


class AttacksStep(PipelineStep):
    name = "attacks"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd
        import polars as pl

        from pipeline.steps.preprocess.transforms import NUMERIC_ENCODING_FILENAME

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError("No synthetic files -- run the generate step first.")

        train = pl.read_parquet(config.train_output_path).to_pandas()
        holdout = pl.read_parquet(config.holdout_output_path).to_pandas()
        encoding_path = os.path.join(config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)
        encoding = {}
        if os.path.exists(encoding_path):
            with open(encoding_path) as f:
                encoding = json.load(f)

        print(f"Members (train): {train.shape[0]} | non-members (holdout): {holdout.shape[0]}")
        encode, num_cols, cat_cols = build_encoder(train, encoding)
        train_num, train_cat = encode(train)
        hold_num, hold_cat = encode(holdout)

        sensitive = [c for c in SENSITIVE_CANDIDATES if c in train.columns]
        print(f"Attribute-inference sensitive targets: {sensitive}")

        results = {"n_members": int(train.shape[0]), "n_nonmembers": int(holdout.shape[0]),
                   "quasi_identifiers": [q for q in QUASI_IDENTIFIERS if q in train.columns],
                   "runs": []}

        rng = np.random.default_rng(config.seed)

        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            print(f"\nAttacking '{run_id}'...")
            t0 = time.time()

            missing = [c for c in train.columns if c not in synth.columns]
            if missing:
                pad = pd.DataFrame(pd.NA, index=synth.index, columns=missing)
                synth = pd.concat([synth, pad], axis=1)
            synth_num, synth_cat = encode(synth)

            entry = {"run_id": run_id}
            entry["membership_inference"] = self._mia(
                train_num, train_cat, hold_num, hold_cat, synth_num, synth_cat, rng)
            m = entry["membership_inference"]
            verdict = "✅" if m["attack_auc_ci95"][0] <= 0.5 <= m["attack_auc_ci95"][1] or m["attack_auc"] < 0.55 else "🚨"
            print(f"  {verdict} MIA attack AUC = {m['attack_auc']} "
                  f"(95% CI {m['attack_auc_ci95']}; 0.5 = no membership leakage)")

            entry["attribute_inference"] = self._aia(train, holdout, synth, sensitive)
            for a in entry["attribute_inference"]:
                print(f"  AIA {a['sensitive']}: member acc {a['accuracy_members']} vs "
                      f"non-member {a['accuracy_nonmembers']} (baseline {a['baseline_accuracy']}) "
                      f"-> membership advantage {a['membership_advantage']:+.4f}")

            entry["anonymeter"] = self._anonymeter(train, holdout, synth)
            entry["duration_seconds"] = round(time.time() - t0, 1)
            results["runs"].append(entry)

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "DT4H_Privacy_Attacks.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        md_path = os.path.join(out_dir, "DT4H_Privacy_Attacks.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"\nSaved attack results -> {json_path} / {md_path}")

    # --- membership inference ---

    def _mia(self, train_num, train_cat, hold_num, hold_cat, synth_num, synth_cat, rng) -> dict:
        from sklearn.metrics import roc_auc_score

        d_mem, _ = nearest_two_distances(train_num, train_cat, synth_num, synth_cat)
        d_non, _ = nearest_two_distances(hold_num, hold_cat, synth_num, synth_cat)
        scores = np.concatenate([-d_mem, -d_non])  # closer = "more likely member"
        labels = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
        auc = float(roc_auc_score(labels, scores))

        boot = []
        n = len(labels)
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            if labels[idx].min() == labels[idx].max():
                continue
            boot.append(roc_auc_score(labels[idx], scores[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"attack": "nearest-synthetic-distance",
                "attack_auc": round(auc, 4),
                "attack_auc_ci95": [round(float(lo), 4), round(float(hi), 4)],
                "member_median_distance": round(float(np.median(d_mem)), 6),
                "nonmember_median_distance": round(float(np.median(d_non)), 6)}

    # --- attribute inference ---

    def _aia(self, train, holdout, synth, sensitive) -> list[dict]:
        import pandas as pd

        quasi = [q for q in QUASI_IDENTIFIERS if q in train.columns and q in synth.columns]
        if not quasi or not sensitive:
            return []

        def _q_matrix(df):
            cols = []
            for q in quasi:
                v = pd.to_numeric(df[q], errors="coerce")
                if v.notna().mean() > 0.5:
                    rng_ = v.max() - v.min()
                    cols.append(((v - v.min()) / rng_ if rng_ > 0 else v * 0).fillna(0.5).to_numpy())
                else:
                    cols.append(pd.factorize(df[q].astype(str))[0].astype(float))
            return np.column_stack(cols)

        qs = _q_matrix(synth)
        out = []
        for s in sensitive:
            if s not in synth.columns:
                continue
            synth_vals = synth[s].astype("object").where(synth[s].notna(), "Missing").astype(str)
            baseline = float(synth_vals.value_counts(normalize=True).iloc[0])

            def _attack(df_real):
                qr = _q_matrix(df_real)
                # nearest synthetic record in quasi-identifier space
                pred = []
                for start in range(0, len(qr), 512):
                    chunk = qr[start:start + 512]
                    d = np.abs(chunk[:, None, :] - qs[None, :, :]).sum(axis=2)
                    pred.extend(synth_vals.iloc[np.argmin(d, axis=1)].tolist())
                truth = df_real[s].astype("object").where(df_real[s].notna(), "Missing").astype(str)
                return float((np.asarray(pred) == truth.to_numpy()).mean())

            acc_m = _attack(train)
            acc_n = _attack(holdout)
            out.append({"sensitive": s,
                        "baseline_accuracy": round(baseline, 4),
                        "accuracy_members": round(acc_m, 4),
                        "accuracy_nonmembers": round(acc_n, 4),
                        "membership_advantage": round(acc_m - acc_n, 4)})
        return out

    # --- optional anonymeter ---

    def _anonymeter(self, train, holdout, synth):
        try:
            from anonymeter.evaluators import LinkabilityEvaluator, SinglingOutEvaluator
        except ImportError:
            return {"note": "anonymeter not installed; singling-out/linkability skipped "
                            "(native MIA and AIA above are the primary evidence)"}
        try:
            so = SinglingOutEvaluator(ori=train, syn=synth, control=holdout, n_attacks=200)
            so.evaluate(mode="univariate")
            so_risk = so.risk()
            aux = [q for q in QUASI_IDENTIFIERS if q in train.columns]
            link = LinkabilityEvaluator(ori=train, syn=synth, control=holdout,
                                        aux_cols=(aux[:2], aux[2:] or aux[:1]), n_attacks=200)
            link.evaluate()
            link_risk = link.risk()
            return {"singling_out_risk": round(float(so_risk.value), 4),
                    "linkability_risk": round(float(link_risk.value), 4)}
        except Exception as e:
            return {"note": f"anonymeter failed: {type(e).__name__}: {e}"}

    @staticmethod
    def _render_markdown(r: dict) -> str:
        lines = [
            "# Adversarial Privacy Attacks",
            "",
            f"Members: {r['n_members']} training records; non-members: {r['n_nonmembers']} "
            "holdout records (real, unseen patients). Membership inference AUC of 0.5 means "
            "the synthetic data reveals nothing about who was in the training set. Attribute "
            "inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy "
            "on non-members; population-level inference (both above baseline, equally) is the "
            "intended use of released data, only member-specific advantage is leakage.",
            "",
            "| run | MIA AUC (95% CI) | worst AIA membership advantage | anonymeter |",
            "|---|---|---|---|",
        ]
        for run in r["runs"]:
            m = run["membership_inference"]
            adv = max((a["membership_advantage"] for a in run["attribute_inference"]),
                      default=None)
            anon = run.get("anonymeter", {})
            anon_cell = (f"SO {anon['singling_out_risk']}, link {anon['linkability_risk']}"
                         if "singling_out_risk" in anon else "skipped")
            flag = "" if m["attack_auc"] < 0.55 else " 🚨"
            lines.append(f"| {run['run_id']}{flag} | {m['attack_auc']} "
                         f"({m['attack_auc_ci95'][0]}-{m['attack_auc_ci95'][1]}) "
                         f"| {adv if adv is not None else '-'} | {anon_cell} |")
        lines += ["", "## Attribute inference detail", "",
                  "| run | sensitive attribute | baseline | member acc | non-member acc | advantage |",
                  "|---|---|---|---|---|---|"]
        for run in r["runs"]:
            for a in run["attribute_inference"]:
                lines.append(f"| {run['run_id']} | {a['sensitive']} | {a['baseline_accuracy']} "
                             f"| {a['accuracy_members']} | {a['accuracy_nonmembers']} "
                             f"| {a['membership_advantage']:+.4f} |")
        return "\n".join(lines) + "\n"

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
from pipeline.common.alignment import align_categorical_case
from pipeline.steps.base import PipelineStep
from pipeline.steps.privacy.distance import (
    build_encoder,
    nearest_k_distances,
    nearest_two_distances,
)

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

        # Per-record vulnerability axis, computed once: how ATYPICAL each
        # member is (distance to their 5th-nearest fellow member, self
        # excluded). Attack success stratified by this answers the
        # governance question aggregates cannot -- WHO is at risk: is any
        # residual membership signal concentrated on outlier patients?
        print("Computing member atypicality (distance to 5th-nearest member)...")
        d5 = nearest_k_distances(train_num, train_cat, train_num, train_cat,
                                 k=5, exclude_self=True)[:, -1]
        cuts = np.quantile(d5, [0.25, 0.5, 0.75])
        # Rank-based quartiles, not value cuts: heavily tied distances
        # (many near-duplicate records) would otherwise collapse the
        # bins into one. Ranks always yield four equal groups,
        # deterministically.
        order = np.argsort(d5, kind="stable")
        member_quartile = np.empty(len(d5), dtype=int)
        member_quartile[order] = (np.arange(len(d5)) * 4) // len(d5)
        print(f"  atypicality quartile cuts (informational): "
              f"{[round(float(c), 4) for c in cuts]}")

        results = {"n_members": int(train.shape[0]), "n_nonmembers": int(holdout.shape[0]),
                   "quasi_identifiers": [q for q in QUASI_IDENTIFIERS if q in train.columns],
                   "member_atypicality": {"k": 5,
                                          "quartile_cuts": [round(float(c), 6) for c in cuts]},
                   "runs": []}

        rng = np.random.default_rng(config.seed)

        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            synth, _ = align_categorical_case(synth, train)
            print(f"\nAttacking '{run_id}'...")
            t0 = time.time()

            missing = [c for c in train.columns if c not in synth.columns]
            if missing:
                pad = pd.DataFrame(pd.NA, index=synth.index, columns=missing)
                synth = pd.concat([synth, pad], axis=1)
            synth_num, synth_cat = encode(synth)

            entry = {"run_id": run_id}
            entry["membership_inference"] = self._mia(
                train_num, train_cat, hold_num, hold_cat, synth_num, synth_cat, rng,
                member_quartile=member_quartile)
            m = entry["membership_inference"]
            worst = max(m["attack_auc"], m["learned_attack_auc"])
            worst_ci = (m["attack_auc_ci95"] if m["attack_auc"] >= m["learned_attack_auc"]
                        else m["learned_attack_auc_ci95"])
            verdict = "✅" if worst_ci[0] <= 0.5 <= worst_ci[1] or worst < 0.55 else "🚨"
            print(f"  {verdict} MIA distance attack AUC = {m['attack_auc']} "
                  f"(95% CI {m['attack_auc_ci95']}); learned attack AUC = "
                  f"{m['learned_attack_auc']} (95% CI {m['learned_attack_auc_ci95']}; "
                  f"0.5 = no membership leakage)")
            if m.get("mia_by_atypicality"):
                qs = m["mia_by_atypicality"]
                print("  who is at risk: "
                      + " | ".join(f"{q['quartile'].split()[0]} AUC {q['attack_auc']}"
                                   for q in qs))
            print(f"  empirical ε lower bound (DP audit): "
                  f"{m['empirical_epsilon_lower_bound']} "
                  f"(a DP run's claimed budget must exceed this)")

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

    def _mia(self, train_num, train_cat, hold_num, hold_cat, synth_num, synth_cat, rng,
             member_quartile=None) -> dict:
        from sklearn.metrics import roc_auc_score

        d_mem, d2_mem = nearest_two_distances(train_num, train_cat, synth_num, synth_cat)
        d_non, d2_non = nearest_two_distances(hold_num, hold_cat, synth_num, synth_cat)
        scores = np.concatenate([-d_mem, -d_non])  # closer = "more likely member"
        labels = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
        auc = float(roc_auc_score(labels, scores))

        def _boot_ci(sc):
            boot = []
            n = len(labels)
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, n, n)
                if labels[idx].min() == labels[idx].max():
                    continue
                boot.append(roc_auc_score(labels[idx], sc[idx]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            return [round(float(lo), 4), round(float(hi), 4)]

        ci = _boot_ci(scores)

        # LEARNED attack: a cross-validated classifier over the distance
        # profile (nearest, second-nearest, their ratio) -- strictly more
        # powerful than thresholding the nearest distance alone, so a
        # chance-level result here is stronger evidence than the
        # single-feature attack's. Scored out-of-fold: the attack model
        # never sees the records it scores.
        eps = 1e-12
        feats = np.column_stack([
            np.concatenate([d_mem, d_non]),
            np.concatenate([d2_mem, d2_non]),
            np.concatenate([d_mem / (d2_mem + eps), d_non / (d2_non + eps)]),
        ])
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        oof = np.zeros(len(labels))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr_idx, te_idx in skf.split(feats, labels):
            scaler = StandardScaler().fit(feats[tr_idx])
            clf = LogisticRegression(max_iter=1000).fit(
                scaler.transform(feats[tr_idx]), labels[tr_idx])
            oof[te_idx] = clf.decision_function(scaler.transform(feats[te_idx]))
        learned_auc = float(roc_auc_score(labels, oof))
        learned_ci = _boot_ci(oof)

        # EMPIRICAL DP AUDIT: an attack-derived lower bound on the
        # effective epsilon (Jagielski/Steinke-style): over score
        # thresholds, the largest log-ratio of Clopper-Pearson-bounded
        # accept/reject rates between members and non-members. A
        # chance-level attack yields ~0; a bound approaching a DP run's
        # CLAIMED budget would falsify the accounting. Both attacks'
        # scores are audited and the larger bound is reported.
        emp_eps = max(self._empirical_epsilon(-d_mem, -d_non),
                      self._empirical_epsilon(oof[:len(d_mem)], oof[len(d_mem):]))

        out = {"attack": "nearest-synthetic-distance",
               "attack_auc": round(auc, 4),
               "attack_auc_ci95": ci,
               "learned_attack": "cv-logreg over (d1, d2, d1/d2)",
               "learned_attack_auc": round(learned_auc, 4),
               "learned_attack_auc_ci95": learned_ci,
               "member_median_distance": round(float(np.median(d_mem)), 6),
               "nonmember_median_distance": round(float(np.median(d_non)), 6),
               "empirical_epsilon_lower_bound": emp_eps,
               "empirical_epsilon_method": "max log(CP05(TPR)/CP95(FPR)) over "
                                           "thresholds, both attack score sets, "
                                           "both accept/reject directions"}

        # WHO is at risk: attack success per member-atypicality quartile
        # (each quartile's members scored against all non-members).
        if member_quartile is not None:
            names = ["Q1 (most typical)", "Q2", "Q3", "Q4 (most atypical)"]
            by_q = []
            for qi in range(4):
                mask = member_quartile == qi
                if not mask.any():
                    continue
                sc = np.concatenate([-d_mem[mask], -d_non])
                lb = np.concatenate([np.ones(int(mask.sum())), np.zeros(len(d_non))])
                by_q.append({
                    "quartile": names[qi],
                    "n_members": int(mask.sum()),
                    "attack_auc": round(float(roc_auc_score(lb, sc)), 4),
                    "member_median_distance": round(float(np.median(d_mem[mask])), 6),
                })
            out["mia_by_atypicality"] = by_q
        return out

    @staticmethod
    def _empirical_epsilon(scores_pos, scores_neg) -> float:
        """Attack-derived lower bound on effective epsilon. For each
        threshold, epsilon >= ln(TPR/FPR) must hold for any
        (epsilon, ~0)-DP mechanism; using the 95% Clopper-Pearson lower
        bound of the accept rate on members against the upper bound on
        non-members (and the reject-direction complement) yields a
        conservative empirical bound. ~0 at chance."""
        from scipy.stats import beta

        pos = np.asarray(scores_pos, dtype=float)
        neg = np.asarray(scores_neg, dtype=float)
        n_p, n_n = len(pos), len(neg)
        thresholds = np.unique(np.quantile(np.concatenate([pos, neg]),
                                           np.linspace(0.02, 0.98, 49)))
        best = 0.0
        for thr in thresholds:
            tp = int((pos >= thr).sum())
            fp = int((neg >= thr).sum())
            tpr_lo = float(beta.ppf(0.05, tp, n_p - tp + 1)) if tp > 0 else 0.0
            fpr_hi = float(beta.ppf(0.95, fp + 1, n_n - fp)) if fp < n_n else 1.0
            if tpr_lo > 0 and fpr_hi > 0:
                best = max(best, np.log(tpr_lo / fpr_hi))
            tn, fn = n_n - fp, n_p - tp
            tnr_lo = float(beta.ppf(0.05, tn, n_n - tn + 1)) if tn > 0 else 0.0
            fnr_hi = float(beta.ppf(0.95, fn + 1, n_p - fn)) if fn < n_p else 1.0
            if tnr_lo > 0 and fnr_hi > 0:
                best = max(best, np.log(tnr_lo / fnr_hi))
        return round(float(max(best, 0.0)), 4)

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

    @staticmethod
    def _harmonized_frames(train, *others):
        """anonymeter requires IDENTICAL schemas across ori/syn/control and
        chokes on arrow-backed extension dtypes. The column type is decided
        ONCE from the train frame and every frame is coerced to it: train-
        numeric columns become float64 everywhere; train-categorical columns
        become plain strings everywhere -- with integer-safe formatting, so
        a synthetic '2017' parsed as a number by read_csv round-trips to
        '2017', never '2017.0'."""
        import pandas as pd

        numeric_cols = {c for c in train.columns
                        if pd.api.types.is_numeric_dtype(train[c])
                        and not pd.api.types.is_bool_dtype(train[c])}

        def _as_cat_str(v):
            if v is None or (isinstance(v, float) and v != v):
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                return str(int(f)) if f.is_integer() else str(v)
            return str(v)

        def convert(df):
            out = {}
            for c in train.columns:
                col = df[c] if c in df.columns else pd.Series([None] * len(df), index=df.index)
                if c in numeric_cols:
                    out[c] = pd.to_numeric(col, errors="coerce").astype("float64")
                else:
                    out[c] = col.astype("object").where(col.notna(), None).map(_as_cat_str)
            return pd.DataFrame(out, index=df.index)

        return [convert(f) for f in (train, *others)]

    def _anonymeter(self, train, holdout, synth):
        try:
            from anonymeter.evaluators import LinkabilityEvaluator, SinglingOutEvaluator
        except Exception as e:  # not just ImportError: a numpy-version
            # mismatch can surface as AttributeError/TypeError deep in
            # numba's import chain, and the attacks step must survive it.
            return {"note": f"anonymeter unavailable ({type(e).__name__}: {e}); "
                            "singling-out/linkability skipped "
                            "(native MIA and AIA above are the primary evidence)"}
        train, holdout, synth = self._harmonized_frames(train, holdout, synth)
        out = {}
        try:
            so = SinglingOutEvaluator(ori=train, syn=synth, control=holdout, n_attacks=200)
            so.evaluate(mode="univariate")
            out["singling_out_risk"] = round(float(so.risk().value), 4)
        except Exception as e:
            out["singling_out_note"] = f"failed: {type(e).__name__}: {e}"
        aux = [q for q in QUASI_IDENTIFIERS if q in train.columns]
        if len(aux) >= 2:
            try:
                link = LinkabilityEvaluator(ori=train, syn=synth, control=holdout,
                                            aux_cols=(aux[:2], aux[2:] or aux[:1]),
                                            n_attacks=200)
                link.evaluate()
                out["linkability_risk"] = round(float(link.risk().value), 4)
            except Exception as e:
                out["linkability_note"] = f"failed: {type(e).__name__}: {e}"
        else:
            out["linkability_note"] = "skipped: fewer than 2 quasi-identifier columns"
        return out

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
            "| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |",
            "|---|---|---|---|---|---|",
        ]
        for run in r["runs"]:
            m = run["membership_inference"]
            adv = max((a["membership_advantage"] for a in run["attribute_inference"]),
                      default=None)
            anon = run.get("anonymeter", {})
            so = anon.get("singling_out_risk")
            lk = anon.get("linkability_risk")
            anon_cell = (f"SO {so if so is not None else '-'}, link {lk if lk is not None else '-'}"
                         if (so is not None or lk is not None) else "skipped")
            l_auc = m.get("learned_attack_auc")
            l_ci = m.get("learned_attack_auc_ci95") or ["-", "-"]
            learned_cell = (f"{l_auc} ({l_ci[0]}-{l_ci[1]})" if l_auc is not None else "-")
            flag = "" if max(m["attack_auc"], l_auc or 0) < 0.55 else " 🚨"
            lines.append(f"| {run['run_id']}{flag} | {m['attack_auc']} "
                         f"({m['attack_auc_ci95'][0]}-{m['attack_auc_ci95'][1]}) "
                         f"| {learned_cell} "
                         f"| {m.get('empirical_epsilon_lower_bound', '-')} "
                         f"| {adv if adv is not None else '-'} | {anon_cell} |")
        lines += [
            "", "## Who is at risk: membership inference by patient atypicality", "",
            "Members are split into quartiles of atypicality (distance to their "
            "5th-nearest fellow member); each quartile is attacked against all "
            "non-members. A model that leaks selectively on unusual patients shows "
            "an elevated Q4 AUC even when the overall AUC is at chance.",
            "",
            "| run | Q1 typical | Q2 | Q3 | Q4 atypical |",
            "|---|---|---|---|---|",
        ]
        for run in r["runs"]:
            qs = run["membership_inference"].get("mia_by_atypicality") or []
            if len(qs) == 4:
                flag = " 🚨" if qs[3]["attack_auc"] >= 0.6 else ""
                lines.append(f"| {run['run_id']}{flag} | "
                             + " | ".join(str(q["attack_auc"]) for q in qs) + " |")
        lines += ["", "## Attribute inference detail", "",
                  "| run | sensitive attribute | baseline | member acc | non-member acc | advantage |",
                  "|---|---|---|---|---|---|"]
        for run in r["runs"]:
            for a in run["attribute_inference"]:
                lines.append(f"| {run['run_id']} | {a['sensitive']} | {a['baseline_accuracy']} "
                             f"| {a['accuracy_members']} | {a['accuracy_nonmembers']} "
                             f"| {a['membership_advantage']:+.4f} |")
        return "\n".join(lines) + "\n"

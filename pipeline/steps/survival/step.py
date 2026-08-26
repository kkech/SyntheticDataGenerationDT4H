"""
Step: survival

Time-to-event fidelity -- the clinically decisive utility test for a
heart-failure cohort, where the canonical analyses are Kaplan-Meier
curves and effect estimates for death and rehospitalization.

Endpoint construction (documented assumption, applied identically to
real and synthetic data): the f5a columns record days to the first
event within a five-year follow-up. A recorded time is an event; a null
is administrative censoring at 1825 days. This is exactly the
missingness-carries-meaning semantics the sentinel design preserves.

Three comparisons per endpoint:
  * Kaplan-Meier curves for train, holdout, and every synthetic run
    (curve points are stored so the figures step can overlay them);
  * a log-rank test of each synthetic curve against the holdout curve
    (the holdout-vs-train p-value is the calibration: real samples
    differ by sampling noise too);
  * effect-estimate replication: the same multivariable model
    (Cox proportional hazards when `lifelines` is installed, otherwise
    a logistic model of 1-year mortality implemented natively) fitted
    on real and on synthetic data -- do the coefficient signs and
    magnitudes agree? "Research on the synthetic cohort reaches the
    same conclusions" is the strongest utility claim available.

Aggregate statistics and curve coordinates only -- safe to commit.
"""

import glob
import json
import os

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.common.alignment import align_categorical_case
from pipeline.steps.base import PipelineStep

FOLLOW_UP_DAYS = 1825

ENDPOINTS = {
    "all_cause_death": "cause_of_death_number_of_days_to_death_for_all_cause_f5a_first",
    "hf_rehospitalization": "encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first",
}

# Covariates for effect replication: core clinical predictors, chosen
# for low missingness and clinical face validity.
EFFECT_COVARIATES = (
    "patient_demographics_age",
    "patient_demographics_gender",
    "nyha_nyha_pET",
    "lab_results_valideGFR_value_first",
    "lab_results_sodium_value_first",
    "vital_signs_systolicBp_value_first",
)
ONE_YEAR = 365


def km_curve(time: np.ndarray, event: np.ndarray, grid_days: int = 30):
    """Kaplan-Meier estimate, evaluated on a fixed grid so curves from
    different frames are directly comparable."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    n = len(t)
    at_risk = n - np.arange(n)
    surv = 1.0
    times, probs = [0.0], [1.0]
    for i in range(n):
        if e[i]:
            surv *= 1.0 - 1.0 / at_risk[i]
            times.append(float(t[i]))
            probs.append(float(surv))
    grid = np.arange(0, FOLLOW_UP_DAYS + 1, grid_days, dtype=float)
    grid_probs = np.ones_like(grid)
    ti = np.asarray(times)
    pi = np.asarray(probs)
    for k, g in enumerate(grid):
        idx = np.searchsorted(ti, g, side="right") - 1
        grid_probs[k] = pi[max(idx, 0)]
    return grid.tolist(), np.round(grid_probs, 5).tolist()


def km_at(time: np.ndarray, event: np.ndarray, horizon: float):
    """Kaplan-Meier survival at a horizon with its Greenwood standard
    error. Row-wise accumulation: with d tied events at n at risk the
    per-row terms telescope to the grouped d/(n(n-d)), so ties are
    handled exactly."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    n = len(t)
    at_risk = n - np.arange(n)
    surv = 1.0
    var_sum = 0.0
    for i in range(n):
        if t[i] > horizon:
            break
        if e[i]:
            surv *= 1.0 - 1.0 / at_risk[i]
            if at_risk[i] > 1:
                var_sum += 1.0 / (at_risk[i] * (at_risk[i] - 1.0))
    return float(surv), float(surv * np.sqrt(var_sum))


# Equivalence margin for survival probabilities: a synthetic curve is
# declared EQUIVALENT to the holdout at a horizon when the 90% CI of the
# survival difference lies entirely within +/- this margin (TOST logic).
# Non-significance of a log-rank test is NOT evidence of equivalence --
# this is.
EQUIVALENCE_MARGIN = 0.05
EQUIVALENCE_HORIZONS = {"1y": 365.0, "3y": 1095.0, "5y": 1825.0}


def survival_equivalence(time_a, event_a, time_b, event_b) -> dict:
    """TOST-style equivalence of two KM curves at fixed horizons."""
    out = {"margin": EQUIVALENCE_MARGIN, "horizons": {}}
    for name, h in EQUIVALENCE_HORIZONS.items():
        sa, sea = km_at(time_a, event_a, h)
        sb, seb = km_at(time_b, event_b, h)
        diff = sa - sb
        se = float(np.sqrt(sea ** 2 + seb ** 2))
        lo, hi = diff - 1.645 * se, diff + 1.645 * se
        out["horizons"][name] = {
            "difference": round(diff, 4),
            "ci90": [round(lo, 4), round(hi, 4)],
            "equivalent": bool(-EQUIVALENCE_MARGIN < lo and hi < EQUIVALENCE_MARGIN),
        }
    out["equivalent_all_horizons"] = all(
        v["equivalent"] for v in out["horizons"].values())
    return out


def logrank(time_a, event_a, time_b, event_b) -> float:
    """Two-sample log-rank test p-value (chi-square, 1 df)."""
    from scipy import stats

    all_times = np.unique(np.concatenate([time_a[event_a == 1], time_b[event_b == 1]]))
    o_minus_e = 0.0
    var = 0.0
    for t in all_times:
        n_a = float((time_a >= t).sum())
        n_b = float((time_b >= t).sum())
        d_a = float(((time_a == t) & (event_a == 1)).sum())
        d_b = float(((time_b == t) & (event_b == 1)).sum())
        n = n_a + n_b
        d = d_a + d_b
        if n < 2 or d == 0:
            continue
        expected_a = d * n_a / n
        o_minus_e += d_a - expected_a
        var += d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)
    if var <= 0:
        return 1.0
    chi2 = o_minus_e ** 2 / var
    return float(stats.chi2.sf(chi2, df=1))


class SurvivalStep(PipelineStep):
    name = "survival"

    def run(self, config: PipelineConfig) -> None:
        import pandas as pd

        synthetic_files = sorted(
            glob.glob(os.path.join(config.step_dir("generate"), "DT4H_Synthetic_*.csv"))
        )
        if not synthetic_files:
            raise FileNotFoundError("No synthetic files -- run the generate step first.")

        train = self._load_decoded(config.train_output_path, config)
        holdout = self._load_decoded(config.holdout_output_path, config)

        out_dir = config.step_dir(self.name)
        os.makedirs(out_dir, exist_ok=True)

        results = {"follow_up_days": FOLLOW_UP_DAYS, "endpoints": {}, "effects": {}}

        for name, col in ENDPOINTS.items():
            if col not in train.columns:
                print(f"⚠️  Endpoint column {col} absent; skipping '{name}'.")
                continue
            print(f"\nEndpoint: {name} ({col})")
            entry = {"column": col, "curves": {}, "logrank_vs_holdout": {}}

            t_tr, e_tr = self._surv(train, col)
            t_ho, e_ho = self._surv(holdout, col)
            entry["curves"]["train"] = self._curve_record(t_tr, e_tr)
            entry["curves"]["holdout"] = self._curve_record(t_ho, e_ho)
            p_floor = logrank(t_tr, e_tr, t_ho, e_ho)
            entry["logrank_train_vs_holdout_p"] = round(p_floor, 4)
            # Real-vs-real equivalence calibrates what the margin means
            # at this sample size.
            entry["equivalence_train_vs_holdout"] = survival_equivalence(
                t_tr, e_tr, t_ho, e_ho)
            entry["equivalence_vs_holdout"] = {}
            print(f"  events: train {int(e_tr.sum())}/{len(e_tr)}, holdout {int(e_ho.sum())}/{len(e_ho)} | "
                  f"log-rank train-vs-holdout p={p_floor:.3f} (the sampling-noise calibration)")

            for path in synthetic_files:
                run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
                synth = pd.read_csv(path, low_memory=False)
                synth, _ = align_categorical_case(synth, train)
                if col not in synth.columns:
                    entry["logrank_vs_holdout"][run_id] = None
                    continue
                t_s, e_s = self._surv(synth, col)
                entry["curves"][run_id] = self._curve_record(t_s, e_s)
                p = logrank(t_s, e_s, t_ho, e_ho)
                entry["logrank_vs_holdout"][run_id] = round(p, 4)
                entry["equivalence_vs_holdout"][run_id] = survival_equivalence(
                    t_s, e_s, t_ho, e_ho)
            close = [rid for rid, p in entry["logrank_vs_holdout"].items()
                     if p is not None and p > 0.05]
            equiv = [rid for rid, q in entry["equivalence_vs_holdout"].items()
                     if q["equivalent_all_horizons"]]
            print(f"  runs indistinguishable from holdout at p>0.05: {len(close)}/"
                  f"{len(entry['logrank_vs_holdout'])}")
            print(f"  runs EQUIVALENT to holdout (TOST, ±{EQUIVALENCE_MARGIN:.0%} at "
                  f"1y/3y/5y): {len(equiv)}/{len(entry['equivalence_vs_holdout'])} "
                  f"-- equivalence is the claim non-significance cannot make")
            results["endpoints"][name] = entry

        print("\nEffect-estimate replication (1-year all-cause mortality)...")
        results["effects"] = self._effect_replication(train, holdout, synthetic_files, config)

        json_path = os.path.join(out_dir, "DT4H_Survival_Fidelity.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        md_path = os.path.join(out_dir, "DT4H_Survival_Fidelity.md")
        with open(md_path, "w") as f:
            f.write(self._render_markdown(results))
        print(f"Saved survival fidelity -> {json_path} / {md_path}")

    # --- endpoint construction ---

    @staticmethod
    def _surv(df, col):
        import pandas as pd

        days = pd.to_numeric(df[col], errors="coerce")
        event = days.notna().to_numpy().astype(int)
        time = days.fillna(FOLLOW_UP_DAYS).clip(lower=0, upper=FOLLOW_UP_DAYS).to_numpy(dtype=float)
        return time, event

    @staticmethod
    def _curve_record(time, event):
        grid, probs = km_curve(time, event)
        return {"n": int(len(time)), "events": int(event.sum()),
                "grid_days": grid, "survival": probs,
                "survival_1y": probs[min(len(probs) - 1, ONE_YEAR // 30)],
                "survival_5y": probs[-1]}

    # --- effect replication ---

    def _effect_frame(self, df):
        """Design matrix + 1-year mortality outcome; rows with a missing
        covariate are dropped (documented complete-case analysis)."""
        import pandas as pd

        col = ENDPOINTS["all_cause_death"]
        days = pd.to_numeric(df[col], errors="coerce")
        y = (days.notna() & (days <= ONE_YEAR)).astype(int)

        x = pd.DataFrame(index=df.index)
        for c in EFFECT_COVARIATES:
            if c not in df.columns:
                continue
            if c == "patient_demographics_gender":
                x["male"] = (df[c].astype("object").astype(str).str.lower() == "male").astype(float)
            else:
                x[c] = pd.to_numeric(df[c], errors="coerce")
        keep = x.notna().all(axis=1)
        return x[keep], y[keep], days[keep]

    def _fit_effects(self, df):
        """Standardized effect estimates. Cox (lifelines) if available,
        otherwise logistic regression on 1-year mortality via sklearn.
        Coefficients are per-SD for numeric covariates, so magnitudes
        are comparable across frames."""
        import pandas as pd

        x, y, days = self._effect_frame(df)
        if len(x) < 100 or y.sum() < 20 or y.sum() == len(y):
            return None
        # standardize numeric columns (per-SD effects)
        xs = x.copy()
        for c in xs.columns:
            sd = xs[c].std()
            if c != "male" and sd > 0:
                xs[c] = (xs[c] - xs[c].mean()) / sd
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            CoxPHFitter = None

        if CoxPHFitter is not None:
            # A degenerate synthetic frame can make the Cox fit diverge
            # (lifelines ConvergenceError, ill-conditioned matrices).
            # That is a property of THAT frame, not a reason to kill the
            # step: fall back to the logistic model for it.
            try:
                frame = xs.copy()
                frame["T"] = days.fillna(FOLLOW_UP_DAYS).clip(0, FOLLOW_UP_DAYS)
                frame["E"] = (pd.to_numeric(days, errors="coerce").notna()).astype(int)
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(frame, duration_col="T", event_col="E")
                return {"model": "cox_ph (lifelines)", "n": int(len(xs)),
                        "events": int(frame["E"].sum()),
                        "coefficients": {c: round(float(v), 4)
                                         for c, v in cph.params_.items()}}
            except Exception as e:
                print(f"  ⚠️  Cox fit did not converge on this frame "
                      f"({type(e).__name__}) -- falling back to the native "
                      f"logistic model for it.")

        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(xs, y)
            return {"model": "logistic_1y_mortality (native fallback)", "n": int(len(xs)),
                    "events": int(y.sum()),
                    "coefficients": {c: round(float(b), 4)
                                     for c, b in zip(xs.columns, clf.coef_[0])}}
        except Exception as e:
            # Both estimators failed on this frame: report it as not
            # estimable (the existing degenerate-frame path) rather than
            # aborting the whole survival step.
            print(f"  ⚠️  Effects not estimable on this frame "
                  f"({type(e).__name__}: {e}).")
            return None

    def _effect_replication(self, train, holdout, synthetic_files, config):
        import pandas as pd

        real_fit = self._fit_effects(train)
        if real_fit is None:
            print("  ⚠️  Not enough labelled data for effect replication; skipped.")
            return {"note": "insufficient data"}
        hold_fit = self._fit_effects(holdout)
        out = {"real_train": real_fit, "real_holdout": hold_fit, "synthetic": {}}
        print(f"  model: {real_fit['model']} | train coefficients: {real_fit['coefficients']}")

        real_coef = real_fit["coefficients"]
        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            synth, _ = align_categorical_case(synth, train)
            fit = self._fit_effects(synth) if ENDPOINTS["all_cause_death"] in synth.columns else None
            if fit is None:
                out["synthetic"][run_id] = {"note": "not estimable"}
                continue
            signs = [c for c in real_coef
                     if c in fit["coefficients"]
                     and np.sign(fit["coefficients"][c]) == np.sign(real_coef[c])
                     and abs(real_coef[c]) > 0.02]
            comparable = [c for c in real_coef if abs(real_coef[c]) > 0.02]
            fit["sign_agreement"] = f"{len(signs)}/{len(comparable)}"
            fit["mean_abs_coef_error"] = round(float(np.mean(
                [abs(fit["coefficients"].get(c, 0.0) - v) for c, v in real_coef.items()])), 4)
            out["synthetic"][run_id] = fit
        return out

    # --- misc ---

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
        lines = ["# Survival Fidelity", "",
                 "Endpoints use the five-year follow-up columns: a recorded days-to-event is an "
                 "event, a null is administrative censoring at 1825 days -- the same rule for real "
                 "and synthetic data. The train-vs-holdout log-rank p-value calibrates what pure "
                 "sampling noise looks like.", ""]
        for name, e in r.get("endpoints", {}).items():
            lines += [f"## {name}", "",
                      f"train events {e['curves']['train']['events']}/{e['curves']['train']['n']} | "
                      f"holdout {e['curves']['holdout']['events']}/{e['curves']['holdout']['n']} | "
                      f"log-rank train-vs-holdout p = {e['logrank_train_vs_holdout_p']}", "",
                      "| run | 1y survival | 5y survival | log-rank vs holdout (p) | equivalent (TOST ±5pp, 1y/3y/5y) |",
                      "|---|---|---|---|---|"]
            tv_eq = (e.get("equivalence_train_vs_holdout") or {}).get(
                "equivalent_all_horizons")
            for rid, curve in e["curves"].items():
                if rid in ("train", "holdout"):
                    eq_cell = ("yes ✅" if tv_eq else "no") if rid == "train" else ""
                    lines.append(f"| **{rid}** | {curve['survival_1y']} | {curve['survival_5y']} | "
                                 f"{'-' if rid == 'train' else ''} | {eq_cell} |")
            for rid, p in e["logrank_vs_holdout"].items():
                c = e["curves"].get(rid, {})
                q = (e.get("equivalence_vs_holdout") or {}).get(rid) or {}
                eq_cell = ("yes ✅" if q.get("equivalent_all_horizons")
                           else "no" if q else "-")
                lines.append(f"| {rid} | {c.get('survival_1y', '-')} | {c.get('survival_5y', '-')} "
                             f"| {p} | {eq_cell} |")
            lines.append("")
            lines.append("Equivalence is a POSITIVE claim (90% CI of the survival "
                         "difference within ±5pp at every horizon) -- unlike a "
                         "non-significant log-rank, which is only absence of evidence.")
            lines.append("")
        eff = r.get("effects", {})
        if eff.get("real_train"):
            lines += ["## Effect-estimate replication", "",
                      f"Model: {eff['real_train']['model']}, standardized (per-SD) coefficients.",
                      "", "| frame | n | events | sign agreement | mean |coef error| |",
                      "|---|---|---|---|---|",
                      f"| real train | {eff['real_train']['n']} | {eff['real_train']['events']} | - | - |"]
            if eff.get("real_holdout"):
                h = eff["real_holdout"]
                lines.append(f"| real holdout | {h['n']} | {h['events']} | - | - |")
            for rid, fit in eff.get("synthetic", {}).items():
                if fit.get("note"):
                    lines.append(f"| {rid} | - | - | {fit['note']} | - |")
                else:
                    lines.append(f"| {rid} | {fit['n']} | {fit['events']} | {fit['sign_agreement']} "
                                 f"| {fit['mean_abs_coef_error']} |")
        return "\n".join(lines) + "\n"

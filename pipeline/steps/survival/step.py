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
A recorded time BEYOND 1825 days is out of horizon: the patient is
known event-free through the follow-up window, so it is treated as
CENSORED at 1825 (not an event at the boundary); the count is reported
per frame as "times_beyond_horizon".

Known, unrepairable asymmetry (disclosed here and in the JSON as
"decode_note"): the generate step's sentinel decode nulls any synthetic
numeric value below the column's real observed minimum. For a
days-to-event column that erases synthetic early events shorter than
the shortest real event and this step then reads them as censoring at
1825 days. The original synthetic value is gone from the released CSV,
so the effect cannot be undone downstream -- synthetic survival can be
biased upward by exactly those erased early events.

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


def _km_order(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Stable sort order: time ascending, EVENTS BEFORE CENSORINGS at
    equal times (the KM convention -- a record censored at t is still at
    risk for an event at t). np.argsort's default quicksort is unstable
    and event/censoring order at ties was previously arbitrary, which
    matters at the 1825-day horizon where censorings pile up."""
    return np.lexsort((1 - np.asarray(event), np.asarray(time)))


def km_curve(time: np.ndarray, event: np.ndarray, grid_days: int = 30):
    """Kaplan-Meier estimate, evaluated on a fixed grid so curves from
    different frames are directly comparable."""
    order = _km_order(time, event)
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
    order = _km_order(time, event)
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

        results = {
            "follow_up_days": FOLLOW_UP_DAYS,
            "decode_note": (
                "Synthetic days-to-event values below the real observed minimum "
                "were nulled by the generate step's sentinel decode and are read "
                "here as censoring at 1825 days; erased synthetic early events "
                "cannot be recovered from the released CSVs, so synthetic "
                "survival may be biased upward by exactly those events."),
            "endpoints": {},
            "effects": {},
        }

        for name, col in ENDPOINTS.items():
            if col not in train.columns:
                print(f"⚠️  Endpoint column {col} absent; skipping '{name}'.")
                continue
            print(f"\nEndpoint: {name} ({col})")
            entry = {"column": col, "curves": {}, "logrank_vs_holdout": {},
                     "times_beyond_horizon": {}}

            t_tr, e_tr, b_tr = self._surv(train, col)
            t_ho, e_ho, b_ho = self._surv(holdout, col)
            entry["times_beyond_horizon"]["train"] = b_tr
            entry["times_beyond_horizon"]["holdout"] = b_ho
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
                t_s, e_s, b_s = self._surv(synth, col)
                entry["times_beyond_horizon"][run_id] = b_s
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
        """(time, event, times_beyond_horizon). A recorded time beyond
        the 1825-day follow-up is out of horizon: the patient is known
        event-free through the window, so it is CENSORED at 1825, not an
        event at the boundary (the coherence audit flags the same values
        as violations -- an event at exactly 1825 would double-count
        them as legitimate deaths)."""
        import pandas as pd

        days = pd.to_numeric(df[col], errors="coerce")
        beyond = days > FOLLOW_UP_DAYS
        event = (days.notna() & ~beyond).to_numpy().astype(int)
        time = days.fillna(FOLLOW_UP_DAYS).clip(lower=0, upper=FOLLOW_UP_DAYS).to_numpy(dtype=float)
        return time, event, int(beyond.sum())

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
        covariate are dropped (documented complete-case analysis).
        Gender maps male->1, female->0 and ANYTHING else (NaN, 'Missing',
        unexpected categories) to NaN, so missing gender is dropped by
        the complete-case filter instead of being silently coded female.
        Returns (x, y, days, n_dropped_missing_gender)."""
        import pandas as pd

        col = ENDPOINTS["all_cause_death"]
        days = pd.to_numeric(df[col], errors="coerce")
        y = (days.notna() & (days <= ONE_YEAR)).astype(int)

        x = pd.DataFrame(index=df.index)
        for c in EFFECT_COVARIATES:
            if c not in df.columns:
                continue
            if c == "patient_demographics_gender":
                g = df[c].astype("object").where(df[c].notna(), "missing").astype(str).str.lower()
                x["male"] = g.map({"male": 1.0, "female": 0.0})
            else:
                x[c] = pd.to_numeric(df[c], errors="coerce")
        dropped_gender = int(x["male"].isna().sum()) if "male" in x.columns else 0
        keep = x.notna().all(axis=1)
        return x[keep], y[keep], days[keep], dropped_gender

    def _fit_effects(self, df, scaler: dict | None = None):
        """Standardized effect estimates. Cox (lifelines) if available,
        otherwise logistic regression on 1-year mortality via sklearn.
        Numeric covariates are standardized by the REAL TRAIN split's
        mean/SD (`scaler`, from _train_scaler) in EVERY frame, so a
        synthetic frame with the wrong scale shows up as a coefficient
        discrepancy instead of being silently re-normalized away; only
        when no scaler is given (calibration/tests) does the frame
        standardize by its own moments."""
        import pandas as pd

        x, y, days, dropped_gender = self._effect_frame(df)
        if len(x) < 100 or y.sum() < 20 or y.sum() == len(y):
            return None
        # standardize numeric columns (per-SD effects on the REAL TRAIN scale)
        xs = x.copy()
        for c in xs.columns:
            if c == "male":
                continue
            mean, sd = (scaler.get(c, (xs[c].mean(), xs[c].std())) if scaler is not None
                        else (xs[c].mean(), xs[c].std()))
            if sd and sd > 0:
                xs[c] = (xs[c] - mean) / sd
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
                        "dropped_missing_gender": dropped_gender,
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
                    "dropped_missing_gender": dropped_gender,
                    "coefficients": {c: round(float(b), 4)
                                     for c, b in zip(xs.columns, clf.coef_[0])}}
        except Exception as e:
            # Both estimators failed on this frame: report it as not
            # estimable (the existing degenerate-frame path) rather than
            # aborting the whole survival step.
            print(f"  ⚠️  Effects not estimable on this frame "
                  f"({type(e).__name__}: {e}).")
            return None

    def _train_scaler(self, train) -> dict:
        """{covariate: (mean, sd)} from the REAL TRAIN effect frame, so
        every frame is standardized on the same scale."""
        x, _, _, _ = self._effect_frame(train)
        return {c: (float(x[c].mean()), float(x[c].std()))
                for c in x.columns if c != "male"}

    # Fewer than this many shared covariates between the real and a
    # synthetic fit and the coefficient comparison is not meaningful.
    MIN_MATCHED_COEFFICIENTS = 4

    def _effect_replication(self, train, holdout, synthetic_files, config):
        import pandas as pd

        scaler = self._train_scaler(train)
        real_fit = self._fit_effects(train, scaler)
        if real_fit is None:
            print("  ⚠️  Not enough labelled data for effect replication; skipped.")
            return {"note": "insufficient data"}
        hold_fit = self._fit_effects(holdout, scaler)
        out = {"real_train": real_fit, "real_holdout": hold_fit, "synthetic": {},
               "standardization": "all frames standardized by the real TRAIN "
                                  "split's covariate mean/SD"}
        print(f"  model: {real_fit['model']} | train coefficients: {real_fit['coefficients']}")

        real_coef = real_fit["coefficients"]
        for path in synthetic_files:
            run_id = os.path.basename(path)[len("DT4H_Synthetic_"):-len(".csv")]
            synth = pd.read_csv(path, low_memory=False)
            synth, _ = align_categorical_case(synth, train)
            fit = (self._fit_effects(synth, scaler)
                   if ENDPOINTS["all_cause_death"] in synth.columns else None)
            if fit is None:
                out["synthetic"][run_id] = {"note": "not estimable"}
                continue
            signs = [c for c in real_coef
                     if c in fit["coefficients"]
                     and np.sign(fit["coefficients"][c]) == np.sign(real_coef[c])
                     and abs(real_coef[c]) > 0.02]
            comparable = [c for c in real_coef if abs(real_coef[c]) > 0.02]
            fit["sign_agreement"] = f"{len(signs)}/{len(comparable)}"
            # Error only over covariates present in BOTH fits: substituting
            # 0.0 for an absent covariate rewarded degenerate fits that
            # dropped columns.
            matched = [c for c in real_coef if c in fit["coefficients"]]
            fit["coefficients_matched"] = f"{len(matched)}/{len(real_coef)}"
            if len(matched) < self.MIN_MATCHED_COEFFICIENTS:
                fit["mean_abs_coef_error"] = None
                fit["note"] = (f"not comparable ({len(matched)}/{len(real_coef)} "
                               "covariates shared with the real fit)")
            else:
                fit["mean_abs_coef_error"] = round(float(np.mean(
                    [abs(fit["coefficients"][c] - real_coef[c]) for c in matched])), 4)
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
                 "and synthetic data. A recorded time beyond 1825 days is treated as censoring at "
                 "1825 (out of horizon), counted per frame as `times_beyond_horizon`. The "
                 "train-vs-holdout log-rank p-value calibrates what pure sampling noise looks like.",
                 "",
                 "Disclosure: synthetic days-to-event values below the real observed minimum were "
                 "nulled by the sentinel decode upstream and are read here as censoring; those "
                 "erased early events cannot be recovered from the released CSVs (see `decode_note` "
                 "in the JSON).",
                 "",
                 "Effect-replication covariates are standardized in EVERY frame (real train, real "
                 "holdout, synthetic) by the REAL TRAIN split's mean/SD, so scale infidelity in a "
                 "synthetic frame shows up as a coefficient discrepancy instead of being "
                 "re-normalized away.", ""]
        for name, e in r.get("endpoints", {}).items():
            beyond = e.get("times_beyond_horizon") or {}
            beyond_note = ""
            if any(beyond.values()):
                worst = {k: v for k, v in beyond.items() if v}
                beyond_note = (f" | times beyond 1825d censored at horizon: {worst}")
            lines += [f"## {name}", "",
                      f"train events {e['curves']['train']['events']}/{e['curves']['train']['n']} | "
                      f"holdout {e['curves']['holdout']['events']}/{e['curves']['holdout']['n']} | "
                      f"log-rank train-vs-holdout p = {e['logrank_train_vs_holdout_p']}"
                      f"{beyond_note}", "",
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
                      f"Model: {eff['real_train']['model']}, per-SD coefficients standardized by "
                      "the real TRAIN split's mean/SD in every frame. The coefficient error is "
                      "computed only over covariates present in BOTH the real and the synthetic "
                      "fit (`matched`); runs sharing fewer than "
                      f"{SurvivalStep.MIN_MATCHED_COEFFICIENTS} covariates are not comparable.",
                      "", "| frame | n | events | sign agreement | coef matched | mean |coef error| |",
                      "|---|---|---|---|---|---|",
                      f"| real train | {eff['real_train']['n']} | {eff['real_train']['events']} | - | - | - |"]
            if eff.get("real_holdout"):
                h = eff["real_holdout"]
                lines.append(f"| real holdout | {h['n']} | {h['events']} | - | - | - |")
            for rid, fit in eff.get("synthetic", {}).items():
                if fit.get("mean_abs_coef_error") is None:
                    lines.append(f"| {rid} | {fit.get('n', '-')} | {fit.get('events', '-')} "
                                 f"| {fit.get('sign_agreement', '-')} "
                                 f"| {fit.get('coefficients_matched', '-')} "
                                 f"| {fit.get('note', 'not estimable')} |")
                else:
                    lines.append(f"| {rid} | {fit['n']} | {fit['events']} | {fit['sign_agreement']} "
                                 f"| {fit['coefficients_matched']} | {fit['mean_abs_coef_error']} |")
        return "\n".join(lines) + "\n"

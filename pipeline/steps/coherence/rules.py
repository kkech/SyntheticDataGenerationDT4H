"""
Row-coherence rules: is a synthetic row a clinically coherent patient,
or just a bundle of individually well-distributed values?

Three rule families, all grounded in the REAL data rather than hand
authored where possible:

  * MINED boolean implications (a=true => b=true): discovered on the
    TRAIN split among all true/false columns, kept only when well
    supported and essentially never violated by real patients. This
    automatically recovers medication=>medication-history hierarchies,
    death-window monotonicity (died within 7 days => died within 1
    month), and cause-specific => all-cause death implications.
  * LEARNED category-range consistency: for (categorical, numeric)
    pairs that encode the same measurement (CKD severity vs eGFR,
    hyperkalemia severity vs potassium), the observed numeric range per
    category is learned from the train split (with a tolerance margin);
    a synthetic row whose category contradicts its own numeric value
    violates the rule.
  * EXPLICIT survival-logic checks: a recorded days-to-death implies
    the corresponding window flags; flags bound the recorded time;
    times respect the 5-year follow-up horizon.

Every rule is also evaluated on the HOLDOUT split: real unseen patients
are the fair baseline for how often "real clinical data" violates its
own logic. The mined rule set is committed as JSON so it is auditable.
"""

import numpy as np
import pandas as pd

MIN_SUPPORT = 50          # a=true must occur at least this often in train
MAX_TRAIN_VIOLATION = 0.002   # rule must hold on >=99.8% of supporting train rows
RANGE_MARGIN_FRACTION = 0.05  # tolerance around learned per-category ranges

# (categorical column, numeric column) pairs encoding the same measurement.
CATEGORY_RANGE_PAIRS = (
    ("ckd_severity_from_calculated_egfr", "lab_results_valideGFR_value_first"),
    ("ckd_severity_from_calculated_egfr", "lab_results_valideGFR_value_last"),
    ("hyperkalemia_severity_categorizedValue", "lab_results_potassium_value_first"),
    ("hyperkalemia_severity_categorizedValue", "lab_results_potassium_value_last"),
)

FOLLOW_UP_DAYS = 1825  # f5a = five-year horizon
WINDOW_DAYS = {"w7d": 7, "w1mo": 30, "w3mo": 91, "w6mo": 182,
               "w1a": 365, "w3a": 1095, "w5a": FOLLOW_UP_DAYS}
DAYS_TO_FLAG_BASE = {
    "cause_of_death_number_of_days_to_death_for_all_cause_f5a_first": "cause_of_death_isAllCause_f5a_{w}_first",
    "cause_of_death_number_of_days_to_death_for_CV_f5a_first": "cause_of_death_isCV_f5a_{w}_first",
    "cause_of_death_number_of_days_to_death_for_renal_f5a_first": "cause_of_death_isRenal_f5a_{w}_first",
}


def _bool_matrix(df: pd.DataFrame):
    """True/false columns as boolean masks. Returns (columns, T, F) where
    T[i,j]=column j is 'true' in row i, F likewise for 'false' (a row can
    be neither when the value is Missing)."""
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        vals = set(s.astype("object").where(s.notna(), "Missing").astype(str).str.lower().unique())
        if vals <= {"true", "false", "missing"} and ("true" in vals or "false" in vals):
            cols.append(c)
    if not cols:
        return [], np.zeros((len(df), 0), bool), np.zeros((len(df), 0), bool)
    low = df[cols].astype("object").where(df[cols].notna(), "missing").astype(str)
    low = low.apply(lambda s: s.str.lower())
    return cols, (low == "true").to_numpy(), (low == "false").to_numpy()


def mine_boolean_implications(train: pd.DataFrame) -> list[dict]:
    """Implications a=true => b=true holding on the train split."""
    cols, T, F = _bool_matrix(train)
    rules = []
    if not cols:
        return rules
    support = T.sum(axis=0)
    # violations[i, j] = count(a_i true AND b_j explicitly false)
    violations = T.astype(np.int64).T @ F.astype(np.int64)
    for i, a in enumerate(cols):
        if support[i] < MIN_SUPPORT:
            continue
        for j, b in enumerate(cols):
            if i == j:
                continue
            # skip trivial rules: b true almost everywhere anyway
            if T[:, j].mean() > 0.95:
                continue
            rate = violations[i, j] / support[i]
            if rate <= MAX_TRAIN_VIOLATION:
                rules.append({
                    "type": "implication",
                    "if_true": a,
                    "then_true": b,
                    "train_support": int(support[i]),
                    "train_violation_rate": round(float(rate), 5),
                })
    return rules


def learn_category_ranges(train: pd.DataFrame) -> list[dict]:
    rules = []
    for cat_col, num_col in CATEGORY_RANGE_PAIRS:
        if cat_col not in train.columns or num_col not in train.columns:
            continue
        num = pd.to_numeric(train[num_col], errors="coerce")
        full_range = float(num.max() - num.min()) if num.notna().any() else 0.0
        margin = RANGE_MARGIN_FRACTION * full_range
        ranges = {}
        for cat, grp in num.groupby(train[cat_col].astype("object").where(train[cat_col].notna(), "Missing").astype(str)):
            observed = grp.dropna()
            if len(observed) < 20 or cat == "Missing":
                continue
            ranges[cat] = [round(float(observed.min() - margin), 4),
                           round(float(observed.max() + margin), 4)]
        if ranges:
            rules.append({"type": "category_range", "categorical": cat_col,
                          "numeric": num_col, "ranges": ranges,
                          "margin": round(margin, 4)})
    return rules


def survival_logic_rules(train: pd.DataFrame) -> list[dict]:
    rules = []
    for days_col, flag_tpl in DAYS_TO_FLAG_BASE.items():
        if days_col not in train.columns:
            continue
        rules.append({"type": "days_bounds", "days": days_col,
                      "max_days": FOLLOW_UP_DAYS})
        for w, wdays in WINDOW_DAYS.items():
            flag = flag_tpl.format(w=w)
            if flag in train.columns:
                rules.append({"type": "flag_days_consistency", "days": days_col,
                              "flag": flag, "window_days": wdays})
    return rules


def build_rules(train: pd.DataFrame) -> list[dict]:
    return (mine_boolean_implications(train)
            + learn_category_ranges(train)
            + survival_logic_rules(train))


def _lower(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), "missing").astype(str).str.lower()


def evaluate_rules(df: pd.DataFrame, rules: list[dict]) -> list[dict]:
    """Violation counts per rule on one frame; a rule only counts rows
    where its precondition applies and the consequent is decidable."""
    out = []
    for r in rules:
        entry = {k: r[k] for k in r if k != "ranges"}
        if r["type"] == "implication":
            if r["if_true"] not in df.columns or r["then_true"] not in df.columns:
                entry.update({"applicable": 0, "violations": 0})
            else:
                a = _lower(df[r["if_true"]]) == "true"
                b = _lower(df[r["then_true"]])
                applicable = a & (b != "missing")
                entry.update({"applicable": int(applicable.sum()),
                              "violations": int((applicable & (b == "false")).sum())})
        elif r["type"] == "category_range":
            if r["categorical"] not in df.columns or r["numeric"] not in df.columns:
                entry.update({"applicable": 0, "violations": 0})
            else:
                cat = df[r["categorical"]].astype("object").where(df[r["categorical"]].notna(), "Missing").astype(str)
                num = pd.to_numeric(df[r["numeric"]], errors="coerce")
                applicable = 0
                violations = 0
                for c, (lo, hi) in r["ranges"].items():
                    mask = (cat == c) & num.notna()
                    applicable += int(mask.sum())
                    violations += int(((num < lo) | (num > hi))[mask].sum())
                entry.update({"applicable": applicable, "violations": violations})
        elif r["type"] == "days_bounds":
            if r["days"] not in df.columns:
                entry.update({"applicable": 0, "violations": 0})
            else:
                days = pd.to_numeric(df[r["days"]], errors="coerce")
                applicable = days.notna()
                entry.update({"applicable": int(applicable.sum()),
                              "violations": int(((days < 0) | (days > r["max_days"]))[applicable].sum())})
        elif r["type"] == "flag_days_consistency":
            if r["days"] not in df.columns or r["flag"] not in df.columns:
                entry.update({"applicable": 0, "violations": 0})
            else:
                days = pd.to_numeric(df[r["days"]], errors="coerce")
                flag = _lower(df[r["flag"]])
                # flag=true => a death time exists and falls within the window
                applicable = flag == "true"
                bad = applicable & (days.isna() | (days > r["window_days"]))
                # and a recorded time within the window => flag must be true
                applicable2 = days.notna() & (days <= r["window_days"]) & (flag != "missing")
                bad2 = applicable2 & (flag == "false")
                entry.update({"applicable": int(applicable.sum() + applicable2.sum()),
                              "violations": int(bad.sum() + bad2.sum())})
        rate = entry["violations"] / entry["applicable"] if entry["applicable"] else None
        entry["violation_rate"] = round(rate, 5) if rate is not None else None
        out.append(entry)
    return out


def summarize_rule_results(results: list[dict]) -> dict:
    applicable = sum(r["applicable"] for r in results)
    violations = sum(r["violations"] for r in results)
    return {
        "rules": len(results),
        "rule_checks_applicable": int(applicable),
        "violations": int(violations),
        "overall_violation_rate": round(violations / applicable, 5) if applicable else None,
        "rules_violated": sum(1 for r in results if r["violations"] > 0),
    }


def row_violation_mask(df: pd.DataFrame, rules: list[dict]) -> pd.Series:
    """Boolean Series over df.index: True where the row violates at
    least one rule. Mirrors evaluate_rules() rule-for-rule -- the total
    True count can be LOWER than evaluate_rules' violation sum because a
    row violating several rules is flagged once."""
    bad = pd.Series(False, index=df.index)
    for r in rules:
        if r["type"] == "implication":
            if r["if_true"] not in df.columns or r["then_true"] not in df.columns:
                continue
            a = _lower(df[r["if_true"]]) == "true"
            b = _lower(df[r["then_true"]])
            bad |= a & (b == "false")
        elif r["type"] == "category_range":
            if r["categorical"] not in df.columns or r["numeric"] not in df.columns:
                continue
            cat = df[r["categorical"]].astype("object").where(
                df[r["categorical"]].notna(), "Missing").astype(str)
            num = pd.to_numeric(df[r["numeric"]], errors="coerce")
            for c, (lo, hi) in r["ranges"].items():
                mask = (cat == c) & num.notna()
                bad |= mask & ((num < lo) | (num > hi))
        elif r["type"] == "days_bounds":
            if r["days"] not in df.columns:
                continue
            days = pd.to_numeric(df[r["days"]], errors="coerce")
            bad |= days.notna() & ((days < 0) | (days > r["max_days"]))
        elif r["type"] == "flag_days_consistency":
            if r["days"] not in df.columns or r["flag"] not in df.columns:
                continue
            days = pd.to_numeric(df[r["days"]], errors="coerce")
            flag = _lower(df[r["flag"]])
            bad |= (flag == "true") & (days.isna() | (days > r["window_days"]))
            bad |= (days.notna() & (days <= r["window_days"])
                    & (flag != "missing") & (flag == "false"))
    return bad

"""
Classifier two-sample test (C2ST): the single-number FULL-JOINT fidelity
measure. A gradient-boosting classifier is trained to distinguish real
rows from synthetic rows; its out-of-fold AUC says how distinguishable
the joints are. 0.5 = indistinguishable, 1.0 = trivially separable.

Calibration: the same test run on train-vs-holdout (two real samples)
should sit at ~0.5; anything above it there is split artifact, and the
synthetic AUCs are read against that floor.

Two design points keep the number honest:

  * only the INTERSECTION of columns present in both frames is used.
    NaN-padding columns the other frame lacks hands the classifier the
    schema itself as a feature -- a width-limited synthetic file scores
    AUC 1.0 on coverage alone, saying nothing about the joint it DID
    model. Schema coverage is reported alongside the AUC instead, so a
    narrow file is visibly narrow rather than falsely separable.
  * the AUC is out-of-fold over stratified K-fold CV rather than one
    70/30 split, with the per-fold sd reported, so a lucky (or unlucky)
    split cannot move the headline number.
"""

import statistics

import numpy as np
import pandas as pd


def _encode_for_c2st(df: pd.DataFrame, columns: list[str], categories: dict) -> pd.DataFrame:
    # Built as a dict and materialized once -- inserting ~200 columns
    # one at a time fragments the frame. `columns` is always the
    # intersection of both frames (see c2st_auc), so every column
    # exists here.
    out = {}
    for c in columns:
        if c in categories:
            s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
            codes = categories[c].get_indexer(s).astype(float)
            codes[codes < 0] = -1.0  # unseen category: a real, learnable signal
            out[c] = codes
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)


def c2st_auc(real: pd.DataFrame, other: pd.DataFrame, columns: list[str],
             seed: int = 0, n_folds: int = 5) -> dict:
    """C2ST over the columns present in BOTH frames. Categories are
    fitted on the real side.

    Returns a dict:
      auc              pooled out-of-fold AUC (None if nothing usable)
      auc_sd           sd of the per-fold AUCs -- the split noise
      fold_aucs        the per-fold AUCs themselves
      n_folds          folds actually used
      columns_used     how many of the requested columns both frames have
      schema_coverage  columns_used / len(columns) -- a width-limited
                       file shows up here, not as fake separability
      n_real, n_other  class sizes (the caller matches these to the
                       floor's sizes; see the evaluate step)
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    used = [c for c in columns if c in real.columns and c in other.columns]
    result = {
        "auc": None,
        "auc_sd": None,
        "fold_aucs": [],
        "n_folds": n_folds,
        "columns_used": len(used),
        "schema_coverage": round(len(used) / len(columns), 4) if columns else None,
        "n_real": int(len(real)),
        "n_other": int(len(other)),
    }
    if not used:
        return result

    categories = {}
    for c in used:
        if not (pd.api.types.is_numeric_dtype(real[c])
                and not pd.api.types.is_bool_dtype(real[c])):
            s = real[c].astype("object").where(real[c].notna(), "Missing").astype(str)
            categories[c] = pd.Index(sorted(s.unique()))

    xr = _encode_for_c2st(real, used, categories)
    xo = _encode_for_c2st(other, used, categories)
    x = pd.concat([xr, xo], ignore_index=True)
    y = np.concatenate([np.zeros(len(xr)), np.ones(len(xo))])

    usable = [c for c in x.columns if x[c].nunique(dropna=True) >= 2]
    if not usable:
        return result  # every shared column is constant: nothing to learn from

    oof = np.full(len(y), np.nan)
    fold_aucs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(x, y):
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(x.iloc[tr_idx][usable], y[tr_idx])
        p = clf.predict_proba(x.iloc[te_idx][usable])[:, 1]
        oof[te_idx] = p
        fold_aucs.append(float(roc_auc_score(y[te_idx], p)))

    result["auc"] = round(float(roc_auc_score(y, oof)), 4)
    result["auc_sd"] = round(statistics.stdev(fold_aucs), 4) if len(fold_aucs) > 1 else None
    result["fold_aucs"] = [round(a, 4) for a in fold_aucs]
    return result

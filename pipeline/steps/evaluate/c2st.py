"""
Classifier two-sample test (C2ST): the single-number FULL-JOINT fidelity
measure. A gradient-boosting classifier is trained to distinguish real
rows from synthetic rows; its held-out AUC says how distinguishable the
joints are. 0.5 = indistinguishable, 1.0 = trivially separable.

Calibration: the same test run on train-vs-holdout (two real samples)
should sit at ~0.5; anything above it there is split artifact, and the
synthetic AUCs are read against that floor.
"""

import numpy as np
import pandas as pd


def _encode_for_c2st(df: pd.DataFrame, columns: list[str], categories: dict) -> pd.DataFrame:
    # Built as a dict and materialized once -- inserting ~200 columns
    # one at a time fragments the frame.
    out = {}
    for c in columns:
        if c not in df.columns:
            out[c] = np.full(len(df), np.nan)
        elif c in categories:
            s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
            codes = categories[c].get_indexer(s).astype(float)
            codes[codes < 0] = -1.0  # unseen category: a real, learnable signal
            out[c] = codes
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)


def c2st_auc(real: pd.DataFrame, other: pd.DataFrame, columns: list[str],
             seed: int = 0, test_fraction: float = 0.3) -> float:
    """AUC of a classifier separating `real` rows from `other` rows over
    the given columns. Categories are fitted on the real side."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    categories = {}
    for c in columns:
        if c in real.columns and not (pd.api.types.is_numeric_dtype(real[c])
                                      and not pd.api.types.is_bool_dtype(real[c])):
            s = real[c].astype("object").where(real[c].notna(), "Missing").astype(str)
            categories[c] = pd.Index(sorted(s.unique()))

    xr = _encode_for_c2st(real, columns, categories)
    xo = _encode_for_c2st(other, columns, categories)
    x = pd.concat([xr, xo], ignore_index=True)
    y = np.concatenate([np.zeros(len(xr)), np.ones(len(xo))])

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_fraction, random_state=seed, stratify=y)
    usable = [c for c in x_tr.columns if x_tr[c].nunique(dropna=True) >= 2]
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(x_tr[usable], y_tr)
    return round(float(roc_auc_score(y_te, clf.predict_proba(x_te[usable])[:, 1])), 4)

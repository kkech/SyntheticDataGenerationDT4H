"""
Shared clinical-outcome target selection for the TSTR utility step and
the AIM column-importance selection (which must include the TSTR targets
so the reduced-width AIM output stays utility-evaluable).

Selection is diversified by outcome FAMILY: the metadata declares many
windowed variants of the same endpoint (w7d/w1mo/w6mo/w1a/w3a), and
five variants of one endpoint are not five independent pieces of
evidence. One best-balanced variant is taken per family, and a
mortality family is force-included when any of its variants is
eligible, because all-cause death is the clinically central endpoint
of a heart-failure cohort.

Targets are always selected on the TRAIN split, so nothing about the
holdout influences any modelling choice.
"""

import json
import re

MIN_CLASS_TRAIN = 10   # each class must appear at least this often in training data
MIN_CLASS_TEST = 10    # and this often in the holdout used for testing
MORTALITY_PATTERN = re.compile(r"death|mortal", re.IGNORECASE)
# Time-window / form tokens that distinguish variants of the same
# endpoint: _f5a, any _w<number><unit> window (w7d, w1mo, w6mo, w3mo,
# w1a, w3a, w5a, ...), and the _first/_last aggregation suffixes.
_FAMILY_TOKENS = re.compile(r"(_f5a|_w\d+[a-z]+|_first|_last)")


def to_binary(series):
    """Boolean-ish column -> 0/1 with Missing rows dropped; None if it
    is not a two-class true/false column."""
    s = series.astype("object").where(series.notna(), "missing").astype(str).str.lower()
    s = s[s != "missing"]
    values = set(s.unique())
    if not values or not values <= {"true", "false"}:
        return None
    return (s == "true").astype(int)


def declared_outcomes(metadata_path: str) -> set[str]:
    with open(metadata_path) as f:
        raw = json.load(f)
    return {v["name"] for v in raw["entries"][0]["outcomes"]}


def family_of(column: str) -> str:
    return _FAMILY_TOKENS.sub("", column)


def select_targets(train, outcome_cols: set[str], max_targets: int,
                   explicit: tuple | None = None) -> list[str]:
    """Eligible, family-diversified targets ordered by class balance.

    `explicit` (config.utility_targets) overrides the automatic choice
    but is still validated: absent or non-binary columns are skipped
    with a warning rather than crashing a long run, and the same
    minimum-class-count check as the automatic selection applies (an
    explicitly requested target with 3 positives is no more evaluable
    than an automatically found one).
    """
    if explicit:
        picked = []
        for t in explicit:
            y = to_binary(train[t]) if t in train.columns else None
            if y is None:
                print(f"⚠️  Requested utility target '{t}' is absent or not a "
                      "two-class boolean column; skipped.")
                continue
            if min(int(y.sum()), int((1 - y).sum())) < MIN_CLASS_TRAIN:
                print(f"⚠️  Requested utility target '{t}' has fewer than "
                      f"{MIN_CLASS_TRAIN} training records of one class; skipped.")
                continue
            picked.append(t)
        return picked

    candidates = []  # (balance, column, family)
    for col in outcome_cols:
        if col not in train.columns:
            continue
        y = to_binary(train[col])
        if y is None:
            continue
        pos, neg = int(y.sum()), int((1 - y).sum())
        if min(pos, neg) < MIN_CLASS_TRAIN:
            continue
        candidates.append((min(pos, neg) / max(pos, neg), col, family_of(col)))

    # Best-balanced variant per family, then families ranked by balance.
    best_per_family: dict[str, tuple] = {}
    for balance, col, fam in candidates:
        if fam not in best_per_family or balance > best_per_family[fam][0]:
            best_per_family[fam] = (balance, col)
    ranked = sorted(best_per_family.items(), key=lambda kv: -kv[1][0])
    targets = [col for _, (_, col) in ranked[:max_targets]]

    # Force in the best-balanced mortality family if eligible but not chosen.
    if not any(MORTALITY_PATTERN.search(t) for t in targets):
        mortality = [(bal, col) for fam, (bal, col) in ranked if MORTALITY_PATTERN.search(fam)]
        if mortality:
            best = max(mortality)[1]
            if len(targets) >= max_targets:
                targets[-1] = best
            else:
                targets.append(best)
            print(f"Force-included mortality target '{best}' (clinically central endpoint).")
    return targets

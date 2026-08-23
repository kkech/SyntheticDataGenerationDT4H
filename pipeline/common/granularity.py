"""
Empirical granularity of numeric columns, and snapping synthetic values
onto it.

Real clinical values live on grids -- integer ages and blood pressures,
one-decimal weights, two-decimal assay results -- while neural samplers
emit full-precision floats. Snapping restores face validity of released
records. (Measured caveat, reported in the paper: snapping does NOT
reduce C2ST distinguishability -- the ablation left the AUC unchanged --
so this is a cosmetic fix, not a fidelity one.)

Snapped values are kept on the observed support's grid: a value that
would round below the column's observed minimum is lifted to the
smallest grid point at or above it, so an up-to-date sentinel decode
never reinterprets a snapped value as missing.
"""

import math

import numpy as np
import pandas as pd

MAX_DECIMALS = 4
CONFORMITY = 0.995  # share of observed values that must sit on the grid


def infer_granularity(reference: pd.DataFrame) -> dict:
    """Per-column decimal places for numeric columns whose observed
    values sit on a grid: {column: {"decimals": d, "min": m, "max": M}}.
    Columns with no detectable grid (or no observed values) are omitted
    and must not be snapped."""
    out = {}
    for c in reference.columns:
        if not pd.api.types.is_numeric_dtype(reference[c]):
            continue
        v = pd.to_numeric(reference[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(v) == 0:
            continue
        for d in range(MAX_DECIMALS + 1):
            snapped = np.round(v, d)
            ok = np.isclose(v, snapped, rtol=0, atol=10.0 ** (-d) * 1e-6 + 1e-9)
            if ok.mean() >= CONFORMITY:
                out[c] = {"decimals": d,
                          "min": float(v.min()), "max": float(v.max())}
                break
    return out


def snap_to_granularity(df: pd.DataFrame, grid: dict):
    """Round each gridded numeric column of `df` to its reference
    decimals, lifting results below the observed minimum onto the
    smallest grid point >= min. Nulls untouched. Returns (frame,
    {column: cells_changed})."""
    out = df.copy()
    changed = {}
    for c, spec in grid.items():
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        d = spec["decimals"]
        snapped = col.round(d)
        # smallest grid point at or above the observed minimum, so a
        # snapped value can never fall below the sentinel-decode floor
        step = 10.0 ** (-d)
        floor_grid = math.ceil(spec["min"] / step - 1e-9) * step
        snapped = snapped.where(snapped.isna() | (snapped >= floor_grid), floor_grid)
        n = int(((snapped != col) & col.notna()).sum())
        if n:
            changed[c] = n
        out[c] = snapped.round(d)  # re-round: floor_grid substitution is float math
    return out, changed

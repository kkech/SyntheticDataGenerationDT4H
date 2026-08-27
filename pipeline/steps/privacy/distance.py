"""
Record-level distance machinery for the privacy assessment.

Distance between two records is a Gower-style mixed-type mean over
columns: numeric columns contribute |a-b| scaled by the column's range
(capped at 1), categorical columns contribute 0 on match and 1 on
mismatch. Distances live in [0,1]; 0 means an identical record.

Computed in the SENTINEL-ENCODED space, deliberately: sentinels make
"missing" a concrete comparable value, so two records that are missing
the same labs are (correctly) close in that respect, and a synthetic
record can only be near a real one by matching both its values and its
missingness pattern. Synthetic frames (stored decoded) are re-encoded
through the persisted map before measuring.
"""

import numpy as np
import pandas as pd


def build_encoder(real: pd.DataFrame, encoding: dict):
    """Fit the column roles and scales on the real (sentinel-space) frame.

    Returns a function that maps any frame with the same columns to a
    pair of matrices (numeric scaled to [0,~1], categorical int codes),
    plus the list of columns actually used.
    """
    numeric_cols, cat_cols, scales, offsets = [], [], [], []
    categories: dict[str, pd.Index] = {}

    for c in real.columns:
        if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c]):
            col = pd.to_numeric(real[c], errors="coerce")
            lo, hi = float(col.min()), float(col.max())
            if hi - lo <= 0:
                continue  # constant numeric: contributes nothing to any distance
            numeric_cols.append(c)
            offsets.append(lo)
            scales.append(hi - lo)
        else:
            s = real[c].astype("object").where(real[c].notna(), "Missing").astype(str)
            cats = pd.Index(s.unique())
            if len(cats) < 2:
                continue  # constant categorical: contributes nothing
            cat_cols.append(c)
            categories[c] = cats

    offsets_arr = np.array(offsets)
    scales_arr = np.array(scales)

    def encode(df: pd.DataFrame):
        num = np.empty((len(df), len(numeric_cols)))
        for i, c in enumerate(numeric_cols):
            col = pd.to_numeric(df[c], errors="coerce")
            if c in encoding:
                # re-encode decoded nulls back to the sentinel
                col = col.fillna(encoding[c]["sentinel"])
            else:
                col = col.fillna(offsets_arr[i])
            num[:, i] = np.clip((col.to_numpy(dtype=float) - offsets_arr[i]) / scales_arr[i], -1.0, 2.0)

        cat = np.empty((len(df), len(cat_cols)), dtype=np.int32)
        for i, c in enumerate(cat_cols):
            s = df[c].astype("object").where(df[c].notna(), "Missing").astype(str)
            # unseen synthetic categories get a code of their own (-1),
            # which mismatches every real category -- the correct behavior
            cat[:, i] = categories[c].get_indexer(s)
        return num, cat

    return encode, numeric_cols, cat_cols


def nearest_two_distances(query_num, query_cat, ref_num, ref_cat, exclude_self=False,
                          chunk_rows: int = 128):
    """For each query record: distance to its nearest and second-nearest
    reference record. Chunked brute force -- exact, no approximation, and
    a few minutes at 4694x4694 with ~250 columns."""
    n_cols = query_num.shape[1] + query_cat.shape[1]
    n_query, n_ref = query_num.shape[0], ref_num.shape[0]
    d1 = np.empty(n_query)
    d2 = np.empty(n_query)

    for start in range(0, n_query, chunk_rows):
        end = min(start + chunk_rows, n_query)
        # numeric: mean |a-b|/range capped at 1
        num_d = np.abs(query_num[start:end, None, :] - ref_num[None, :, :])
        np.minimum(num_d, 1.0, out=num_d)
        dist = num_d.sum(axis=2)
        # categorical: mismatch count
        dist += (query_cat[start:end, None, :] != ref_cat[None, :, :]).sum(axis=2)
        dist /= n_cols

        if exclude_self:
            idx = np.arange(start, end)
            dist[np.arange(end - start), idx] = np.inf

        part = np.partition(dist, 1, axis=1)
        d1[start:end] = part[:, 0]
        d2[start:end] = part[:, 1]

    return d1, d2


def summarize_dcr(d1: np.ndarray, d2: np.ndarray) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        # d2 == 0 means the record coincides EXACTLY with (at least) two
        # training records -- the most alarming case, not the safest, so
        # it must score 0 (locked onto real records), never the
        # "population structure" value 1.
        nndr = np.where(d2 > 0, d1 / d2, 0.0)
    return {
        "records": int(len(d1)),
        "dcr_min": round(float(d1.min()), 6),
        "dcr_p5": round(float(np.percentile(d1, 5)), 6),
        "dcr_median": round(float(np.median(d1)), 6),
        "dcr_mean": round(float(d1.mean()), 6),
        "exact_matches": int((d1 == 0).sum()),
        "nndr_median": round(float(np.median(nndr)), 4),
        "nndr_p5": round(float(np.percentile(nndr, 5)), 4),
    }


def nearest_k_distances(query_num, query_cat, ref_num, ref_cat, k: int = 5,
                        exclude_self=False, chunk_rows: int = 128):
    """For each query record: sorted distances to its k nearest reference
    records (same metric and chunking as nearest_two_distances)."""
    n_cols = query_num.shape[1] + query_cat.shape[1]
    n_query = query_num.shape[0]
    out = np.empty((n_query, k))

    for start in range(0, n_query, chunk_rows):
        end = min(start + chunk_rows, n_query)
        num_d = np.abs(query_num[start:end, None, :] - ref_num[None, :, :])
        np.minimum(num_d, 1.0, out=num_d)
        dist = num_d.sum(axis=2)
        dist += (query_cat[start:end, None, :] != ref_cat[None, :, :]).sum(axis=2)
        dist /= n_cols
        if exclude_self:
            idx = np.arange(start, end)
            dist[np.arange(end - start), idx] = np.inf
        part = np.partition(dist, k - 1, axis=1)[:, :k]
        part.sort(axis=1)
        out[start:end] = part

    return out

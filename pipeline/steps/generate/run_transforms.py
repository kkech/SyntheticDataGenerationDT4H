"""
Model-side, invertible training-frame transforms for individual runs.

Applied AROUND a synthesizer -- the training frame is transformed before
fit, and samples are inverse-transformed before any post-processing --
so leakage checks, sentinel decode, and every downstream evaluation see
the ordinary sentinel-space schema regardless of what the model itself
was shown. The fitted generator is saved WRAPPED (TransformedSynthesizer)
so regenerate.py / conditional_demo.py / top-up sampling produce
already-inverted samples and cannot leak the transformed space.

Two transforms, each answering one roadmap question:

  * "quantile" -- rank/quantile-normalize numeric columns (fit on the
    training split, inverted exactly onto the empirical quantiles).
    Tests whether the neural models' numeric-fidelity gap is a
    heavy-tail representation problem.
  * "indicator" -- replace each sentinel-encoded numeric's missing cells
    with the column median and add an explicit boolean missing-indicator
    column; inverted by restoring the sentinel where the sampled
    indicator says missing. Ablates the sentinel design against the
    classic indicator+value encoding.
"""

import numpy as np
import pandas as pd

INDICATOR_SUFFIX = "__was_missing"


class QuantileRunTransform:
    name = "quantile"

    def __init__(self, seed=0):
        self.seed = seed
        self._qt = None
        self._cols = None

    def forward(self, train: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import QuantileTransformer

        self._cols = [c for c in train.columns
                      if pd.api.types.is_numeric_dtype(train[c])
                      and not pd.api.types.is_bool_dtype(train[c])]
        out = train.copy()
        if not self._cols:
            return out
        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, len(train)),
            subsample=10 ** 9,
            random_state=self.seed,
        )
        vals = out[self._cols].to_numpy(dtype=float)
        if np.isnan(vals).any():
            bad = [c for c in self._cols if out[c].isna().any()]
            raise ValueError(
                f"quantile transform expects the sentinel-encoded training frame "
                f"(no numeric nulls); found NaN in {bad[:5]}")
        out[self._cols] = self._qt.fit_transform(vals)
        return out

    def inverse(self, sampled: pd.DataFrame) -> pd.DataFrame:
        out = sampled.copy()
        cols = [c for c in (self._cols or []) if c in out.columns]
        if not cols or self._qt is None:
            return out
        # inverse only the columns present, preserving NaN cells
        vals = out[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        nan_mask = np.isnan(vals)
        idx = [self._cols.index(c) for c in cols]
        full = np.zeros((len(out), len(self._cols)))
        full[:, idx] = np.nan_to_num(vals)
        inv = self._qt.inverse_transform(full)[:, idx]
        inv[nan_mask] = np.nan
        out[cols] = inv
        return out


class IndicatorRunTransform:
    name = "indicator"

    def __init__(self, encoding: dict):
        # {column: {"sentinel", "decode_threshold", ...}} from the
        # committed public sentinel map.
        self.encoding = encoding
        self._applied = None  # {column: {"sentinel", "fill"}}

    def forward(self, train: pd.DataFrame) -> pd.DataFrame:
        out = train.copy()
        self._applied = {}
        indicators = {}
        for col, spec in self.encoding.items():
            if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
                continue
            vals = pd.to_numeric(out[col], errors="coerce")
            missing = vals <= spec["decode_threshold"]
            if not missing.any():
                continue
            fill = float(vals[~missing].median())
            out[col] = vals.where(~missing, fill)
            # string categories, matching the schema's boolean spelling
            indicators[col + INDICATOR_SUFFIX] = missing.map({True: "true", False: "false"})
            self._applied[col] = {"sentinel": spec["sentinel"], "fill": fill}
        if indicators:
            out = pd.concat([out, pd.DataFrame(indicators, index=out.index)], axis=1)
        return out

    def inverse(self, sampled: pd.DataFrame) -> pd.DataFrame:
        out = sampled.copy()
        for col, spec in (self._applied or {}).items():
            ind = col + INDICATOR_SUFFIX
            if col not in out.columns:
                continue
            if ind in out.columns:
                missing = (out[ind].astype("object").where(out[ind].notna(), "false")
                           .astype(str).str.strip().str.lower() == "true")
                vals = pd.to_numeric(out[col], errors="coerce")
                out[col] = vals.where(~missing, spec["sentinel"])
        drop = [c for c in out.columns if c.endswith(INDICATOR_SUFFIX)]
        return out.drop(columns=drop)


class TransformedSynthesizer:
    """A fitted synthesizer bundled with its run transform: .sample()
    returns already-inverted (ordinary sentinel-space) rows, so saved
    pickles of transformed runs behave exactly like plain ones."""

    def __init__(self, synth, transform):
        self._synth = synth
        self._transform = transform
        self.name = f"{getattr(synth, 'name', type(synth).__name__)}+{transform.name}"
        self.params = getattr(synth, "params", {})
        self.is_dp = getattr(synth, "is_dp", False)
        self.uses_gpu = getattr(synth, "uses_gpu", False)

    def sample(self, n_rows: int) -> pd.DataFrame:
        return self._transform.inverse(self._synth.sample(n_rows))

    def describe(self) -> dict:
        d = self._synth.describe() if hasattr(self._synth, "describe") else {}
        d["run_transform"] = self._transform.name
        return d


def build_run_transform(kind, encoding: dict, seed=0):
    """kind: None | 'quantile' | 'indicator' -> transform instance or None."""
    if not kind:
        return None
    if kind == "quantile":
        return QuantileRunTransform(seed=seed)
    if kind == "indicator":
        if not encoding:
            raise ValueError("indicator transform needs the numeric sentinel "
                             "encoding map (run the preprocess step first)")
        return IndicatorRunTransform(encoding)
    raise ValueError(f"Unknown run transform: {kind!r}")

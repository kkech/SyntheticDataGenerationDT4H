"""
Non-DP synthesizers from SDV (Synthetic Data Vault).

NOTE ON LICENSING: SDV moved from MIT to the Business Source License in
2023. Research/non-production use is permitted, but production use is
restricted -- worth confirming with the consortium before this becomes
load-bearing. SDV's evaluation library (SDMetrics) remains MIT.
"""

import pandas as pd

from pipeline.steps.generate.synthesizers.base import Synthesizer, gpu_available


def _build_metadata(df: pd.DataFrame):
    """
    SDV renamed its metadata API between versions (SingleTableMetadata ->
    Metadata). Try the current one first, fall back to the older one.
    """
    try:
        from sdv.metadata import Metadata

        return Metadata.detect_from_dataframe(data=df)
    except (ImportError, AttributeError):
        from sdv.metadata import SingleTableMetadata

        md = SingleTableMetadata()
        md.detect_from_dataframe(data=df)
        return md


def _gpu_kwarg(model_cls) -> dict:
    """
    SDV renamed the GPU flag from `cuda` to `enable_gpu` and deprecated
    the old name. Pick whichever the installed version actually accepts,
    so this works across versions without emitting a deprecation warning
    on new ones or crashing on old ones.
    """
    import inspect

    try:
        params = inspect.signature(model_cls.__init__).parameters
    except (TypeError, ValueError):
        return {}

    use_gpu = gpu_available()
    if "enable_gpu" in params:
        return {"enable_gpu": use_gpu}
    if "cuda" in params:
        return {"cuda": use_gpu}
    return {}


def _seeded_sample(model, n_rows: int, seed) -> pd.DataFrame:
    """Sample with the RUN's seed instead of SDV's pinned one.

    SDV deliberately fixes its sampling RNG (single_table/base.py sets a
    FIXED_RNG_SEED at sample time unless a random state was explicitly
    set), so without this, every seed of the same fitted model produces
    byte-identical output and the seed-variance measurement is fiction.
    _set_random_state is private but stable across the SDV 1.x line; if
    a future version removes it, fall back to the pinned behaviour
    loudly rather than silently."""
    if seed is not None:
        try:
            model._set_random_state(int(seed))
        except Exception as e:
            print(f"⚠️  Could not set SDV sampling seed ({type(e).__name__}: {e}) -- "
                  f"sampling falls back to SDV's fixed internal seed, so outputs "
                  f"will NOT vary across seeds.")
    return model.sample(num_rows=n_rows)


class _SDVSynthesizer(Synthesizer):
    """Shared per-call sampling seed bookkeeping for the SDV models.

    Resetting SDV's RNG to the SAME integer on every sample() call made
    repeated calls return IDENTICAL batches. That silently broke every
    caller that samples more than once from one fitted model:
    conditional_demo's rejection sampling and coherent_sample's top-up
    loops would draw the same rows forever, and the "extra" rows a
    top-up produced were duplicates of the first batch.

    Fix: seed call i with `seed + i`, counting calls on the INSTANCE. A
    single-batch run is unchanged (i = 0), so the committed CSVs still
    reproduce, and successive calls are different but still fully
    determined by the run seed. The counter is excluded from the pickled
    state (see __getstate__), so a generator loaded from disk always
    starts again at i = 0 and its first batch reproduces the committed
    output byte for byte.
    """

    def _next_sample_seed(self):
        seed = self.params.get("seed")
        call = getattr(self, "_sample_calls", 0)
        self._sample_calls = call + 1
        return None if seed is None else int(seed) + call

    def sample(self, n_rows: int) -> pd.DataFrame:
        return _seeded_sample(self._model, n_rows, self._next_sample_seed())

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_sample_calls", None)  # a fresh load starts at call 0
        return state


def save_metadata(model, path: str) -> bool:
    """
    Persist the detected SDV metadata alongside the run.

    SDV warns that metadata should be saved for replicability across
    versions, and for a dataset headed for publication that warning is
    worth heeding: the metadata records how each column was interpreted
    (categorical vs numerical vs datetime), which is exactly the detail
    needed to regenerate the same output later.
    """
    md = getattr(model, "metadata", None) or getattr(model, "_metadata", None)
    if md is None or not hasattr(md, "save_to_json"):
        return False
    try:
        md.save_to_json(path)
        return True
    except Exception:
        return False


class SDVCTGANSynthesizer(_SDVSynthesizer):
    """Conditional Tabular GAN. The long-standing baseline; recent
    benchmarks report diffusion-based models and TVAE outperforming it."""

    name = "ctgan"
    is_dp = False
    uses_gpu = True

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        from sdv.single_table import CTGANSynthesizer

        metadata = _build_metadata(df)
        self._model = CTGANSynthesizer(
            metadata,
            epochs=self.params.get("epochs", 500),
            batch_size=self.params.get("batch_size", 500),
            verbose=self.params.get("verbose", True),
            **_gpu_kwarg(CTGANSynthesizer),
        )
        self._model.fit(df)


class SDVTVAESynthesizer(_SDVSynthesizer):
    """Tabular VAE. Usually stronger than CTGAN on mixed-type tabular data
    and cheaper to train, so it is a useful second baseline."""

    name = "tvae"
    is_dp = False
    uses_gpu = True

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        from sdv.single_table import TVAESynthesizer

        metadata = _build_metadata(df)
        # Optional capacity overrides for the architecture sweep; SDV
        # defaults apply when unset (embedding 128, (128,128) nets).
        capacity = {k: tuple(v) if isinstance(v, list) else v
                    for k in ("embedding_dim", "compress_dims", "decompress_dims")
                    if (v := self.params.get(k)) is not None}
        self._model = TVAESynthesizer(
            metadata,
            epochs=self.params.get("epochs", 500),
            batch_size=self.params.get("batch_size", 500),
            **capacity,
            **_gpu_kwarg(TVAESynthesizer),
        )
        self._model.fit(df)


class SDVGaussianCopulaSynthesizer(_SDVSynthesizer):
    """Fast statistical baseline -- no neural training, no GPU. Useful as a
    sanity floor: any deep model that cannot beat it is misconfigured."""

    name = "gaussian_copula"
    is_dp = False
    uses_gpu = False

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        from sdv.single_table import GaussianCopulaSynthesizer

        metadata = _build_metadata(df)
        self._model = GaussianCopulaSynthesizer(metadata)
        self._model.fit(df)

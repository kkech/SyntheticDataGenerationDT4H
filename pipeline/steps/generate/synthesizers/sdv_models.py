"""
Non-DP synthesizers from SDV (Synthetic Data Vault).

NOTE ON LICENSING: SDV moved from MIT to the Business Source License in
2023. Research/non-production use is permitted, but production use is
restricted -- worth confirming with the consortium before this becomes
load-bearing. SDV's evaluation library (SDMetrics) remains MIT.
"""

import pandas as pd

from pipeline.steps.generate.synthesizers.base import Synthesizer


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


class SDVCTGANSynthesizer(Synthesizer):
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
            cuda=self.params.get("cuda", True),
            verbose=self.params.get("verbose", True),
        )
        self._model.fit(df)

    def sample(self, n_rows: int) -> pd.DataFrame:
        return self._model.sample(num_rows=n_rows)


class SDVTVAESynthesizer(Synthesizer):
    """Tabular VAE. Usually stronger than CTGAN on mixed-type tabular data
    and cheaper to train, so it is a useful second baseline."""

    name = "tvae"
    is_dp = False
    uses_gpu = True

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        from sdv.single_table import TVAESynthesizer

        metadata = _build_metadata(df)
        self._model = TVAESynthesizer(
            metadata,
            epochs=self.params.get("epochs", 500),
            batch_size=self.params.get("batch_size", 500),
            cuda=self.params.get("cuda", True),
        )
        self._model.fit(df)

    def sample(self, n_rows: int) -> pd.DataFrame:
        return self._model.sample(num_rows=n_rows)


class SDVGaussianCopulaSynthesizer(Synthesizer):
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

    def sample(self, n_rows: int) -> pd.DataFrame:
        return self._model.sample(num_rows=n_rows)

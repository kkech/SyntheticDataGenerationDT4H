"""
Differentially private synthesizers from OpenDP smartnoise-synth.

Two families, and the choice between them matters:

  * Marginal-based (AIM, MST, PATE-CTGAN's cousins): measure noisy
    low-dimensional marginals under DP and fit a graphical model to them.
    Utility-oriented benchmarks consistently report these OUTPERFORMING
    DP-GANs on tabular data -- at moderate epsilon, AIM has been reported
    close to real-data utility. AIM is workload-aware and generally beats
    MST, especially at higher epsilon. These are CPU-only: there is no
    GPU training to speed up.

  * DP-GAN (DPCTGAN): the deep-learning route. Included for comparison
    with the project's original approach, but the literature suggests it
    is the weaker choice for tabular data.

SCALING CAVEAT: AIM and MST both build on Private-PGM, which is documented
to struggle as the column count grows, in both fitting and sampling. This
dataset has ~329 columns, which is squarely in the risky range. Use
`max_columns` to trial a subset before committing to a full run, and fall
back to MST (cheaper, pairwise-tree-based) if AIM will not fit in memory.
"""

import pandas as pd

from pipeline.steps.generate.synthesizers.base import Synthesizer


class _SmartNoiseBase(Synthesizer):
    """Shared fit/sample for smartnoise-synth's Synthesizer.create API."""

    algorithm: str
    is_dp = True

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        from snsynth import Synthesizer as SNSynthesizer

        create_kwargs = {"epsilon": self.params.get("epsilon", 15.0)}
        # Only the GAN-based synthesizers take epochs/batch_size; passing
        # them to AIM/MST raises.
        if self.params.get("pass_training_params", False):
            create_kwargs["epochs"] = self.params.get("epochs", 300)
            create_kwargs["batch_size"] = self.params.get("batch_size", 50)

        self._model = SNSynthesizer.create(self.algorithm, **create_kwargs)
        self._model.fit(
            df,
            categorical_columns=categorical_columns,
            continuous_columns=continuous_columns,
            preprocessor_eps=self.params.get("preprocessor_eps", 1.0),
            nullable=False,  # preprocessing guarantees no nulls/NaN remain
        )

    def sample(self, n_rows: int) -> pd.DataFrame:
        return self._model.sample(n_rows)


class AIMSynthesizer(_SmartNoiseBase):
    """Marginal-based, workload-aware. Current recommendation for DP
    tabular synthesis -- but watch the Private-PGM scaling caveat above."""

    name = "aim"
    algorithm = "aim"
    uses_gpu = False


class MSTSynthesizer(_SmartNoiseBase):
    """Marginal-based over a maximum spanning tree of pairwise
    correlations. Cheaper than AIM; the fallback if AIM will not scale."""

    name = "mst"
    algorithm = "mst"
    uses_gpu = False


class PATECTGANSynthesizer(_SmartNoiseBase):
    """DP-GAN variant using the PATE framework."""

    name = "patectgan"
    algorithm = "patectgan"
    uses_gpu = True

    def __init__(self, **params):
        params.setdefault("pass_training_params", True)
        super().__init__(**params)


class DPCTGANSynthesizer(_SmartNoiseBase):
    """
    The project's original DP approach. Kept for comparison, but note the
    benchmark evidence favours AIM/MST for tabular data -- treat this as
    the baseline being measured against, not the default.
    """

    name = "dpctgan"
    algorithm = "dpctgan"
    uses_gpu = True

    def __init__(self, **params):
        params.setdefault("pass_training_params", True)
        super().__init__(**params)

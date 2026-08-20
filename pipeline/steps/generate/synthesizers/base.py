"""
Common interface every synthesizer implements, so the generate step can
swap between them without knowing which library is underneath.

Heavy dependencies (sdv, snsynth, torch) are imported lazily inside fit()
rather than at module import, so a missing or broken install only fails
the synthesizer that needs it -- the others still run. That matters here:
sdv and smartnoise-synth pin different dependency versions and are easy
to end up with only one of installed.
"""

from abc import ABC, abstractmethod

import pandas as pd


def gpu_available() -> bool:
    """
    True only if torch is installed AND a CUDA device is actually usable.

    Checked at fit time rather than hardcoded, so the same code runs on a
    GPU box and a CPU-only one: the model trains either way, just slower
    without a GPU. Passing cuda=True on a machine with no CUDA device
    makes the SDV models raise instead of falling back.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class Synthesizer(ABC):
    #: Registry key, used in config and as the output subfolder name.
    name: str
    #: Whether this synthesizer provides differential privacy guarantees.
    is_dp: bool = False
    #: Whether it can use the GPU (marginal-based DP methods are CPU-only).
    uses_gpu: bool = False

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, df: pd.DataFrame, categorical_columns: list[str], continuous_columns: list[str]) -> None:
        """Train on the real preprocessed data."""
        ...

    @abstractmethod
    def sample(self, n_rows: int) -> pd.DataFrame:
        """Generate n_rows synthetic records."""
        ...

    def describe(self) -> dict:
        """Run metadata recorded in the generation summary."""
        return {
            "name": self.name,
            "is_dp": self.is_dp,
            "uses_gpu": self.uses_gpu,
            "params": dict(self.params),
        }

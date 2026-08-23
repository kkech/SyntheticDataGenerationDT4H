"""
Synthesizer registry. Adding a new model means writing a Synthesizer
subclass and adding one line here -- the generate step and config need no
changes.
"""

from pipeline.steps.generate.synthesizers.base import Synthesizer
from pipeline.steps.generate.synthesizers.ddpm import DDPMSynthesizer
from pipeline.steps.generate.synthesizers.sdv_models import (
    SDVCTGANSynthesizer,
    SDVGaussianCopulaSynthesizer,
    SDVTVAESynthesizer,
)
from pipeline.steps.generate.synthesizers.smartnoise_models import (
    AIMSynthesizer,
    DPCTGANSynthesizer,
    MSTSynthesizer,
    PATECTGANSynthesizer,
)

REGISTRY: dict[str, type[Synthesizer]] = {
    cls.name: cls
    for cls in (
        # non-DP
        SDVCTGANSynthesizer,
        SDVTVAESynthesizer,
        SDVGaussianCopulaSynthesizer,
        DDPMSynthesizer,
        # DP
        AIMSynthesizer,
        MSTSynthesizer,
        PATECTGANSynthesizer,
        DPCTGANSynthesizer,
    )
}


def build_synthesizer(name: str, **params) -> Synthesizer:
    if name not in REGISTRY:
        raise ValueError(f"Unknown synthesizer '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


__all__ = ["REGISTRY", "Synthesizer", "build_synthesizer"]

"""Interface every pipeline step implements."""

from abc import ABC, abstractmethod

from pipeline.config import PipelineConfig


class PipelineStep(ABC):
    #: Unique id used for status tracking and the --only/--force-step CLI flags.
    name: str

    @abstractmethod
    def run(self, config: PipelineConfig) -> None:
        """Execute the step. Raise on failure -- the runner records it and stops."""
        ...

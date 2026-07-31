from abc import ABC, abstractmethod

from ab_tea_lab._params import ParamsMixin
from ab_tea_lab.types import SimulationData


class BaseSimulator(ParamsMixin, ABC):
    """Abstract base class for data-generating processes.

    Subclasses must implement ``run`` (produce one simulated dataset) and the
    ``true_effect`` property (the ground-truth treatment effect used by the
    evaluator).
    """

    @abstractmethod
    def run(self) -> SimulationData:
        """Generate and return a single simulated dataset."""

    @property
    @abstractmethod
    def true_effect(self) -> float:
        """The true treatment effect embedded in the simulation."""

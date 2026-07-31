from abc import ABC, abstractmethod

from ab_tea_lab.types import EvaluationResult, ModelResult


class BaseEvaluator(ABC):
    """Abstract base class for computing performance metrics over simulation runs."""

    @abstractmethod
    def evaluate(self, results: list[ModelResult], true_effect: float) -> EvaluationResult:
        """Aggregate *results* from many simulation runs into performance metrics.

        Parameters
        ----------
        results:
            One :class:`ModelResult` per simulation run.
        true_effect:
            The ground-truth treatment effect used by the simulator.
        """

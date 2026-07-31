import json
import logging
from pathlib import Path

from ab_tea_lab.evaluator.base import BaseEvaluator
from ab_tea_lab.model.base import BaseModel
from ab_tea_lab.simulator.base import BaseSimulator
from ab_tea_lab.types import EvaluationResult, ModelResult

logger = logging.getLogger(__name__)


class Experiment:
    """Orchestrates a Monte Carlo simulation study.

    Runs a simulator many times, applies each statistical model to every
    generated dataset, and feeds the collected results into an evaluator.

    Parameters
    ----------
    simulator:
        Data-generating process.
    models:
        Statistical methods to compare.  Accepts either a *dict* mapping
        names to model instances, or a *list* (names are derived from class
        names, suffixed with an index when duplicates exist).
    evaluator:
        Computes aggregate performance metrics.
    n_simulations:
        Number of Monte Carlo repetitions.
    random_seed:
        Optional seed forwarded to the simulator (if it accepts one via
        ``set_params``).
    save_dir:
        Optional directory path. When provided, each per-iteration
        ``ModelResult`` is saved as JSON under
        ``save_dir/<model_name>/iter_<i>.json``.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        models: dict[str, BaseModel] | list[BaseModel],
        evaluator: BaseEvaluator,
        n_simulations: int = 1000,
        random_seed: int | None = None,
        save_dir: str | Path | None = None,
    ) -> None:
        self.simulator = simulator
        self.models = self._normalize_models(models)
        self.evaluator = evaluator
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.raw_results: dict[str, list[ModelResult]] | None = None

    def run(self) -> dict[str, EvaluationResult]:
        """Execute the full simulation study.

        Returns a dict mapping model names to their :class:`EvaluationResult`.
        """
        if self.random_seed is not None:
            try:
                self.simulator.set_params(random_seed=self.random_seed)
            except ValueError:
                logger.debug("Simulator does not accept 'random_seed'; ignoring.")

        collected: dict[str, list[ModelResult]] = {name: [] for name in self.models}

        for i in range(self.n_simulations):
            data = self.simulator.run()
            for name, model in self.models.items():
                model.fit(data)
                collected[name].append(model.result)
                if self.save_dir is not None:
                    out_dir = self.save_dir / name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    model.save_result(out_dir / f"iter_{i}.json")

            if (i + 1) % max(1, self.n_simulations // 10) == 0:
                logger.info("Completed %d / %d simulations", i + 1, self.n_simulations)

        self.raw_results = collected

        true_effect = self.simulator.true_effect
        return {name: self.evaluator.evaluate(results, true_effect) for name, results in collected.items()}

    @staticmethod
    def _normalize_models(models: dict[str, BaseModel] | list[BaseModel]) -> dict[str, BaseModel]:
        if isinstance(models, dict):
            return models

        named: dict[str, BaseModel] = {}
        counts: dict[str, int] = {}
        for model in models:
            cls_name = type(model).__name__
            idx = counts.get(cls_name, 0)
            counts[cls_name] = idx + 1
            key = cls_name if idx == 0 else f"{cls_name}_{idx}"
            named[key] = model

        # Retroactively suffix the first entry when duplicates appeared
        for cls_name, count in counts.items():
            if count > 1 and cls_name in named:
                named[f"{cls_name}_0"] = named.pop(cls_name)

        return named


def evaluate_from_dir(
    save_dir: str | Path,
    evaluator: BaseEvaluator,
    true_effect: float,
) -> dict[str, EvaluationResult]:
    """Reconstruct ``ModelResult`` objects from a save directory and evaluate them.

    Parameters
    ----------
    save_dir:
        Root directory previously written by ``Experiment.run(save_dir=...)``.
        Expected layout: ``save_dir/<model_name>/iter_<i>.json``.
    evaluator:
        Evaluator instance used to compute aggregate metrics.
    true_effect:
        Ground-truth treatment effect for evaluation.

    Returns
    -------
    dict mapping model names (subdirectory names) to their
    :class:`EvaluationResult`.
    """
    save_dir = Path(save_dir)
    results: dict[str, EvaluationResult] = {}

    for model_dir in sorted(save_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        iter_files = sorted(
            model_dir.glob("iter_*.json"),
            key=lambda p: int(p.stem.split("_", 1)[1]),
        )
        model_results: list[ModelResult] = []
        for f in iter_files:
            with open(f) as fh:
                model_results.append(ModelResult(**json.load(fh)))

        if model_results:
            results[model_dir.name] = evaluator.evaluate(model_results, true_effect)

    return results

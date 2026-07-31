"""Integration tests using real Evaluator, InverseVariance model, and RandomSimulator."""

from __future__ import annotations

from ab_tea_lab.evaluator.evaluator import Evaluator
from ab_tea_lab.experiment.base import Experiment
from ab_tea_lab.model.inverse_variance import InverseVariance
from ab_tea_lab.simulator.random import RandomSimulator
from ab_tea_lab.types import EvaluationResult


def _make_simulator(effect_mean: float = 0.0, seed: int = 42) -> RandomSimulator:
    return RandomSimulator(
        num_advertisers=20,
        campaigns_mean=3.0,
        spend_mean=665.0,
        spend_sd=872.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=effect_mean,
        effect_sd=0.1,
        equal_spend=True,
        random_seed=seed,
    )


class TestIntegrationNullEffect:
    """Full pipeline under H0 (no treatment effect)."""

    def test_end_to_end_null(self) -> None:
        simulator = _make_simulator(effect_mean=0.0)
        models = [InverseVariance(method_re=None, alpha=0.05)]
        evaluator = Evaluator()

        experiment = Experiment(
            simulator=simulator,
            models=models,
            evaluator=evaluator,
            n_simulations=50,
        )

        results = experiment.run()

        assert len(results) == 1
        name = next(iter(results))
        er = results[name]
        assert isinstance(er, EvaluationResult)

        assert 0.0 <= er.coverage <= 1.0
        assert er.rmse >= 0.0
        assert er.mean_interval_width > 0.0
        assert er.mean_p_value is None or 0.0 <= er.mean_p_value <= 1.0

    def test_coverage_reasonable_under_null(self) -> None:
        simulator = _make_simulator(effect_mean=0.0)
        models = [InverseVariance(method_re=None, alpha=0.05)]
        evaluator = Evaluator()

        experiment = Experiment(
            simulator=simulator,
            models=models,
            evaluator=evaluator,
            n_simulations=200,
            random_seed=99,
        )

        results = experiment.run()
        er = next(iter(results.values()))
        assert er.coverage >= 0.80, f"Coverage too low under H0: {er.coverage:.2f}"


class TestIntegrationNonNullEffect:
    """Full pipeline under H1 (positive treatment effect)."""

    def test_end_to_end_positive_effect(self) -> None:
        simulator = _make_simulator(effect_mean=0.2, seed=123)
        models = [InverseVariance(method_re=None, alpha=0.05)]
        evaluator = Evaluator()

        experiment = Experiment(
            simulator=simulator,
            models=models,
            evaluator=evaluator,
            n_simulations=50,
        )

        results = experiment.run()
        er = next(iter(results.values()))
        assert isinstance(er, EvaluationResult)
        assert er.rmse >= 0.0
        assert er.mean_interval_width > 0.0


class TestIntegrationMultipleModels:
    """Pipeline with several InverseVariance configurations."""

    def test_fe_vs_re(self) -> None:
        simulator = _make_simulator(effect_mean=0.0, seed=7)
        models = {
            "FE": InverseVariance(method_re=None, alpha=0.05),
            "RE_iterated": InverseVariance(method_re="iterated", alpha=0.05),
        }
        evaluator = Evaluator()

        experiment = Experiment(
            simulator=simulator,
            models=models,
            evaluator=evaluator,
            n_simulations=50,
        )

        results = experiment.run()

        assert "FE" in results
        assert "RE_iterated" in results

        for name, er in results.items():
            assert isinstance(er, EvaluationResult)
            assert 0.0 <= er.coverage <= 1.0
            assert er.mean_interval_width > 0.0

        assert results["RE_iterated"].mean_tau2 is not None

    def test_raw_results_stored(self) -> None:
        simulator = _make_simulator(effect_mean=0.0, seed=11)
        n_sims = 30
        models = [InverseVariance()]
        evaluator = Evaluator()

        experiment = Experiment(
            simulator=simulator,
            models=models,
            evaluator=evaluator,
            n_simulations=n_sims,
        )

        experiment.run()

        assert experiment.raw_results is not None
        name = next(iter(experiment.raw_results))
        assert len(experiment.raw_results[name]) == n_sims

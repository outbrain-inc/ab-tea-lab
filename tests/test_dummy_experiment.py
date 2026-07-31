"""Minimal dummy implementations to smoke-test the full experiment pipeline."""

from __future__ import annotations

import math
import random
from typing import Self

import pandas as pd

from ab_tea_lab.evaluator.base import BaseEvaluator
from ab_tea_lab.experiment.base import Experiment
from ab_tea_lab.model.base import BaseModel
from ab_tea_lab.simulator.base import BaseSimulator
from ab_tea_lab.types import EvaluationResult, ModelResult, SimulationData

# ---------------------------------------------------------------------------
# Dummy simulator
# ---------------------------------------------------------------------------


class DummySimulator(BaseSimulator):
    def __init__(self, n_rows: int = 50, effect: float = 0.1, seed: int | None = None) -> None:
        self.n_rows = n_rows
        self.effect = effect
        self.seed = seed

    def run(self) -> SimulationData:
        rng = random.Random(self.seed)
        rows = []
        for i in range(self.n_rows):
            ctrl_spend = rng.uniform(100, 1000)
            ctrl_convs = ctrl_spend * rng.uniform(0.01, 0.05)
            treat_spend = rng.uniform(100, 1000)
            treat_convs = treat_spend * rng.uniform(0.01, 0.05) * (1 + self.effect)
            rows.append(
                {
                    "adv_id": f"adv_{i % 5}",
                    "cpg_id": f"cpg_{i}",
                    "ctrl_spend_h": 0.0,
                    "ctrl_convs_h": 0.0,
                    "ctrl_spend": ctrl_spend,
                    "ctrl_convs": ctrl_convs,
                    "treat_spend": treat_spend,
                    "treat_convs": treat_convs,
                }
            )
        return SimulationData(data=pd.DataFrame(rows))

    @property
    def true_effect(self) -> float:
        return self.effect


# ---------------------------------------------------------------------------
# Dummy model: computes CPA difference with a naive z-interval
# ---------------------------------------------------------------------------


class DummyModel(BaseModel):
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def fit(self, data: SimulationData) -> Self:
        df = data.data
        ctrl_cpa = (df["ctrl_spend"] / df["ctrl_convs"]).mean()
        treat_cpa = (df["treat_spend"] / df["treat_convs"]).mean()

        diff = (treat_cpa - ctrl_cpa) / ctrl_cpa
        se = abs(diff) * 0.5 + 0.01
        z = 1.96

        self.result = ModelResult(
            effect=diff,
            std_error=se,
            interval_lower=diff - z * se,
            interval_upper=diff + z * se,
            p_value=max(0.001, 1 - abs(diff) / (se + 1e-9)),
            reject_h0=abs(diff) / (se + 1e-9) > z,
        )
        return self


# ---------------------------------------------------------------------------
# Dummy evaluator
# ---------------------------------------------------------------------------


class DummyEvaluator(BaseEvaluator):
    def evaluate(self, results: list[ModelResult], true_effect: float) -> EvaluationResult:
        n = len(results)
        estimates = [r.effect for r in results]

        mean_est = sum(estimates) / n
        bias = mean_est - true_effect
        mse = sum((e - true_effect) ** 2 for e in estimates) / n
        rmse = math.sqrt(mse)
        std_est = math.sqrt(sum((e - mean_est) ** 2 for e in estimates) / n)

        covers = sum(r.interval_lower <= true_effect <= r.interval_upper for r in results)
        coverage = covers / n

        ci_widths = [r.interval_upper - r.interval_lower for r in results]
        mean_ci_width = sum(ci_widths) / n

        rejections = [r.reject_h0 for r in results if r.reject_h0 is not None]
        rejection_rate = sum(1 for r in rejections if r) / len(rejections) if rejections else None
        is_null = true_effect == 0.0

        p_values = [r.p_value for r in results if r.p_value is not None]
        mean_p_value = sum(p_values) / len(p_values) if p_values else None

        posterior_probs = [r.posterior_prob for r in results if r.posterior_prob is not None]
        mean_posterior_prob = sum(posterior_probs) / len(posterior_probs) if posterior_probs else None

        tau2s = [r.tau2 for r in results if r.tau2 is not None]
        mean_tau2 = sum(tau2s) / len(tau2s) if tau2s else None
        std_tau2 = math.sqrt(sum((t - mean_tau2) ** 2 for t in tau2s) / len(tau2s)) if tau2s else None

        return EvaluationResult(
            coverage=coverage,
            bias=bias,
            rmse=rmse,
            mean_estimate=mean_est,
            std_estimate=std_est,
            mean_interval_width=mean_ci_width,
            power=0.0 if is_null else rejection_rate,
            type_i_error=rejection_rate if is_null else 0.0,
            mean_p_value=mean_p_value,
            mean_posterior_prob=mean_posterior_prob,
            mean_tau2=mean_tau2,
            std_tau2=std_tau2,
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_experiment_runs_end_to_end() -> None:
    simulator = DummySimulator(n_rows=30, effect=0.1)
    models = [DummyModel(alpha=0.05)]
    evaluator = DummyEvaluator()

    experiment = Experiment(
        simulator=simulator,
        models=models,
        evaluator=evaluator,
        n_simulations=20,
    )

    results = experiment.run()

    assert len(results) == 1
    name = next(iter(results.keys()))
    assert name == "DummyModel"

    er = results[name]
    assert isinstance(er, EvaluationResult)
    assert 0.0 <= er.coverage <= 1.0
    assert 0.0 <= er.power <= 1.0
    assert er.rmse >= 0.0
    assert er.mean_interval_width > 0.0
    assert er.mean_p_value is None or 0.0 <= er.mean_p_value <= 1.0
    assert er.mean_posterior_prob is None or 0.0 <= er.mean_posterior_prob <= 1.0

    assert experiment.raw_results is not None
    assert len(experiment.raw_results[name]) == 20

    print(f"Coverage:       {er.coverage:.2f}")
    print(f"Bias:           {er.bias:.4f}")
    print(f"RMSE:           {er.rmse:.4f}")
    print(f"Mean estimate:  {er.mean_estimate:.4f}")
    print(f"Std estimate:   {er.std_estimate:.4f}")
    print(f"Mean CI width:  {er.mean_interval_width:.4f}")
    print(f"Power:          {er.power:.2f}")
    print(f"Type I error:   {er.type_i_error:.2f}")
    print(f"Mean p-value:   {er.mean_p_value}")
    print(f"Mean post prob: {er.mean_posterior_prob}")


def test_multiple_models() -> None:
    simulator = DummySimulator(n_rows=30, effect=0.0)
    models = [DummyModel(alpha=0.05), DummyModel(alpha=0.10)]
    evaluator = DummyEvaluator()

    experiment = Experiment(
        simulator=simulator,
        models=models,
        evaluator=evaluator,
        n_simulations=10,
    )

    results = experiment.run()

    assert len(results) == 2
    assert "DummyModel_0" in results
    assert "DummyModel_1" in results

    for name, er in results.items():
        assert isinstance(er, EvaluationResult)
        print(f"\n--- {name} ---")
        print(f"  Type I error: {er.type_i_error:.2f}")
        print(f"  Coverage:     {er.coverage:.2f}")


if __name__ == "__main__":
    print("=== Test: single model, effect=0.1 ===")
    test_experiment_runs_end_to_end()
    print("\n=== Test: two models, null effect ===")
    test_multiple_models()
    print("\nAll tests passed.")

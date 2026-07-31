"""Tests for BayesianPoissonRegression model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ab_tea_lab.model.bayesian_poisson_regression import BayesianPoissonRegression
from ab_tea_lab.types import ModelResult, SimulationData

SAMPLER_KWARGS: dict[str, int | float | bool | str] = dict(
    chains=2,
    draws=500,
    tune=1000,
    target_accept=0.95,
)

MODEL_KWARGS: dict[str, float] = dict(
    sigma_theta=1,
    sigma_tau=1,
)


def _make_data(
    n: int = 20,
    true_effect: float = 0.0,
    seed: int = 42,
) -> SimulationData:
    rng = np.random.RandomState(seed)
    ctrl_spend = rng.uniform(200, 800, size=n)
    base_rate = rng.uniform(0.02, 0.06, size=n)
    ctrl_convs = rng.poisson(ctrl_spend * base_rate)
    treat_spend = rng.uniform(200, 800, size=n)
    treat_convs = rng.poisson(treat_spend * base_rate * np.exp(true_effect))

    df = pd.DataFrame(
        {
            "adv_id": [f"adv_{i % 5}" for i in range(n)],
            "cpg_id": [f"cpg_{i}" for i in range(n)],
            "ctrl_spend_h": 0.0,
            "ctrl_convs_h": 0.0,
            "ctrl_spend": ctrl_spend,
            "ctrl_convs": ctrl_convs,
            "treat_spend": treat_spend,
            "treat_convs": treat_convs,
        }
    )
    return SimulationData(data=df)


@pytest.fixture(scope="module")
def null_model() -> BayesianPoissonRegression:
    """Fit once under H0 and reuse across tests in this module."""
    return BayesianPoissonRegression(**MODEL_KWARGS, **SAMPLER_KWARGS).fit(_make_data(n=20, true_effect=0.0))


@pytest.fixture(scope="module")
def positive_model() -> BayesianPoissonRegression:
    """Fit once under H1 (positive effect) and reuse."""
    return BayesianPoissonRegression(**MODEL_KWARGS, **SAMPLER_KWARGS).fit(_make_data(n=30, true_effect=0.3))


@pytest.fixture(scope="module")
def negative_model() -> BayesianPoissonRegression:
    """Fit once under H1 (negative effect) and reuse."""
    return BayesianPoissonRegression(**MODEL_KWARGS, **SAMPLER_KWARGS).fit(_make_data(n=30, true_effect=-0.3))


class TestBasicFit:
    def test_returns_self(self, null_model: BayesianPoissonRegression) -> None:
        assert isinstance(null_model, BayesianPoissonRegression)

    def test_result_is_model_result(self, null_model: BayesianPoissonRegression) -> None:
        assert isinstance(null_model.result, ModelResult)

    def test_hdi_contains_estimate(self, null_model: BayesianPoissonRegression) -> None:
        r = null_model.result
        assert r.interval_lower <= r.effect <= r.interval_upper

    def test_std_error_positive(self, null_model: BayesianPoissonRegression) -> None:
        assert null_model.result.std_error > 0

    def test_tau2_non_negative(self, null_model: BayesianPoissonRegression) -> None:
        assert null_model.result.tau2 >= 0

    def test_effect_is_finite(self, null_model: BayesianPoissonRegression) -> None:
        assert math.isfinite(null_model.result.effect)

    def test_posterior_prob_in_valid_range(self, null_model: BayesianPoissonRegression) -> None:
        assert 0.0 <= null_model.result.posterior_prob <= 1.0

    def test_no_p_value(self, null_model: BayesianPoissonRegression) -> None:
        assert null_model.result.p_value is None


class TestNullEffect:
    """Under H0 (true_effect=0), the log rate-ratio should be close to 0."""

    def test_null_effect_estimate_near_zero(self, null_model: BayesianPoissonRegression) -> None:
        assert abs(null_model.result.effect) < 0.5

    def test_null_hdi_covers_zero(self, null_model: BayesianPoissonRegression) -> None:
        r = null_model.result
        assert r.interval_lower <= 0.0 <= r.interval_upper

    def test_null_posterior_prob_near_half(self, null_model: BayesianPoissonRegression) -> None:
        assert 0.1 < null_model.result.posterior_prob < 0.9

    def test_null_does_not_reject(self, null_model: BayesianPoissonRegression) -> None:
        assert not null_model.result.reject_h0


class TestNonNullEffect:
    """Under H1, estimate should reflect the direction of the true effect."""

    def test_positive_effect_detected(self, positive_model: BayesianPoissonRegression) -> None:
        assert positive_model.result.effect > 0.0

    def test_positive_posterior_prob_high(self, positive_model: BayesianPoissonRegression) -> None:
        assert positive_model.result.posterior_prob > 0.5

    def test_negative_effect_detected(self, negative_model: BayesianPoissonRegression) -> None:
        assert negative_model.result.effect < 0.0

    def test_negative_posterior_prob_low(self, negative_model: BayesianPoissonRegression) -> None:
        assert negative_model.result.posterior_prob < 0.5


class TestTwoSidedRejection:
    """Two-sided rejection should trigger when posterior mass is extreme in either tail."""

    def test_two_sided_rejects_positive(self) -> None:
        model = BayesianPoissonRegression(two_sided=True, **MODEL_KWARGS, **SAMPLER_KWARGS)
        model.fit(_make_data(n=60, true_effect=0.8, seed=42))
        assert model.result.reject_h0

    def test_two_sided_rejects_negative(self) -> None:
        model = BayesianPoissonRegression(two_sided=True, **MODEL_KWARGS, **SAMPLER_KWARGS)
        model.fit(_make_data(n=60, true_effect=-0.8, seed=42))
        assert model.result.reject_h0


class TestOneSidedRejection:
    """One-sided rejection only detects positive effects (theta > 0)."""

    def test_one_sided_rejects_positive(self) -> None:
        model = BayesianPoissonRegression(two_sided=False, **MODEL_KWARGS, **SAMPLER_KWARGS)
        model.fit(_make_data(n=60, true_effect=0.8, seed=42))
        assert model.result.reject_h0

    def test_one_sided_does_not_reject_negative(self) -> None:
        model = BayesianPoissonRegression(two_sided=False, **MODEL_KWARGS, **SAMPLER_KWARGS)
        model.fit(_make_data(n=60, true_effect=-0.8, seed=42))
        assert not model.result.reject_h0

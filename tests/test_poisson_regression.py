"""Tests for PoissonRegression model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from ab_tea_lab.model.poisson_regression import PoissonRegression
from ab_tea_lab.types import ModelResult, SimulationData

SAMPLER_KWARGS: dict[str, int | float | bool | str] = dict(
    chains=2,
    draws=300,
    tune=500,
    target_accept=0.90,
)


def _make_data(
    n: int = 20,
    true_effect: float = 0.0,
    seed: int = 42,
) -> SimulationData:
    """Generate a simple SimulationData with Poisson-distributed counts."""
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
def null_model() -> PoissonRegression:
    """Fit once under H0 and reuse across tests in this module."""
    return PoissonRegression(**SAMPLER_KWARGS).fit(_make_data(n=20, true_effect=0.0))


@pytest.fixture(scope="module")
def positive_model() -> PoissonRegression:
    """Fit once under H1 (positive effect) and reuse."""
    return PoissonRegression(**SAMPLER_KWARGS).fit(_make_data(n=30, true_effect=0.3))


@pytest.fixture(scope="module")
def negative_model() -> PoissonRegression:
    """Fit once under H1 (negative effect) and reuse."""
    return PoissonRegression(**SAMPLER_KWARGS).fit(_make_data(n=30, true_effect=-0.3))


class TestBasicFit:
    def test_returns_self(self, null_model: PoissonRegression) -> None:
        assert isinstance(null_model, PoissonRegression)

    def test_result_is_model_result(self, null_model: PoissonRegression) -> None:
        assert isinstance(null_model.result, ModelResult)

    def test_ci_contains_estimate(self, null_model: PoissonRegression) -> None:
        r = null_model.result
        assert r.interval_lower <= r.effect <= r.interval_upper

    def test_p_value_in_valid_range(self, null_model: PoissonRegression) -> None:
        assert 0.0 <= null_model.result.p_value <= 1.0

    def test_std_error_positive(self, null_model: PoissonRegression) -> None:
        assert null_model.result.std_error > 0

    def test_tau2_non_negative(self, null_model: PoissonRegression) -> None:
        assert null_model.result.tau2 >= 0

    def test_effect_is_finite(self, null_model: PoissonRegression) -> None:
        assert math.isfinite(null_model.result.effect)


class TestNullEffect:
    """Under H0 (true_effect=0), the log rate-ratio should be close to 0."""

    def test_null_effect_estimate_near_zero(self, null_model: PoissonRegression) -> None:
        assert abs(null_model.result.effect) < 0.5

    def test_null_effect_ci_covers_zero(self, null_model: PoissonRegression) -> None:
        r = null_model.result
        assert r.interval_lower <= 0.0 <= r.interval_upper


class TestNonNullEffect:
    """Under H1, estimate should reflect the direction of the true effect."""

    def test_positive_effect_detected(self, positive_model: PoissonRegression) -> None:
        assert positive_model.result.effect > 0.0

    def test_negative_effect_detected(self, negative_model: PoissonRegression) -> None:
        assert negative_model.result.effect < 0.0


class TestOneSided:
    def test_one_sided_p_value_half_of_two_sided(self, positive_model: PoissonRegression) -> None:
        """For a positive effect, one-sided sf(z) should be half of two-sided 2*sf(|z|)."""
        r = positive_model.result
        z = r.effect / r.std_error
        one_sided = sp_stats.norm.sf(z)
        two_sided = 2 * sp_stats.norm.sf(abs(z))
        assert one_sided < two_sided
        assert one_sided == pytest.approx(two_sided / 2, rel=1e-6)

    def test_one_sided_large_p_when_effect_negative(self, negative_model: PoissonRegression) -> None:
        """sf(z) with z < 0 gives p > 0.5, so one-sided doesn't detect negative effects."""
        r = negative_model.result
        z = r.effect / r.std_error
        one_sided = sp_stats.norm.sf(z)
        assert one_sided > 0.5

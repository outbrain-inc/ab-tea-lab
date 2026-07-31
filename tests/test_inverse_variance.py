"""Tests for InverseVariance model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ab_tea_lab.model.inverse_variance import InverseVariance
from ab_tea_lab.types import ModelResult, SimulationData


def _make_data(
    n: int = 20,
    true_effect: float = 0.0,
    seed: int = 42,
    zero_idx: list[int] | None = None,
) -> SimulationData:
    """Generate a simple SimulationData with Poisson-like counts."""
    rng = np.random.RandomState(seed)
    ctrl_spend = rng.uniform(200, 800, size=n)
    base_rate = rng.uniform(0.02, 0.06, size=n)
    ctrl_convs = ctrl_spend * base_rate
    treat_spend = rng.uniform(200, 800, size=n)
    treat_convs = treat_spend * base_rate * np.exp(true_effect)

    if zero_idx is not None:
        for i in zero_idx:
            ctrl_convs[i] = 0.0

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


class TestBasicFit:
    def test_returns_self(self) -> None:
        model = InverseVariance()
        data = _make_data()
        assert model.fit(data) is model

    def test_result_is_model_result(self) -> None:
        model = InverseVariance().fit(_make_data())
        assert isinstance(model.result, ModelResult)

    def test_ci_contains_estimate(self) -> None:
        model = InverseVariance().fit(_make_data())
        r = model.result
        assert r.interval_lower <= r.effect <= r.interval_upper

    def test_p_value_in_valid_range(self) -> None:
        model = InverseVariance().fit(_make_data())
        assert 0.0 <= model.result.p_value <= 1.0

    def test_std_error_positive(self) -> None:
        model = InverseVariance().fit(_make_data())
        assert model.result.std_error > 0


class TestNullEffect:
    """Under H0 (true_effect=0), the model should generally not reject."""

    def test_null_effect_estimate_near_zero(self) -> None:
        model = InverseVariance().fit(_make_data(n=50, true_effect=0.0))
        assert abs(model.result.effect) < 0.5

    def test_null_effect_ci_covers_zero(self) -> None:
        model = InverseVariance().fit(_make_data(n=50, true_effect=0.0))
        r = model.result
        assert r.interval_lower <= 0 <= r.interval_upper


class TestNonNullEffect:
    """Under H1 (true_effect != 0), estimate should be in the right direction."""

    def test_negative_effect_detected(self) -> None:
        model = InverseVariance().fit(_make_data(n=50, true_effect=-0.3))
        assert model.result.effect < 0

    def test_positive_effect_detected(self) -> None:
        model = InverseVariance().fit(_make_data(n=50, true_effect=0.3))
        assert model.result.effect > 0


class TestOneSided:
    def test_one_sided_p_value_smaller_when_effect_positive(self) -> None:
        data = _make_data(n=50, true_effect=0.3)
        one_sided = InverseVariance(two_sided=False).fit(data).result.p_value
        two_sided = InverseVariance(two_sided=True).fit(data).result.p_value
        assert one_sided < two_sided

    def test_one_sided_large_p_when_effect_negative(self) -> None:
        data = _make_data(n=50, true_effect=-0.3)
        model = InverseVariance(two_sided=False).fit(data)
        assert model.result.p_value > 0.5


class TestRandomEffects:
    def test_re_iterated_produces_tau2(self) -> None:
        model = InverseVariance(method_re="iterated").fit(_make_data())
        assert model.result.tau2 is not None
        assert model.result.tau2 >= 0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_re_chi2_produces_tau2(self) -> None:
        model = InverseVariance(method_re="chi2").fit(_make_data())
        assert model.result.tau2 is not None

    def test_fe_has_zero_tau2(self) -> None:
        model = InverseVariance(method_re=None).fit(_make_data())
        assert model.result.tau2 == 0.0


class TestZeroCorrection:
    def test_handles_zero_counts(self) -> None:
        data = _make_data(zero_idx=[0, 1])
        model = InverseVariance(zero_correction=0.5).fit(data)
        r = model.result
        assert math.isfinite(r.effect)
        assert math.isfinite(r.std_error)
        assert math.isfinite(r.p_value)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_no_correction_silently_drops_zero_study(self) -> None:
        """Without correction, zero-count studies get infinite variance and
        zero weight in combine_effects, so the combined result differs from
        the corrected version (which includes the study)."""
        data = _make_data(zero_idx=[0])
        r_with = InverseVariance(zero_correction=0.5).fit(data).result
        r_without = InverseVariance(zero_correction=0).fit(data).result
        assert math.isfinite(r_without.effect)
        assert r_with.effect != pytest.approx(r_without.effect)

    def test_correction_only_affects_zero_studies(self) -> None:
        data_clean = _make_data(n=10, zero_idx=None)
        r1 = InverseVariance(zero_correction=0.0).fit(data_clean).result
        r2 = InverseVariance(zero_correction=0.5).fit(data_clean).result
        assert r1.effect == pytest.approx(r2.effect)
        assert r1.std_error == pytest.approx(r2.std_error)

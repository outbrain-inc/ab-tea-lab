import numpy as np
import pandas as pd
import pytest

from ab_tea_lab.simulator.random import RandomSimulator
from ab_tea_lab.types import SimulationData


def _make_sim(seed: int = 123) -> RandomSimulator:
    return RandomSimulator(
        num_advertisers=3,
        campaigns_mean=2.5,
        spend_mean=70.0,
        spend_sd=55.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=0.05,
        effect_sd=0.1,
        equal_spend=False,
        random_seed=seed,
    )


def test_random_simulator_returns_simulationdata_with_required_columns() -> None:
    sim = _make_sim()
    out = sim.run()
    assert isinstance(out, SimulationData)
    df = out.data
    # Required columns present
    for col in SimulationData.REQUIRED_COLUMNS:
        assert col in df.columns
    # Types and basic sanity
    assert (df["ctrl_spend"] >= 0).all()
    assert (df["treat_spend"] >= 0).all()
    assert pd.api.types.is_integer_dtype(df["ctrl_convs"])
    assert pd.api.types.is_integer_dtype(df["treat_convs"])
    assert (df["ctrl_convs"] >= 0).all()
    assert (df["treat_convs"] >= 0).all()
    # At least one advertiser and one campaign
    assert df["adv_id"].nunique() >= 1
    assert df["cpg_id"].nunique() >= 1


def test_random_simulator_reproducible_with_seed() -> None:
    sim1 = _make_sim(seed=42)
    sim2 = _make_sim(seed=42)
    df1 = sim1.run().data
    df2 = sim2.run().data
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize(
    "param_name,kwargs",
    [
        ("num_advertisers", dict(num_advertisers=0)),
        ("campaigns_mean", dict(campaigns_mean=0.0)),
        ("spend_sd", dict(spend_sd=0.0)),
        ("cpa_mean", dict(cpa_mean=0.0)),
        ("cpa_sd", dict(cpa_sd=0.0)),
        ("cpa_within_sd", dict(cpa_within_sd=0.0)),
        ("effect_sd", dict(effect_sd=-0.1)),
        (
            "conversions_nb_dispersion_missing",
            dict(conversions_distribution="negative_binomial", conversions_nb_dispersion=None),
        ),
        (
            "conversions_nb_dispersion_nonpositive",
            dict(conversions_distribution="negative_binomial", conversions_nb_dispersion=0.0),
        ),
    ],
)
def test_random_simulator_validates_parameters(param_name: str, kwargs: dict[str, object]) -> None:
    base = _make_sim()
    bad = base.__class__(**{**base.__dict__ | {"_rng": None}, **kwargs})  # reset rng for a fair run
    with pytest.raises(ValueError), np.errstate(all="ignore"):
        bad.run()


def test_random_simulator_true_effect_matches_effect_mean() -> None:
    sim = _make_sim()
    assert sim.true_effect == pytest.approx(0.05)


def test_random_simulator_negative_binomial_runs() -> None:
    sim = RandomSimulator(
        num_advertisers=3,
        campaigns_mean=2.5,
        spend_mean=70.0,
        spend_sd=55.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=0.05,
        effect_sd=0.1,
        equal_spend=False,
        random_seed=7,
        conversions_distribution="negative_binomial",
        conversions_nb_dispersion=5.0,
    )
    out = sim.run()
    df = out.data
    assert pd.api.types.is_integer_dtype(df["ctrl_convs"])
    assert (df["ctrl_convs"] >= 0).all()
    assert (df["ctrl_convs_h"] >= 0).all()
    assert (df["treat_convs"] >= 0).all()


def test_random_simulator_log_t_runs() -> None:
    sim = RandomSimulator(
        num_advertisers=3,
        campaigns_mean=2.5,
        spend_mean=70.0,
        spend_sd=55.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=0.05,
        effect_sd=0.1,
        equal_spend=False,
        random_seed=42,
        cpa_distribution="log_t",
        cpa_median=50.0,
        cpa_iqr=30.0,
        cpa_within_iqr=10.0,
        cpa_tail_df=3.0,
    )
    out = sim.run()
    df = out.data
    assert pd.api.types.is_integer_dtype(df["ctrl_convs"])
    assert (df["ctrl_convs"] >= 0).all()
    assert (df["ctrl_convs_h"] >= 0).all()
    assert (df["treat_convs"] >= 0).all()
    assert df["adv_id"].nunique() >= 1


def test_random_simulator_log_t_reproducible_with_seed() -> None:
    kwargs = dict(
        num_advertisers=2,
        campaigns_mean=2.0,
        spend_mean=23.0,
        spend_sd=12.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=0.0,
        effect_sd=0.05,
        equal_spend=True,
        random_seed=99,
        cpa_distribution="log_t",
        cpa_median=40.0,
        cpa_iqr=20.0,
        cpa_within_iqr=8.0,
        cpa_tail_df=4.0,
    )
    df1 = RandomSimulator(**kwargs).run().data
    df2 = RandomSimulator(**kwargs).run().data
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


def test_random_simulator_quantile_runs() -> None:
    sim = RandomSimulator(
        num_advertisers=3,
        campaigns_mean=2.5,
        spend_mean=70.0,
        spend_sd=55.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=0.05,
        effect_sd=0.1,
        equal_spend=False,
        random_seed=42,
        cpa_distribution="quantile",
        cpa_quantiles=((0.25, 1.26), (0.50, 3.78), (0.75, 49.11)),
        cpa_within_iqr=5.0,
    )
    out = sim.run()
    df = out.data
    assert pd.api.types.is_integer_dtype(df["ctrl_convs"])
    assert (df["ctrl_convs"] >= 0).all()
    assert (df["ctrl_convs_h"] >= 0).all()
    assert (df["treat_convs"] >= 0).all()
    assert df["adv_id"].nunique() >= 1


def test_random_simulator_quantile_reproducible_with_seed() -> None:
    kwargs = dict(
        num_advertisers=2,
        campaigns_mean=2.0,
        spend_mean=23.0,
        spend_sd=12.0,
        cpa_mean=50.0,
        cpa_sd=20.0,
        cpa_within_sd=10.0,
        effect_mean=0.0,
        effect_sd=0.05,
        equal_spend=True,
        random_seed=99,
        cpa_distribution="quantile",
        cpa_quantiles=((0.10, 0.5), (0.25, 1.26), (0.50, 3.78), (0.75, 49.11), (0.90, 200.0)),
        cpa_within_iqr=5.0,
    )
    df1 = RandomSimulator(**kwargs).run().data
    df2 = RandomSimulator(**kwargs).run().data
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


def test_random_simulator_negative_binomial_reproducible_with_seed() -> None:
    kwargs = dict(
        num_advertisers=2,
        campaigns_mean=2.0,
        spend_mean=23.0,
        spend_sd=12.0,
        cpa_mean=40.0,
        cpa_sd=15.0,
        cpa_within_sd=8.0,
        effect_mean=0.0,
        effect_sd=0.05,
        equal_spend=True,
        random_seed=99,
        conversions_distribution="negative_binomial",
        conversions_nb_dispersion=3.0,
    )
    df1 = RandomSimulator(**kwargs).run().data
    df2 = RandomSimulator(**kwargs).run().data
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)

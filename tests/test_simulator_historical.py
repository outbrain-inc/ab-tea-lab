import numpy as np
import pandas as pd
import pytest

from ab_tea_lab.simulator.historical import FutureFromHistoricalSimulator
from ab_tea_lab.types import SimulationData


def _hist_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "adv_id": np.arange(n, dtype=int),
            "cpg_id": np.arange(n, dtype=int),
            "hist_spend": rng.lognormal(mean=4.0, sigma=0.5, size=n),
            "hist_convs": rng.poisson(lam=10.0, size=n).astype(int) + 1,  # avoid zeros
            "yesterday_spend": rng.lognormal(mean=4.0, sigma=0.5, size=n),
        }
    )


def _make_sim(days: int = 3, seed: int = 123) -> FutureFromHistoricalSimulator:
    return FutureFromHistoricalSimulator(
        historical_df=_hist_df(),
        spend_col="hist_spend",
        convs_col="hist_convs",
        yesterday_spend_col="yesterday_spend",
        adv_id_col="adv_id",
        cpg_id_col="cpg_id",
        effect_mean=0.05,
        effect_sd=0.10,
        days=days,
        random_seed=seed,
    )


def test_historical_simulator_outputs_expected_columns_and_values() -> None:
    sim = _make_sim(days=2)
    out = sim.run()
    assert isinstance(out, SimulationData)
    df = out.data
    # Required columns present
    for col in SimulationData.REQUIRED_COLUMNS:
        assert col in df.columns
    # Spend expectations: ctrl and treat spend equal yesterday_spend * days
    hist = _hist_df()
    expected_spend = (hist["yesterday_spend"].astype(float) * 2.0).to_numpy()
    assert np.allclose(np.sort(df["ctrl_spend"].to_numpy()), np.sort(expected_spend))
    assert np.allclose(df["ctrl_spend"].to_numpy(), df["treat_spend"].to_numpy())
    # Conversions are non-negative integers
    assert (df["ctrl_convs"] >= 0).all() and (df["treat_convs"] >= 0).all()
    assert pd.api.types.is_integer_dtype(df["ctrl_convs"])
    assert pd.api.types.is_integer_dtype(df["treat_convs"])
    # IDs preserved
    assert set(df["adv_id"]) == set(hist["adv_id"])
    assert set(df["cpg_id"]) == set(hist["cpg_id"])


def test_historical_simulator_reproducible_with_seed() -> None:
    sim1 = _make_sim(days=1, seed=7)
    sim2 = _make_sim(days=1, seed=7)
    df1 = sim1.run().data
    df2 = sim2.run().data
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


def test_historical_simulator_validates_inputs() -> None:
    # Missing column
    bad_df = _hist_df().drop(columns=["hist_convs"])
    sim = FutureFromHistoricalSimulator(historical_df=bad_df)
    with pytest.raises(ValueError):
        sim.run()
    # Negative effect_sd
    sim_bad_sd = FutureFromHistoricalSimulator(historical_df=_hist_df(), effect_sd=-0.1)
    with pytest.raises(ValueError):
        sim_bad_sd.run()
    # Invalid days
    sim_bad_days = FutureFromHistoricalSimulator(historical_df=_hist_df(), days=0)
    with pytest.raises(ValueError):
        sim_bad_days.run()


def test_historical_simulator_true_effect_matches_effect_mean() -> None:
    sim = _make_sim()
    assert sim.true_effect == pytest.approx(0.05)

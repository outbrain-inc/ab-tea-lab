import pandas as pd
import pytest

from ab_tea_lab.simulator.concat import ConcatSimulator
from ab_tea_lab.simulator.random import RandomSimulator
from ab_tea_lab.types import SimulationData


def _rand_sim(effect_mean: float = 0.0, seed: int = 1) -> RandomSimulator:
    return RandomSimulator(
        num_advertisers=2,
        campaigns_mean=2.0,
        spend_mean=40.0,
        spend_sd=26.0,
        cpa_mean=40.0,
        cpa_sd=15.0,
        cpa_within_sd=8.0,
        effect_mean=effect_mean,
        effect_sd=0.05,
        equal_spend=True,
        random_seed=seed,
    )


def test_concat_simulator_concatenates_and_remaps_ids() -> None:
    sim_a = _rand_sim(effect_mean=0.1, seed=10)
    sim_b = _rand_sim(effect_mean=0.1, seed=11)
    concat = ConcatSimulator([sim_a, sim_b])
    out = concat.run()
    assert isinstance(out, SimulationData)
    df = out.data
    for col in SimulationData.REQUIRED_COLUMNS:
        assert col in df.columns
    # IDs should be integers and globally unique
    assert pd.api.types.is_integer_dtype(df["adv_id"])
    assert pd.api.types.is_integer_dtype(df["cpg_id"])
    # After remapping, adv_id and cpg_id should form contiguous integer ranges starting at 0
    adv_ids = sorted(pd.unique(df["adv_id"]))
    cpg_ids = sorted(pd.unique(df["cpg_id"]))
    assert adv_ids == list(range(len(adv_ids)))
    assert cpg_ids == list(range(len(cpg_ids)))
    # True effect is preserved and consistent
    assert concat.true_effect == pytest.approx(0.1)


def test_concat_simulator_requires_at_least_one() -> None:
    with pytest.raises(ValueError):
        ConcatSimulator([]).run()


def test_concat_simulator_rejects_inconsistent_effects() -> None:
    sim_a = _rand_sim(effect_mean=0.0, seed=1)
    sim_b = _rand_sim(effect_mean=0.2, seed=2)
    concat = ConcatSimulator([sim_a, sim_b])
    with pytest.raises(ValueError):
        concat.run()
    with pytest.raises(ValueError):
        # Property also checks consistency
        _ = concat.true_effect

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ab_tea_lab.simulator.base import BaseSimulator
from ab_tea_lab.types import SimulationData


@dataclass
class FutureFromHistoricalSimulator(BaseSimulator):
    """Generate future-period experiment data from historical aggregates.

    Expects a historical DataFrame with at least advertiser and campaign IDs and
    spend/conversions columns. It uses the historical CPA (spend per conversion)
    as a base, applies a log-normal multiplicative treatment effect to CPA, and
    samples future conversions given spend and CPA.
    Future spend is assumed to be equal to yesterday's spend times ``days`` (no growth).

    Parameters
    ----------
    historical_df:
        Pandas DataFrame with one row per (adv_id, cpg_id) historical aggregate.
    spend_col:
        Column name in historical_df for historical spend.
    convs_col:
        Column name in historical_df for historical conversions (for CPA calculation).
    yesterday_spend_col:
        Column name in historical_df for yesterday's spend (used for future spend).
    adv_id_col:
        Column name in historical_df for advertiser identifier.
    cpg_id_col:
        Column name in historical_df for campaign identifier.
    effect_mean:
        Location parameter ``mu`` of the underlying Normal used by the log-normal
        multiplicative factor applied to CPA (i.e., effect ~ LogNormal(mu, sigma)).
        Since this factor multiplies CPA directly, values > 1 increase CPA
        (lower implied conversion rate under treatment), and values < 1 decrease it.
    effect_sd:
        Scale parameter ``sigma`` of the underlying Normal used by the log-normal
        multiplicative factor applied to CPA. Must be >= 0; ``0`` yields a
        degenerate factor of ``exp(mu)``.
    days:
        Number of future days to simulate as a single aggregated period (>= 1).
    equal_spend:
        Kept for API compatibility; treatment spend equals control spend because
        future spend is fixed to ``yesterday_spend * days`` for both arms.
    random_seed:
        Optional RNG seed for reproducibility.
    """

    historical_df: pd.DataFrame
    spend_col: str = "hist_spend"
    convs_col: str = "hist_convs"
    yesterday_spend_col: str = "yesterday_spend"
    adv_id_col: str = "adv_id"
    cpg_id_col: str = "cpg_id"

    effect_mean: float = 0.0
    effect_sd: float = 0.02
    days: int = 1

    equal_spend: bool = False
    random_seed: int | None = None

    _rng: np.random.Generator | None = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_seed)
        return self._rng

    def run(self) -> SimulationData:
        rng = self._get_rng()

        required = {self.adv_id_col, self.cpg_id_col, self.spend_col, self.convs_col, self.yesterday_spend_col}
        missing = required - set(self.historical_df.columns)
        if missing:
            raise ValueError(f"historical_df missing required columns: {missing}")
        if self.effect_sd < 0:
            raise ValueError("effect_sd must be >= 0")
        if self.days < 1:
            raise ValueError("days must be >= 1")

        dfh = self.historical_df[
            [self.adv_id_col, self.cpg_id_col, self.spend_col, self.convs_col, self.yesterday_spend_col]
        ].copy()

        eps = 1e-9
        # Base CPA = spend / conversions (guard conversions from zero)
        base_cpa = (dfh[self.spend_col] / np.clip(dfh[self.convs_col], eps, None)).to_numpy()

        # Future-period control spend equals yesterday's spend * days (no growth)
        ctrl_spend = dfh[self.yesterday_spend_col].to_numpy().astype(float) * float(self.days)

        # Treatment CPA via log-normal effect; enforce non-negative CPA
        # Sample log-normal effect (multiplicative on base_cpa)
        # Note: log-normal location is set to self.effect_mean, scale to self.effect_sd
        effect = rng.lognormal(mean=self.effect_mean, sigma=self.effect_sd, size=len(dfh)).astype(float)
        treat_cpa = np.clip(base_cpa * effect, eps, None)

        # Future-period treatment spend equals control spend
        treat_spend = ctrl_spend.copy()

        # Conversions as Poisson with rate = spend / cpa (implicitly scales with days via spend)
        ctrl_lambda = np.clip(ctrl_spend / np.clip(base_cpa, eps, None), 0.0, None)
        treat_lambda = np.clip(treat_spend / np.clip(treat_cpa, eps, None), 0.0, None)
        ctrl_convs = rng.poisson(lam=ctrl_lambda).astype(int)
        treat_convs = rng.poisson(lam=treat_lambda).astype(int)

        out = pd.DataFrame(
            {
                "adv_id": dfh[self.adv_id_col].to_numpy(),
                "cpg_id": dfh[self.cpg_id_col].to_numpy(),
                "ctrl_spend_h": dfh[self.spend_col].to_numpy().astype(float),
                "ctrl_convs_h": dfh[self.convs_col].to_numpy().astype(int),
                "ctrl_spend": ctrl_spend,
                "ctrl_convs": ctrl_convs,
                "treat_spend": treat_spend,
                "treat_convs": treat_convs,
            }
        )

        return SimulationData(out)

    @property
    def true_effect(self) -> float:
        """Return the log-normal location parameter ``mu`` used for CPA scaling."""
        return float(self.effect_mean)

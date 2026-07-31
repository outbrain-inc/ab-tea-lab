from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from ab_tea_lab.simulator.base import BaseSimulator
from ab_tea_lab.types import SimulationData


@dataclass
class ConcatSimulator(BaseSimulator):
    """Run multiple simulators independently and concatenate their outputs.

    - Each simulator is executed via ``run()``.
    - The resulting datasets are concatenated row-wise.
    - ``adv_id`` and ``cpg_id`` are remapped by offsetting with the current
      max ID so far (i.e., ``df[id_col] += max_prev_id + 1``) to avoid
      collisions and ensure uniqueness across the concatenated dataset.
    - The ground-truth effect returned by ``true_effect`` is enforced to be the
      same across all component simulators; otherwise, an error is raised.
    """

    simulators: Sequence[BaseSimulator]

    def run(self) -> SimulationData:
        if not self.simulators:
            raise ValueError("ConcatSimulator requires at least one simulator.")

        dfs: list[pd.DataFrame] = []

        adv_offset = 0
        cpg_offset = 0

        effect_ref: float | None = None

        for sim in self.simulators:
            # Validate consistent true effect across simulators
            try:
                current_effect = float(sim.true_effect)
            except Exception as exc:
                raise ValueError("All simulators must expose a numeric true_effect.") from exc
            if effect_ref is None:
                effect_ref = current_effect
            elif current_effect != effect_ref:
                raise ValueError(f"Inconsistent true_effect across simulators: {current_effect} vs {effect_ref}")

            sim_data = sim.run()
            df = sim_data.data.copy()

            # Remap advertiser IDs by offset to ensure uniqueness
            df["adv_id"] = pd.Series(df["adv_id"]).astype(int) + int(adv_offset)
            adv_offset = int(pd.Series(df["adv_id"]).max()) + 1

            # Remap campaign IDs by offset to ensure uniqueness
            df["cpg_id"] = pd.Series(df["cpg_id"]).astype(int) + int(cpg_offset)
            cpg_offset = int(pd.Series(df["cpg_id"]).max()) + 1

            # Ensure column order and types match SimulationData expectations
            df = df.loc[:, SimulationData.REQUIRED_COLUMNS]
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return SimulationData(combined)

    @property
    def true_effect(self) -> float:
        effect = self.simulators[0].true_effect
        for sim in self.simulators[1:]:
            if sim.true_effect != effect:
                raise ValueError(f"Inconsistent true_effect across simulators: {sim.true_effect} vs {effect}")
        return float(effect)

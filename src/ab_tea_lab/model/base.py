import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Self

import numpy as np

from ab_tea_lab._params import ParamsMixin
from ab_tea_lab.types import ModelResult, SimulationData


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalars to native Python types."""

    def default(self, o: object) -> int | float | bool | list | str:  # type: ignore[override]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class BaseModel(ParamsMixin, ABC):
    """Abstract base class for statistical methods.

    Subclasses must implement ``fit`` which runs the statistical analysis on a
    single simulated dataset and stores the outcome in ``self.result``.

    Note: ``fit`` returns *self* and the constructor accepts all configurable
    parameters as keyword arguments.
    """

    @abstractmethod
    def fit(self, data: SimulationData) -> Self:
        """Run the statistical analysis on *data* and store the outcome.

        After a successful call the ``result`` attribute must hold a :class:`ModelResult`.
        """

    @property
    def result(self) -> ModelResult:
        try:
            return self._result
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__}.fit() must be called before accessing 'result'") from None

    @result.setter
    def result(self, value: ModelResult) -> None:
        if not isinstance(value, ModelResult):
            raise TypeError(f"result must be a ModelResult, got {type(value).__name__}")
        self._result = value

    def save_result(self, path: str | Path) -> None:
        """Serialize the :class:`ModelResult` to a JSON file.

        Parameters
        ----------
        path:
            Destination file path. Parent directories must already exist.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self.result), f, indent=2, cls=_NumpyEncoder)

import logging
from typing import Self

import arviz as az
import numpy as np
import pymc as pm
from scipy import stats as sp_stats

from ab_tea_lab.model.base import BaseModel
from ab_tea_lab.types import ModelResult, SimulationData


class PoissonRegression(BaseModel):
    """Poisson regression model for meta-analysis of conversion rates.

    Parameters
    ----------
    alpha : float, optional
        Significance level for confidence intervals and p-values (default 0.05).
    two_sided : bool, optional
        Whether to compute two-sided p-values (default True).
    sigma_alpha : float, optional
        Standard deviation of the campaign-level effect (default 10).
    sigma_theta : float, optional
        Standard deviation of the treatment effect (default 10).
    sigma_tau : float, optional
        Standard deviation of the random effects variance (default 10).
    **kwargs : dict, optional
        Additional keyword arguments passed to the PyMC sampler.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        two_sided: bool = True,
        sigma_alpha: float = 10,
        sigma_theta: float = 10,
        sigma_tau: float = 10,
        **kwargs: int | float | bool | str,
    ) -> None:
        self.alpha = alpha
        self.two_sided = two_sided
        self.sigma_alpha = sigma_alpha
        self.sigma_theta = sigma_theta
        self.sigma_tau = sigma_tau
        self.kwargs = kwargs

    def fit(self, data: SimulationData) -> Self:
        # Data preparation
        df = data.data
        n = len(df)

        spend = np.concatenate([df["ctrl_spend"].values, df["treat_spend"].values])
        convs = np.concatenate([df["ctrl_convs"].values, df["treat_convs"].values])
        z_idx = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        cpg_idx = np.concatenate([df["cpg_id"].factorize()[0]] * 2)

        # Build PyMC model
        model = self.build_pymc_model(spend, convs, z_idx, cpg_idx, n)

        # Sample from model
        sample_defaults = dict(
            chains=2,
            draws=500,
            tune=1000,
            target_accept=0.95,
            progressbar=False,
        )
        sample_defaults.update(self.kwargs)

        # pymc logs "NUTS[nutpie]: [...]" at INFO level regardless of progressbar
        logging.getLogger("pymc").setLevel(logging.WARNING)

        with model:
            trace = pm.sample(
                nuts_sampler="nutpie",
                var_names=["theta", "tau"],
                **sample_defaults,
            )

        # Compute model result
        self.result = self.compute_model_result(trace)

        return self

    def build_pymc_model(
        self,
        spend_i: np.ndarray,
        convs_i: np.ndarray,
        z_i: np.ndarray,
        cpg_i: np.ndarray,
        n: int,
    ) -> pm.Model:
        with pm.Model() as model:
            # Campaign-level effect
            alpha_i = pm.Normal("alpha_i", mu=0, sigma=self.sigma_alpha, shape=n)

            # Random treatment effect
            theta = pm.Normal("theta", mu=0, sigma=self.sigma_theta)
            tau = pm.HalfNormal("tau", sigma=self.sigma_tau)
            theta_raw_i = pm.Normal("theta_raw_i", mu=0, sigma=1, shape=n)
            theta_i = theta + tau * theta_raw_i

            # Expected number of conversions
            mu_convs = pm.math.exp(alpha_i[cpg_i] + z_i * theta_i[cpg_i]) * spend_i
            pm.Poisson("convs_i", mu=mu_convs, observed=convs_i)

        return model

    def compute_model_result(self, trace: az.InferenceData) -> ModelResult:
        theta_samples = trace.posterior["theta"].values.flatten()

        # Log rate-ratio scale
        eff = np.mean(theta_samples)
        se = np.std(theta_samples)
        ci = np.quantile(theta_samples, [self.alpha / 2, 1 - self.alpha / 2])

        z = eff / se
        p_value = sp_stats.norm.sf(z) if not self.two_sided else 2 * sp_stats.norm.sf(abs(z))

        # Random effects variance
        tau2 = np.mean(trace.posterior["tau"].values.flatten() ** 2)

        return ModelResult(
            effect=eff,
            std_error=se,
            interval_lower=ci[0],
            interval_upper=ci[1],
            p_value=p_value,
            reject_h0=bool(p_value < self.alpha),
            tau2=tau2,
            metadata=dict(
                alpha=self.alpha,
                two_sided=self.two_sided,
                sigma_alpha=self.sigma_alpha,
                sigma_theta=self.sigma_theta,
                sigma_tau=self.sigma_tau,
                **self.kwargs,
            ),
        )

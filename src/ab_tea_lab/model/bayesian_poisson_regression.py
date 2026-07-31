import arviz as az
import numpy as np
import pymc as pm

from ab_tea_lab.model.poisson_regression import PoissonRegression
from ab_tea_lab.types import ModelResult


class BayesianPoissonRegression(PoissonRegression):
    """Bayesian Poisson regression model for meta-analysis of conversion rates.

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
        mu_mu_alpha: float = 0.1,
        sigma_theta: float = 10,
        sigma_tau: float = 10,
        sigma_mu_alpha: float = 10,
        sigma_sigma_alpha: float = 10,
        **kwargs: int | float | bool | str,
    ) -> None:
        self.alpha = alpha
        self.two_sided = two_sided
        self.mu_mu_alpha = mu_mu_alpha
        self.sigma_theta = sigma_theta
        self.sigma_tau = sigma_tau
        self.sigma_mu_alpha = sigma_mu_alpha
        self.sigma_sigma_alpha = sigma_sigma_alpha
        self.kwargs = kwargs

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
            mu_alpha = pm.Normal("mu_alpha", mu=self.mu_mu_alpha, sigma=self.sigma_mu_alpha)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=self.sigma_sigma_alpha)
            alpha_raw_i = pm.Normal("alpha_raw_i", 0, 1, shape=n)
            alpha_i = mu_alpha + sigma_alpha * alpha_raw_i

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

        hdi = az.hdi(theta_samples, prob=1 - self.alpha)

        # P(theta > 0) is equivalent to P(RR > 1)
        posterior_prob = np.mean(theta_samples > 0)

        if self.two_sided:
            reject_h0 = bool(posterior_prob > 1 - self.alpha / 2 or posterior_prob < self.alpha / 2)
        else:
            reject_h0 = bool(posterior_prob > 1 - self.alpha)

        # Random effects variance
        tau2 = np.mean(trace.posterior["tau"].values.flatten() ** 2)

        return ModelResult(
            effect=eff,
            std_error=se,
            interval_lower=hdi[0],
            interval_upper=hdi[1],
            posterior_prob=posterior_prob,
            reject_h0=reject_h0,
            tau2=tau2,
            metadata=dict(
                alpha=self.alpha,
                two_sided=self.two_sided,
                mu_mu_alpha=self.mu_mu_alpha,
                sigma_theta=self.sigma_theta,
                sigma_tau=self.sigma_tau,
                sigma_mu_alpha=self.sigma_mu_alpha,
                sigma_sigma_alpha=self.sigma_sigma_alpha,
                **self.kwargs,
            ),
        )

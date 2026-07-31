from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import t as t_dist

from ab_tea_lab.simulator.base import BaseSimulator
from ab_tea_lab.types import SimulationData


@dataclass
class RandomSimulator(BaseSimulator):
    """Random data generator for advertiser-campaign simulations.

    CPA is generated with a hierarchical (two-level) model:

    1. **Advertiser level** - each advertiser's CPA is drawn from one of three
       distribution families.
    2. **Campaign level** - each campaign's base CPA is drawn from the same
       family, centred on the advertiser's CPA.

    The ``cpa_distribution`` parameter selects the family:

    * ``"gamma"`` -- Gamma (exponential tails), parameterized by
      ``cpa_mean`` / ``cpa_sd`` / ``cpa_within_sd``.
    * ``"pareto"`` -- Pareto Type I (power-law tails), same parameters.
    * ``"log_t"`` -- Log-Student-t (heavier-than-power-law tails,
      **no finite moments**), parameterized by ``cpa_median`` /
      ``cpa_iqr`` / ``cpa_within_iqr`` / ``cpa_tail_df``.
    * ``"quantile"`` -- Spline inverse-CDF defined by explicit
      quantile-value pairs (``cpa_quantiles``). Matches the given
      quantiles exactly and interpolates between them. Campaign-level
      spread is controlled by ``cpa_within_iqr`` (multiplicative
      lognormal noise around the advertiser CPA).

    By default, conversion counts are sampled from a Poisson distribution with mean ``spend * CVR-UC``.
    Set ``conversions_distribution="negative_binomial"`` to sample instead from a negative binomial
    distribution with the same mean and controlled variance (using ``conversions_nb_dispersion``).

    Parameters
    ----------
    num_advertisers:
        Number of advertisers to simulate.
    campaigns_mean:
        Mean of the exponential distribution that samples the number of
        campaigns per advertiser. Rounded down to an integer with a minimum of 1.
    spend_mean:
        Mean of the lognormal spend distribution (in real space).
    spend_sd:
        Standard deviation of the lognormal spend distribution (in real
        space). Must be > 0.
    cpa_mean:
        Mean of the **advertiser-level** CPA distribution.
        Used by ``"gamma"`` and ``"pareto"``; ignored by ``"log_t"``.
    cpa_sd:
        Standard deviation of the **advertiser-level** CPA distribution.
        Used by ``"gamma"`` and ``"pareto"``; ignored by ``"log_t"``.
    cpa_within_sd:
        Standard deviation of the **campaign-level** CPA distribution
        (within an advertiser). Controls how much individual campaign CPAs
        scatter around their advertiser's mean CPA.
        Used by ``"gamma"`` and ``"pareto"``; ignored by ``"log_t"``.
    cpa_distribution:
        Family used for CPA sampling: ``"gamma"``, ``"pareto"``,
        ``"log_t"``, or ``"quantile"``. Defaults to ``"gamma"``.
    cpa_quantiles:
        Sequence of ``(probability, value)`` pairs defining the
        advertiser-level CPA distribution via its quantile function.
        Must be sorted by probability, with probabilities in ``(0, 1)``
        and values ``> 0``. At least two pairs are required.
        Required when ``cpa_distribution="quantile"``; ignored otherwise.
    cpa_median:
        Median of the **advertiser-level** CPA distribution.
        Required when ``cpa_distribution="log_t"``; ignored otherwise.
    cpa_iqr:
        Inter-quartile range of the **advertiser-level** CPA distribution.
        Required when ``cpa_distribution="log_t"``; ignored otherwise.
    cpa_within_iqr:
        Inter-quartile range of the **campaign-level** CPA distribution
        (within an advertiser). The median at this level equals the
        sampled advertiser CPA. For ``"quantile"`` this controls a
        multiplicative lognormal noise around the advertiser CPA.
        Required when ``cpa_distribution`` is ``"log_t"`` or
        ``"quantile"``; ignored otherwise.
    cpa_tail_df:
        Degrees of freedom ``v`` for the underlying Student-t in the
        log-t distribution. Lower values give heavier tails (``v = 1``
        is log-Cauchy, ``v -> inf`` recovers lognormal).
        Required when ``cpa_distribution="log_t"``; ignored otherwise.
    effect_mean:
        Location parameter ``mu`` of the underlying Normal used by the log-normal
        multiplicative factor applied to CVR-UC = 1 / CPA (i.e., effect ~ LogNormal(mu, sigma)).
    effect_sd:
        Scale parameter ``sigma`` of the underlying Normal used by the log-normal
        multiplicative factor applied to CVR-UC = 1 / CPA. Must be >= 0; ``0`` yields a
        degenerate factor of ``exp(mu)``.
    equal_spend:
        If True, control and treatment have equal spend within each campaign
        (treatment spend is set equal to control spend).
    random_seed:
        Optional seed for reproducibility. If provided, used to initialize the RNG
        on first run.
    conversions_distribution:
        How conversion counts are sampled: "poisson" uses Poisson(lam),
        "negative_binomial" uses Negative Binomial with
        mean lam = spend * CVR-UC.
    conversions_nb_dispersion:
        Target dispersion ``r > 0`` for ``conversions_distribution="negative_binomial"``.
        Rounded to the integer ``n`` passed to ``negative_binomial``. Must be ``None``
        when using Poisson.
    """

    num_advertisers: int
    campaigns_mean: float
    spend_mean: float
    spend_sd: float
    cpa_mean: float
    cpa_sd: float
    cpa_within_sd: float
    effect_mean: float
    effect_sd: float
    equal_spend: bool
    cpa_distribution: Literal["gamma", "pareto", "log_t", "quantile"] = "gamma"
    cpa_quantiles: tuple[tuple[float, float], ...] | None = None
    cpa_median: float | None = None
    cpa_iqr: float | None = None
    cpa_within_iqr: float | None = None
    cpa_tail_df: float | None = None
    random_seed: int | None = None
    conversions_distribution: Literal["poisson", "negative_binomial"] = "poisson"
    conversions_nb_dispersion: float | None = None

    # Lazy-initialized RNG; do not include in constructor signature
    _rng: np.random.Generator | None = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_seed)
        return self._rng

    def run(self) -> SimulationData:
        rng = self._get_rng()

        if self.num_advertisers < 1:
            raise ValueError("num_advertisers must be >= 1")
        if self.campaigns_mean <= 0:
            raise ValueError("campaigns_mean must be > 0")
        if self.spend_mean <= 0:
            raise ValueError("spend_mean must be > 0")
        if self.spend_sd <= 0:
            raise ValueError("spend_sd must be > 0")
        if self.cpa_distribution in ("gamma", "pareto"):
            if self.cpa_mean <= 0 or self.cpa_sd <= 0:
                raise ValueError("cpa_mean and cpa_sd must be > 0")
            if self.cpa_within_sd <= 0:
                raise ValueError("cpa_within_sd must be > 0")
        elif self.cpa_distribution == "log_t":
            if self.cpa_median is None or self.cpa_median <= 0:
                raise ValueError("cpa_median must be > 0 for log_t")
            if self.cpa_iqr is None or self.cpa_iqr <= 0:
                raise ValueError("cpa_iqr must be > 0 for log_t")
            if self.cpa_within_iqr is None or self.cpa_within_iqr <= 0:
                raise ValueError("cpa_within_iqr must be > 0 for log_t")
            if self.cpa_tail_df is None or self.cpa_tail_df <= 0:
                raise ValueError("cpa_tail_df must be > 0 for log_t")
        elif self.cpa_distribution == "quantile":
            if self.cpa_quantiles is None or len(self.cpa_quantiles) < 2:
                raise ValueError("cpa_quantiles must have at least 2 entries for quantile")
            probs = [p for p, _ in self.cpa_quantiles]
            vals = [v for _, v in self.cpa_quantiles]
            if probs != sorted(probs):
                raise ValueError("cpa_quantiles must be sorted by probability")
            if any(p <= 0.0 or p >= 1.0 for p in probs):
                raise ValueError("cpa_quantiles probabilities must be in (0, 1)")
            if any(v <= 0.0 for v in vals):
                raise ValueError("cpa_quantiles values must be > 0")
            if self.cpa_within_iqr is None or self.cpa_within_iqr <= 0:
                raise ValueError("cpa_within_iqr must be > 0 for quantile")
        if self.effect_sd < 0:
            raise ValueError("effect_sd must be >= 0")
        if self.conversions_distribution == "negative_binomial":
            r = self.conversions_nb_dispersion
            if r is None or r <= 0:
                raise ValueError(
                    "conversions_nb_dispersion must be > 0 when " "conversions_distribution is 'negative_binomial'"
                )
        elif self.conversions_distribution == "poisson":
            if self.conversions_nb_dispersion is not None:
                raise ValueError("conversions_nb_dispersion must be None when " "conversions_distribution is 'poisson'")
        else:
            raise ValueError(f"Invalid conversions_distribution: {self.conversions_distribution}")

        rows: list[dict[str, int | float]] = []

        # Lognormal re-parameterized via real-space mean (m) and sd (s):
        #   sigma^2 = ln(1 + (s / m)^2),  mu = ln(m) - sigma^2 / 2
        spend_sigma2 = np.log1p((self.spend_sd / self.spend_mean) ** 2)
        spend_mu = np.log(self.spend_mean) - spend_sigma2 / 2
        spend_sigma = np.sqrt(spend_sigma2)

        cpa_dist = self.cpa_distribution

        if cpa_dist == "pareto":
            # Pareto re-parameterized via mean (m) and sd (s):
            #   cv = s/m;  a = 1 + sqrt(1 + 1/cv^2);  x_m = m*(a-1)/a
            adv_cpa_cv = self.cpa_sd / self.cpa_mean
            adv_cpa_alpha = 1.0 + np.sqrt(1.0 + 1.0 / adv_cpa_cv**2)
            adv_cpa_xm = self.cpa_mean * (adv_cpa_alpha - 1.0) / adv_cpa_alpha
        elif cpa_dist == "log_t":
            # Log-t re-parameterized via median (M) and IQR (I):
            #   mu = ln(M);  sigma = arcsinh(I / (2M)) / t_{0.75, v}
            q75 = float(t_dist.ppf(0.75, df=self.cpa_tail_df))
            adv_logt_mu = np.log(self.cpa_median)
            adv_logt_sigma = float(np.arcsinh(self.cpa_iqr / (2.0 * self.cpa_median)) / q75)
        elif cpa_dist == "quantile":
            # Spline inverse-CDF from quantile-value pairs
            probs = np.array([p for p, _ in self.cpa_quantiles])
            vals = np.array([v for _, v in self.cpa_quantiles])
            inv_cdf = PchipInterpolator(probs, vals)
            # Campaign-level multiplicative lognormal: median=1, IQR=cpa_within_iqr
            #   sigma = arcsinh(IQR / 2) / z_{0.75}  (z_{0.75} ~ 0.6745)
            z75 = 0.6744897501960817
            cpg_qn_sigma = float(np.arcsinh(self.cpa_within_iqr / 2.0) / z75)
        else:
            # Gamma re-parameterized via mean (m) and sd (s):
            #   shape k = (m / s)^2,  scale theta = s^2 / m
            adv_cpa_shape = (self.cpa_mean / self.cpa_sd) ** 2
            adv_cpa_scale = (self.cpa_sd**2) / self.cpa_mean

        next_cpg_id = 0
        for adv_id in range(self.num_advertisers):
            # Sample number of campaigns from Exponential(mean), then convert to int with min 1
            n_cpgs = 1 + int(rng.exponential(self.campaigns_mean))

            # --- Hierarchical CPA: advertiser-level ---
            if cpa_dist == "pareto":
                adv_cpa = float((rng.pareto(adv_cpa_alpha) + 1.0) * adv_cpa_xm)
            elif cpa_dist == "log_t":
                adv_cpa = float(np.exp(adv_logt_mu + adv_logt_sigma * rng.standard_t(self.cpa_tail_df)))
            elif cpa_dist == "quantile":
                adv_cpa = float(inv_cdf(rng.uniform()))
            else:
                adv_cpa = float(rng.gamma(adv_cpa_shape, adv_cpa_scale))
            adv_cpa = max(adv_cpa, 1e-12)

            for _ in range(n_cpgs):
                # --- Hierarchical CPA: campaign-level ---
                if cpa_dist == "pareto":
                    cpg_cpa_cv = self.cpa_within_sd / adv_cpa
                    cpg_cpa_alpha = 1.0 + np.sqrt(1.0 + 1.0 / cpg_cpa_cv**2)
                    cpg_cpa_xm = adv_cpa * (cpg_cpa_alpha - 1.0) / cpg_cpa_alpha
                    base_cpa = float((rng.pareto(cpg_cpa_alpha) + 1.0) * cpg_cpa_xm)
                elif cpa_dist == "log_t":
                    cpg_mu = np.log(adv_cpa)
                    cpg_sigma = float(np.arcsinh(self.cpa_within_iqr / (2.0 * adv_cpa)) / q75)
                    base_cpa = float(np.exp(cpg_mu + cpg_sigma * rng.standard_t(self.cpa_tail_df)))
                elif cpa_dist == "quantile":
                    base_cpa = float(adv_cpa * np.exp(cpg_qn_sigma * rng.standard_normal()))
                else:
                    cpg_cpa_shape = (adv_cpa / self.cpa_within_sd) ** 2
                    cpg_cpa_scale = (self.cpa_within_sd**2) / adv_cpa
                    base_cpa = float(rng.gamma(cpg_cpa_shape, cpg_cpa_scale))
                base_cvruc = 1 / max(base_cpa, 1e-12)

                # Historical (pre-period) control
                ctrl_spend_h = float(rng.lognormal(spend_mu, spend_sigma))
                ctrl_convs_h = self._sample_conversions(rng, ctrl_spend_h * base_cvruc)

                # Experiment period - control arm
                ctrl_spend = float(rng.lognormal(spend_mu, spend_sigma))
                ctrl_convs = self._sample_conversions(rng, ctrl_spend * base_cvruc)

                # Campaign-level treatment effect (multiplicative on CVR-UC via LogNormal)
                effect_factor = float(rng.lognormal(self.effect_mean, self.effect_sd))
                treat_cvruc = max(1e-12, base_cvruc * effect_factor)

                # Experiment period - treatment arm
                if self.equal_spend:
                    treat_spend = ctrl_spend
                else:
                    treat_spend = float(rng.lognormal(spend_mu, spend_sigma))
                treat_convs = self._sample_conversions(rng, treat_spend * treat_cvruc)

                rows.append(
                    {
                        "adv_id": adv_id,
                        "cpg_id": next_cpg_id,
                        "ctrl_spend_h": ctrl_spend_h,
                        "ctrl_convs_h": ctrl_convs_h,
                        "ctrl_spend": ctrl_spend,
                        "ctrl_convs": ctrl_convs,
                        "treat_spend": treat_spend,
                        "treat_convs": treat_convs,
                    }
                )

                next_cpg_id += 1

        df = pd.DataFrame(rows, columns=SimulationData.REQUIRED_COLUMNS)
        return SimulationData(df)

    def _sample_conversions(self, rng: np.random.Generator, lam: float) -> int:
        """Sample a non-negative conversion count with mean ``max(lam, 0)``."""
        rate = max(0.0, float(lam))
        if self.conversions_distribution == "poisson":
            return int(rng.poisson(rate))
        r = float(self.conversions_nb_dispersion)  # validated in run()
        if rate == 0.0:
            return 0

        n_succ = max(1, round(r))
        p = n_succ / (n_succ + rate)
        return int(rng.negative_binomial(n_succ, p))

    @property
    def true_effect(self) -> float:
        # Ground-truth effect is the population mean of the campaign effects
        return float(self.effect_mean)

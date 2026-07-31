from typing import Self

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.meta_analysis import combine_effects

from ab_tea_lab.model.base import BaseModel
from ab_tea_lab.types import ModelResult, SimulationData


class InverseVariance(BaseModel):
    """Fixed- or random-effects inverse-variance meta-analysis via ``statsmodels.combine_effects``.

    Parameters
    ----------
    method_re : {"iterated", "chi2"} or None, optional
        Method for random effects (passed to combine_effects), or None to use fixed effects (default None).
    alpha : float, optional
        Significance level for confidence intervals and p-values (default 0.05).
    use_t : bool, optional
        Whether to use the t-distribution for inference (default False).
    two_sided : bool, optional
        Whether to compute two-sided p-values (default True).
    use_hksj : bool, optional
        Whether to use the HKSJ method for false positive rate correction (default False).
    zero_correction : float, optional
        Value added to zero counts before computing log rate-ratios and variances (default 0.5).
        Applied only to studies where at least one arm has zero conversions.
    """

    def __init__(
        self,
        method_re: str | None = None,
        alpha: float = 0.05,
        use_t: bool = False,
        two_sided: bool = True,
        use_hksj: bool = False,
        zero_correction: float = 0.5,
    ) -> None:
        self.method_re = method_re
        self.alpha = alpha
        self.use_t = use_t
        self.two_sided = two_sided
        self.use_hksj = use_hksj
        self.zero_correction = zero_correction

    def fit(self, data: SimulationData) -> Self:
        df = data.data

        # apply continuity correction to campaigns with zero counts,
        # or drop them entirely when zero_correction is 0
        has_zero = (df["ctrl_convs"] == 0) | (df["treat_convs"] == 0)
        if self.zero_correction == 0:
            df = df[~has_zero]
            ctrl_convs = df["ctrl_convs"]
            treat_convs = df["treat_convs"]
        else:
            cc = self.zero_correction * has_zero
            ctrl_convs = df["ctrl_convs"] + cc
            treat_convs = df["treat_convs"] + cc

        # log rate-ratio effect sizes and Poisson-based variances
        effects = np.log(treat_convs / df["treat_spend"]) - np.log(ctrl_convs / df["ctrl_spend"])
        variances = 1 / ctrl_convs + 1 / treat_convs

        res = combine_effects(
            effects,
            variances,
            method_re=self.method_re or "iterated",
            row_names=None,
            use_t=self.use_t,
            alpha=self.alpha,
        )

        is_fe = self.method_re is None
        eff = res.mean_effect_fe if is_fe else res.mean_effect_re
        tau2 = 0.0 if is_fe else res.tau2

        if self.use_hksj:
            se = res.sd_eff_w_fe_hksj if is_fe else res.sd_eff_w_re_hksj
        else:
            se = res.sd_eff_w_fe if is_fe else res.sd_eff_w_re

        # conf_int() indices: 0=FE, 1=RE, 2=FE-HKSJ, 3=RE-HKSJ
        ci_idx = (0 if is_fe else 1) + (2 if self.use_hksj else 0)
        ci = res.conf_int(alpha=self.alpha, use_t=self.use_t)[ci_idx]

        dist = sp_stats.t(df=res.df_resid) if self.use_t else sp_stats.norm
        z = eff / se
        p_value = dist.sf(z) if not self.two_sided else 2 * dist.sf(abs(z))

        self.result = ModelResult(
            effect=eff,
            std_error=se,
            interval_lower=ci[0],
            interval_upper=ci[1],
            p_value=p_value,
            reject_h0=bool(p_value < self.alpha),
            tau2=tau2,
            metadata=dict(method_re=self.method_re, alpha=self.alpha, use_t=self.use_t),
        )
        return self

from __future__ import annotations

import math

from ab_tea_lab.evaluator.base import BaseEvaluator
from ab_tea_lab.types import EvaluationResult, ModelResult


class Evaluator(BaseEvaluator):
    """Evaluator for computing performance metrics over simulation runs."""

    def evaluate(self, results: list[ModelResult], true_effect: float) -> EvaluationResult:
        n = len(results)
        if n == 0:
            raise ValueError("No results to evaluate.")

        estimates = [r.effect for r in results]

        mean_est = sum(estimates) / n
        bias = mean_est - true_effect
        mse = sum((e - true_effect) ** 2 for e in estimates) / n
        rmse = math.sqrt(mse)
        std_est = math.sqrt(sum((e - mean_est) ** 2 for e in estimates) / n)

        covers = sum(r.interval_lower <= true_effect <= r.interval_upper for r in results)
        coverage = covers / n

        ci_widths = [r.interval_upper - r.interval_lower for r in results]
        mean_ci_width = sum(ci_widths) / n

        rejections = [r.reject_h0 for r in results if r.reject_h0 is not None]
        rejection_rate = sum(1 for r in rejections if r) / len(rejections) if rejections else None
        is_null = true_effect == 0.0

        p_values = [r.p_value for r in results if r.p_value is not None]
        mean_p_value = sum(p_values) / len(p_values) if p_values else None

        posterior_probs = [r.posterior_prob for r in results if r.posterior_prob is not None]
        mean_posterior_prob = sum(posterior_probs) / len(posterior_probs) if posterior_probs else None

        tau2s = [r.tau2 for r in results if r.tau2 is not None]
        mean_tau2 = sum(tau2s) / len(tau2s) if tau2s else None
        std_tau2 = math.sqrt(sum((t - mean_tau2) ** 2 for t in tau2s) / len(tau2s)) if tau2s else None

        return EvaluationResult(
            coverage=coverage,
            bias=bias,
            rmse=rmse,
            mean_estimate=mean_est,
            std_estimate=std_est,
            mean_interval_width=mean_ci_width,
            power=0.0 if is_null else rejection_rate,
            type_i_error=rejection_rate if is_null else 0.0,
            mean_p_value=mean_p_value,
            mean_posterior_prob=mean_posterior_prob,
            mean_tau2=mean_tau2,
            std_tau2=std_tau2,
        )

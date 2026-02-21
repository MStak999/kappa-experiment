"""
Weibull MLE fitter with bootstrap confidence intervals.

Given step-level success/failure data from agent runs, fits the Weibull
survival model:

    P(surviving to step t) = exp(-(t / λ)^κ)

and extracts:
    κ  — shape parameter (the key safety metric)
    λ  — scale parameter
    T₅₀ — median survival time (derived from κ and λ)

Validation: recovers known parameters from synthetic data.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import weibull_min
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class WeibullFitResult:
    """Complete fit result with confidence intervals."""
    kappa: float
    kappa_ci: Tuple[float, float]
    lambda_scale: float
    lambda_ci: Tuple[float, float]
    t50: float
    t50_ci: Tuple[float, float]
    n_observations: int
    n_censored: int          # Runs that hit time limit without failing
    log_likelihood: float
    condition_label: str


def weibull_nll(params: np.ndarray, failure_times: np.ndarray,
                censored_times: np.ndarray) -> float:
    """
    Negative log-likelihood for Weibull with right-censoring.

    For observed failures at time t_i:
        log f(t_i) = log(κ/λ) + (κ-1)·log(t_i/λ) - (t_i/λ)^κ

    For right-censored observations (task completed or time limit hit):
        log S(t_i) = -(t_i/λ)^κ
    """
    log_kappa, log_lambda = params
    kappa = np.exp(log_kappa)
    lam = np.exp(log_lambda)

    nll = 0.0

    # Uncensored (observed failures)
    if len(failure_times) > 0:
        z = failure_times / lam
        nll -= np.sum(
            np.log(kappa) - np.log(lam) + (kappa - 1) * np.log(z) - z**kappa
        )

    # Right-censored (survived to this time)
    if len(censored_times) > 0:
        z = censored_times / lam
        nll -= np.sum(-z**kappa)

    return nll


def fit_weibull(failure_times: np.ndarray,
                censored_times: np.ndarray = None,
                n_bootstrap: int = 1000,
                ci_level: float = 0.95,
                condition_label: str = "") -> WeibullFitResult:
    """
    Fit Weibull distribution via MLE with bootstrap CIs.

    Args:
        failure_times: Array of times at which agents failed (in hours or steps).
        censored_times: Array of times for runs that completed without failing
                        (right-censored). None if no censoring.
        n_bootstrap: Number of bootstrap resamples for CIs.
        ci_level: Confidence level (default 95%).
        condition_label: Label for this condition (e.g. "vanilla_tier3").

    Returns:
        WeibullFitResult with point estimates and CIs.
    """
    if censored_times is None:
        censored_times = np.array([])

    failure_times = np.asarray(failure_times, dtype=float)
    censored_times = np.asarray(censored_times, dtype=float)

    # Remove zeros (can't take log)
    failure_times = failure_times[failure_times > 0]
    censored_times = censored_times[censored_times > 0]

    n_failures = len(failure_times)
    n_censored = len(censored_times)

    if n_failures < 2:
        raise ValueError(f"Need at least 2 failure observations, got {n_failures}")

    # ── Point estimate ──
    # Initial guess from method of moments on uncensored data
    init_lambda = np.median(failure_times)
    init_kappa = 1.0
    x0 = np.array([np.log(init_kappa), np.log(init_lambda)])

    result = minimize(weibull_nll, x0, args=(failure_times, censored_times),
                      method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

    kappa_hat = np.exp(result.x[0])
    lambda_hat = np.exp(result.x[1])
    t50_hat = lambda_hat * np.log(2) ** (1 / kappa_hat)
    ll = -result.fun

    # ── Bootstrap CIs ──
    all_times = np.concatenate([failure_times, censored_times])
    is_failure = np.concatenate([np.ones(n_failures), np.zeros(n_censored)])

    boot_kappas = []
    boot_lambdas = []
    boot_t50s = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(len(all_times), size=len(all_times), replace=True)
        boot_times = all_times[idx]
        boot_failures = is_failure[idx]

        bt_fail = boot_times[boot_failures == 1]
        bt_cens = boot_times[boot_failures == 0]

        if len(bt_fail) < 2:
            continue

        try:
            res = minimize(weibull_nll, result.x, args=(bt_fail, bt_cens),
                          method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            bk = np.exp(res.x[0])
            bl = np.exp(res.x[1])
            # Sanity bounds
            if 0.05 < bk < 5.0 and bl > 0:
                boot_kappas.append(bk)
                boot_lambdas.append(bl)
                boot_t50s.append(bl * np.log(2) ** (1 / bk))
        except Exception:
            continue

    alpha = (1 - ci_level) / 2
    if len(boot_kappas) > 10:
        kappa_ci = (np.percentile(boot_kappas, 100 * alpha),
                    np.percentile(boot_kappas, 100 * (1 - alpha)))
        lambda_ci = (np.percentile(boot_lambdas, 100 * alpha),
                     np.percentile(boot_lambdas, 100 * (1 - alpha)))
        t50_ci = (np.percentile(boot_t50s, 100 * alpha),
                  np.percentile(boot_t50s, 100 * (1 - alpha)))
    else:
        # Not enough bootstrap samples; report NaN
        kappa_ci = (float('nan'), float('nan'))
        lambda_ci = (float('nan'), float('nan'))
        t50_ci = (float('nan'), float('nan'))

    return WeibullFitResult(
        kappa=kappa_hat,
        kappa_ci=kappa_ci,
        lambda_scale=lambda_hat,
        lambda_ci=lambda_ci,
        t50=t50_hat,
        t50_ci=t50_ci,
        n_observations=n_failures + n_censored,
        n_censored=n_censored,
        log_likelihood=ll,
        condition_label=condition_label,
    )


def generate_synthetic_data(kappa: float, t50: float, n: int,
                            censoring_time: float = None,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Weibull failure data for validation.

    Args:
        kappa: True shape parameter.
        t50: True median survival time.
        n: Number of observations.
        censoring_time: If set, observations beyond this are right-censored.
        seed: Random seed.

    Returns:
        (failure_times, censored_times) tuple.
    """
    rng = np.random.default_rng(seed)
    lam = t50 / np.log(2) ** (1 / kappa)

    # Draw from Weibull
    raw_times = lam * (-np.log(rng.uniform(size=n))) ** (1 / kappa)

    if censoring_time is None:
        return raw_times, np.array([])

    failures = raw_times[raw_times <= censoring_time]
    censored = np.full(np.sum(raw_times > censoring_time), censoring_time)
    return failures, censored


# ── Validation ──

def validate_fitter():
    """
    Unit test: recover known κ from synthetic data.

    This is the test described in Q7:
      "The Weibull fitter recovers known parameters from synthetic data
       (ground truth κ = 0.5, 0.7, 1.0)."
    """
    print("Weibull fitter validation")
    print("=" * 55)

    test_cases = [
        {"kappa": 0.50, "t50": 10.0, "n": 500, "label": "κ=0.50 (sub-human)"},
        {"kappa": 0.70, "t50": 8.0,  "n": 500, "label": "κ=0.70 (SOTA)"},
        {"kappa": 1.00, "t50": 5.0,  "n": 500, "label": "κ=1.00 (exponential)"},
        {"kappa": 0.37, "t50": 8.0,  "n": 500, "label": "κ=0.37 (human)"},
        # With censoring (more realistic: some runs hit time limit)
        {"kappa": 0.70, "t50": 8.0, "n": 500, "label": "κ=0.70 censored",
         "censoring_time": 20.0},
    ]

    all_pass = True
    for tc in test_cases:
        censor = tc.get("censoring_time", None)
        failures, censored = generate_synthetic_data(
            tc["kappa"], tc["t50"], tc["n"], censoring_time=censor, seed=42
        )
        result = fit_weibull(failures, censored, n_bootstrap=500,
                            condition_label=tc["label"])

        # Check if true value is within CI
        kappa_in_ci = result.kappa_ci[0] <= tc["kappa"] <= result.kappa_ci[1]
        t50_in_ci = result.t50_ci[0] <= tc["t50"] <= result.t50_ci[1]

        status = "PASS" if (kappa_in_ci and t50_in_ci) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n  {tc['label']}:")
        print(f"    True κ = {tc['kappa']:.2f}, "
              f"fitted κ = {result.kappa:.3f} "
              f"({result.kappa_ci[0]:.3f}, {result.kappa_ci[1]:.3f}) "
              f"{'✓' if kappa_in_ci else '✗'}")
        print(f"    True T₅₀ = {tc['t50']:.1f}, "
              f"fitted T₅₀ = {result.t50:.2f} "
              f"({result.t50_ci[0]:.2f}, {result.t50_ci[1]:.2f}) "
              f"{'✓' if t50_in_ci else '✗'}")
        print(f"    n = {result.n_observations} "
              f"({result.n_censored} censored), "
              f"LL = {result.log_likelihood:.1f}  [{status}]")

    print(f"\n{'=' * 55}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


if __name__ == '__main__':
    validate_fitter()

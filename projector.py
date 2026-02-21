"""
Economic projector: from fitted Weibull parameters to policy-relevant thresholds.

Takes κ and T₅₀ from the fitter and computes:
  - Break-even task length (where agent cost = human cost)
  - Verification horizon (max task length where monitoring is viable)
  - Regime accessibility (P(success) for each knowledge regime)
  - Self-sufficiency threshold (when agent generates more value than it costs)

These are the outputs that matter for governance: they tell regulators
when the economic firebreak fails.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EconomicParams:
    """Cost parameters for the deployment model."""
    human_rate: float = 150.0       # $/hour for a human researcher
    cost_per_step: float = 0.15     # $/step for agent compute
    steps_per_hour: float = 80.0    # Agent steps per hour of equivalent task
    monitoring_rate: float = 75.0   # $/hour for a human monitor (cheaper than researcher)
    monitoring_ratio: float = 0.1   # Fraction of task time spent monitoring (at baseline)


@dataclass
class PolicyOutput:
    """Policy-relevant outputs from the economic projector."""
    # From fitted Weibull
    kappa: float
    t50: float
    condition: str

    # Economic thresholds
    breakeven_hours: float           # Task length where agent cost = human cost
    verification_horizon: float      # Max task length where P(undetected failure) < threshold
    
    # Regime accessibility
    p_success_regime_a: float        # P(success) for 120h recombination task
    p_success_regime_b: float        # P(success) for 800h empirical extension
    p_success_regime_c: float        # P(success) for 5000h new theory

    # Regime viability
    regime_a_viable: bool            # Agent cheaper than human for regime A?
    regime_b_viable: bool            # Agent cheaper than human for regime B?
    regime_c_viable: bool

    # Self-sufficiency
    self_sufficient: bool            # Agent generates more value than total cost?
    self_sufficiency_notes: str


def weibull_lambda(t50: float, kappa: float) -> float:
    return t50 / (np.log(2) ** (1.0 / kappa))


def p_success(task_hours: float, t50: float, kappa: float) -> float:
    lam = weibull_lambda(t50, kappa)
    return np.exp(-((task_hours / lam) ** kappa))


def expected_agent_cost(task_hours: float, t50: float, kappa: float,
                        params: EconomicParams) -> float:
    """Expected cost including retries."""
    ps = p_success(task_hours, t50, kappa)
    if ps < 1e-15:
        return float('inf')
    return params.steps_per_hour * task_hours * params.cost_per_step / ps


def human_cost(task_hours: float, params: EconomicParams) -> float:
    return params.human_rate * task_hours


def find_breakeven(t50: float, kappa: float, params: EconomicParams) -> float:
    """Binary search for task length where agent cost = human cost."""
    lo, hi = 0.1, 50000.0
    ac_lo = expected_agent_cost(lo, t50, kappa, params)
    hc_lo = human_cost(lo, params)
    if ac_lo >= hc_lo:
        return 0.0

    for _ in range(100):
        mid = (lo + hi) / 2
        ac = expected_agent_cost(mid, t50, kappa, params)
        hc = human_cost(mid, params)
        if ac < hc:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def verification_horizon(t50: float, kappa: float, params: EconomicParams,
                         safety_threshold: float = 0.05) -> float:
    """
    Maximum task length where the probability of an undetected failure
    stays below safety_threshold, given a monitoring budget.

    Simple model: monitor checks at intervals. P(undetected) depends on
    the gap between the failure mode and the check. For Weibull with κ < 1,
    failures cluster early, so monitoring is most valuable early on.

    Here we use a simpler proxy: the task length where P(success) drops
    below (1 - safety_threshold), meaning the monitoring budget would need
    to cover more than safety_threshold probability of failure.
    """
    target_p = 1 - safety_threshold
    # Binary search
    lo, hi = 0.1, 50000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if p_success(mid, t50, kappa) > target_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_policy_output(kappa: float, t50: float,
                          condition: str = "",
                          params: EconomicParams = None) -> PolicyOutput:
    """
    Compute all policy-relevant outputs from fitted Weibull parameters.

    This is what you'd hand to a regulator: given that this architecture
    has this κ and this T₅₀, here is what it means for deployment safety.
    """
    if params is None:
        params = EconomicParams()

    be = find_breakeven(t50, kappa, params)
    vh = verification_horizon(t50, kappa, params)

    # Regime accessibility
    regime_hours = {'a': 120.0, 'b': 800.0, 'c': 5000.0}
    ps = {k: p_success(h, t50, kappa) for k, h in regime_hours.items()}

    # Regime viability (agent cheaper than human?)
    viable = {}
    for k, h in regime_hours.items():
        ac = expected_agent_cost(h, t50, kappa, params)
        hc = human_cost(h, params)
        viable[k] = ac < hc

    # Self-sufficiency (rough estimate)
    # An agent is self-sufficient if it can generate enough economic value
    # from successful tasks to cover its own compute costs.
    # This is a simplified check: can it profitably do regime A tasks?
    if viable['a']:
        ss = True
        ss_notes = (f"Agent is cheaper than human for tasks up to {be:.0f}h. "
                    f"Regime A (120h) is economically viable.")
    else:
        ss = False
        ss_notes = (f"Break-even at {be:.0f}h, below Regime A threshold (120h). "
                    f"Agent cannot yet profitably do research-scale tasks.")

    return PolicyOutput(
        kappa=kappa, t50=t50, condition=condition,
        breakeven_hours=be,
        verification_horizon=vh,
        p_success_regime_a=ps['a'],
        p_success_regime_b=ps['b'],
        p_success_regime_c=ps['c'],
        regime_a_viable=viable['a'],
        regime_b_viable=viable['b'],
        regime_c_viable=viable['c'],
        self_sufficient=ss,
        self_sufficiency_notes=ss_notes,
    )


def compare_architectures(results: Dict[str, PolicyOutput]) -> str:
    """
    Generate a comparison table across architectures.
    This is the summary a policymaker would read.
    """
    lines = []
    lines.append("Architecture Comparison")
    lines.append("=" * 75)
    lines.append(f"{'Condition':<25} {'κ':>6} {'T₅₀':>6} {'B/E':>7} "
                 f"{'P(A)':>6} {'P(B)':>6} {'P(C)':>6} {'A$':>3} {'B$':>3}")
    lines.append("-" * 75)

    for label, po in sorted(results.items()):
        lines.append(
            f"{label:<25} {po.kappa:>6.3f} {po.t50:>5.1f}h "
            f"{po.breakeven_hours:>6.0f}h "
            f"{po.p_success_regime_a:>6.4f} "
            f"{po.p_success_regime_b:>6.4f} "
            f"{po.p_success_regime_c:>6.4f} "
            f"{'✓' if po.regime_a_viable else '✗':>3} "
            f"{'✓' if po.regime_b_viable else '✗':>3}"
        )

    lines.append("-" * 75)
    lines.append("B/E = break-even task length; P(X) = P(success) for regime X")
    lines.append("A$/B$ = economically viable for regime A/B")
    return "\n".join(lines)


if __name__ == '__main__':
    print("Economic Projector Demo")
    print()

    # Simulate what fitted results might look like for different architectures
    scenarios = {
        "vanilla (κ=0.75)":      (0.75, 8.0),
        "RAG (κ=0.68)":          (0.68, 9.0),
        "scratchpad (κ=0.55)":   (0.55, 8.5),
        "checkpoint (κ=0.48)":   (0.48, 10.0),
        "human baseline":        (0.37, 50.0),   # For reference
    }

    results = {}
    for label, (kappa, t50) in scenarios.items():
        po = compute_policy_output(kappa, t50, condition=label)
        results[label] = po

    print(compare_architectures(results))

    print()
    print("Policy implications:")
    print()
    for label, po in sorted(results.items()):
        if "human" in label:
            continue
        print(f"  {label}:")
        print(f"    {po.self_sufficiency_notes}")
        print(f"    Verification horizon: {po.verification_horizon:.0f}h")
        print()

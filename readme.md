# κ Experiment: Measurement Infrastructure

Working codebase for the two-arm experiment measuring Weibull κ across memory architectures and knowledge regimes.

## Status

| Module | Status | Notes |
|--------|--------|-------|
| `tasks.py` | Schema + examples | 9 example tasks across 5 tiers, both arms. Needs expansion by domain experts. |
| `fitter.py` | **Working, validated** | MLE with bootstrap CIs. Recovers known κ from synthetic data (all 5 test cases pass, including with censoring). |
| `scorer.py` | Schema + rubric | Automated novelty scoring + human evaluation rubric. Maps to q(κ, regime) modifier. |
| `projector.py` | **Working** | Takes fitted κ and T₅₀, outputs break-even, verification horizon, regime accessibility. |
| Agent wrapper | Interface only | Defined in `__init__.py`. Needs API access to implement. |
| Runner | Not yet built | Orchestration and JSONL logging. Straightforward once agent wrapper exists. |

## Quick start

```bash
# Validate the Weibull fitter (no dependencies beyond numpy/scipy)
python -m experiment.fitter

# Run the economic projector demo
python -m experiment.projector

# Check the task suite
python -m experiment.tasks

# Run the scorer demo
python -m experiment.scorer
```

## How the pieces fit together

```
tasks.py          → Generates task specifications (JSON)
                        ↓
[agent wrapper]   → Runs agents on tasks, logs step-level outcomes (JSONL)
                        ↓
fitter.py         → Fits Weibull to step-level data → κ, T₅₀ with CIs
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
projector.py                scorer.py
  ↓                           ↓
Break-even,                 q(κ, regime)
verification horizon,       novelty scores
regime accessibility
              ↓                   ↓
              └─────────┬─────────┘
                        ↓
              Policy dashboard:
              "Architecture X has κ = 0.55,
               break-even at 89h,
               q = 0.3 for novel inference.
               The firebreak holds for now."
```

## Key design decisions

**Why Weibull MLE, not just logistic regression (as METR uses)?**
METR fits a logistic curve to predict success probability as a function of human task duration. This gives T₅₀ but not κ. We need step-level survival analysis to extract the shape of reliability decay, not just its location. The Weibull model with right-censoring handles both completed and timed-out runs.

**Why matched pairs for Arm 2?**
Comparing κ across knowledge regimes is only valid if task length is controlled. A matched pair (recombination vs novel inference at the same tier) isolates the knowledge quality effect from the coherence effect.

**Why the q modifier?**
The simulation (see `../simulation.py`) shows that if success = knowledge, T₅₀ growth eventually brute-forces the frontier regardless of κ. The experiment's core contribution is measuring whether this assumption holds. q captures the gap between task completion and genuine knowledge production.

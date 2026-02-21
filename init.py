"""
kappa_experiment: Measuring the Weibull shape parameter κ across
memory architectures and knowledge regimes.

Modules:
    tasks/       — Task suite generation and schema
    fitter/      — Weibull MLE fitting with bootstrap CIs
    scorer/      — Knowledge quality scoring (automated + human rubric)
    projector/   — Economic projections from fitted parameters
    agents/      — Agent wrapper interface (scaffold-agnostic)
    runner/      — Orchestration and logging
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


# ── Shared types ──

class KnowledgeRegime(Enum):
    """The three knowledge production regimes."""
    RECOMBINATION = "recombination"       # ~120h: synthesise known results
    EMPIRICAL_EXTENSION = "empirical"     # ~800h: new experiments in known frameworks
    NEW_THEORY = "new_theory"             # ~5000h: novel conceptual frameworks


class MemoryArchitecture(Enum):
    """Agent memory scaffolds to compare."""
    VANILLA = "vanilla"                   # Context window only
    RAG = "rag"                           # Retrieval-augmented generation
    SCRATCHPAD = "scratchpad"             # Persistent scratchpad across steps
    CHECKPOINT_RESUME = "checkpoint"      # Periodic state serialisation
    FINETUNE_ON_FLY = "finetune"          # Fine-tuning during task (if feasible)


@dataclass
class TaskSpec:
    """A single task in the benchmark suite."""
    task_id: str
    regime: KnowledgeRegime
    arm: str                              # "kappa" or "knowledge_quality"
    description: str
    expected_hours: float                 # Human-equivalent duration
    length_tier: int                      # 1-5 (1h, 4h, 16h, 40h, 160h)
    ground_truth: Any                     # Verifiable answer / acceptance criterion
    paired_task_id: Optional[str] = None  # For Arm 2: matched recomb/novel pair
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepLog:
    """One step of agent execution."""
    step_number: int
    timestamp_ms: int
    action: str                           # What the agent did
    tokens_used: int
    success: bool                         # Did this step succeed?
    intermediate_output: Optional[str] = None


@dataclass
class RunResult:
    """Complete result of one agent run on one task."""
    run_id: str
    task_id: str
    model: str
    architecture: MemoryArchitecture
    seed: int
    steps: List[StepLog]
    final_success: bool
    total_tokens: int
    wall_clock_seconds: float
    final_output: Optional[str] = None


@dataclass
class WeibullFit:
    """Fitted Weibull parameters for one condition."""
    kappa: float
    kappa_ci_low: float
    kappa_ci_high: float
    lambda_scale: float
    lambda_ci_low: float
    lambda_ci_high: float
    t50: float
    t50_ci_low: float
    t50_ci_high: float
    n_runs: int
    condition: str                        # e.g. "vanilla_recombination_tier3"

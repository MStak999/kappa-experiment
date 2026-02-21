"""
Task suite generator for the κ experiment.

Two arms:
  Arm 1 (κ surveillance): Tasks at 5 length tiers with verifiable ground truth.
     Purpose: fit Weibull survival curves per architecture.
  Arm 2 (knowledge quality): Matched pairs of recombination vs novel inference
     tasks at each tier. Same length, same domain, different cognitive demands.
     Purpose: measure whether κ differs by knowledge regime.

Task design principles:
  - Every task has verifiable ground truth (automated scoring where possible)
  - Novel inference tasks use held-out data the agent cannot have seen
  - Recombination tasks require synthesis but not hypothesis formation
  - Tasks are self-contained (no external dependencies beyond provided data)
  - Intermediate checkpoints allow step-level success/failure logging
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from experiment import TaskSpec, KnowledgeRegime


# ── Length tiers ──
# These are human-equivalent hours. The agent will typically be faster,
# but the Weibull model is parameterised against human task duration
# (following METR's methodology).

LENGTH_TIERS = {
    1: 1.0,     # ~1 hour
    2: 4.0,     # ~4 hours
    3: 16.0,    # ~16 hours
    4: 40.0,    # ~40 hours (roughly 1 work week)
    5: 160.0,   # ~160 hours (roughly 1 work month)
}


# ── Example task specifications ──
# These illustrate the structure. A full suite would have 10-20 tasks per
# tier per arm, constructed by domain experts.

EXAMPLE_TASKS = [

    # ────────────────────────────────────────
    # ARM 1: κ surveillance (varying length)
    # ────────────────────────────────────────

    # Tier 1 (~1h): Short, verifiable
    TaskSpec(
        task_id="k1_t1_debug_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="kappa",
        description=(
            "Given a Python repository with 3 failing unit tests, identify and "
            "fix the bugs. All fixes involve standard library misuse or off-by-one "
            "errors. Tests must pass."
        ),
        expected_hours=1.0,
        length_tier=1,
        ground_truth={"all_tests_pass": True, "n_files_modified": "<=3"},
    ),

    # Tier 2 (~4h): Medium
    TaskSpec(
        task_id="k1_t2_analysis_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="kappa",
        description=(
            "Given a dataset of 50,000 patient records with lab values, "
            "medications, and outcomes, build a logistic regression model "
            "predicting 30-day readmission. Report AUC, calibration plot, "
            "and top 5 risk factors. Ground truth AUC must be within 0.02 "
            "of reference implementation."
        ),
        expected_hours=4.0,
        length_tier=2,
        ground_truth={"auc_tolerance": 0.02, "reference_auc": 0.73},
    ),

    # Tier 3 (~16h): Extended
    TaskSpec(
        task_id="k1_t3_pipeline_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="kappa",
        description=(
            "Build an end-to-end ML pipeline: ingest raw CSV data, clean and "
            "feature-engineer, train 3 model types (linear, tree, neural), "
            "perform hyperparameter search, evaluate on held-out test set, "
            "generate a report with plots comparing model performance. "
            "Final test set RMSE must be within 5% of reference."
        ),
        expected_hours=16.0,
        length_tier=3,
        ground_truth={"rmse_tolerance_pct": 5.0, "reference_rmse": 0.142},
    ),

    # Tier 4 (~40h): Week-scale
    TaskSpec(
        task_id="k1_t4_system_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="kappa",
        description=(
            "Design and implement a distributed task queue system with: "
            "worker pool management, retry logic with exponential backoff, "
            "dead letter queue, monitoring dashboard, and comprehensive "
            "test suite. Must pass integration tests simulating 1000 "
            "concurrent tasks with 10% random failures."
        ),
        expected_hours=40.0,
        length_tier=4,
        ground_truth={"integration_tests_pass": True, "min_test_coverage": 0.80},
    ),

    # Tier 5 (~160h): Month-scale
    TaskSpec(
        task_id="k1_t5_research_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="kappa",
        description=(
            "Replicate the core results of a specified published ML paper "
            "(provided). Implement the model architecture, training procedure, "
            "and evaluation from scratch. Reproduce Table 1 and Figure 2 "
            "within stated confidence intervals. Write a 2-page summary "
            "comparing your results to the original."
        ),
        expected_hours=160.0,
        length_tier=5,
        ground_truth={"table1_within_ci": True, "figure2_correlation": 0.95},
    ),

    # ────────────────────────────────────────
    # ARM 2: Knowledge quality (matched pairs)
    # ────────────────────────────────────────

    # Tier 3 recombination: synthesise known results
    TaskSpec(
        task_id="k2_t3_recomb_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="knowledge_quality",
        description=(
            "Given 5 published papers on transformer attention mechanisms "
            "(provided as PDFs), identify the overlooked connection between "
            "paper 3's sparse attention pattern and paper 5's routing "
            "mechanism. Write a 1-page analysis explaining how combining "
            "these approaches could improve efficiency. The connection is "
            "known in the literature but not cited by either paper."
        ),
        expected_hours=16.0,
        length_tier=3,
        ground_truth={
            "identifies_connection": True,
            "connection_type": "sparse_routing_equivalence",
            "known_in_literature": True,
        },
        paired_task_id="k2_t3_novel_01",
    ),

    # Tier 3 novel inference: form and test a hypothesis on held-out data
    TaskSpec(
        task_id="k2_t3_novel_01",
        regime=KnowledgeRegime.EMPIRICAL_EXTENSION,
        arm="knowledge_quality",
        description=(
            "Given experimental results A-D from a materials science study "
            "(provided), predict the outcome of experiment E (held out). "
            "You must: (1) state a hypothesis for why A-D show the pattern "
            "they do, (2) derive a quantitative prediction for E from that "
            "hypothesis, (3) explain what would falsify your hypothesis. "
            "Experiment E's actual result is the ground truth."
        ),
        expected_hours=16.0,
        length_tier=3,
        ground_truth={
            "prediction_within_tolerance": True,
            "tolerance_pct": 15.0,
            "experiment_e_result": 0.847,
            "requires_novel_hypothesis": True,
        },
        paired_task_id="k2_t3_recomb_01",
    ),

    # Tier 4 recombination
    TaskSpec(
        task_id="k2_t4_recomb_01",
        regime=KnowledgeRegime.RECOMBINATION,
        arm="knowledge_quality",
        description=(
            "Given a codebase implementing 4 different reinforcement learning "
            "algorithms and benchmark results on 3 environments, write a "
            "comprehensive analysis comparing their sample efficiency, "
            "stability, and compute requirements. Identify which algorithm "
            "is best suited for each environment type and explain why using "
            "known theoretical results."
        ),
        expected_hours=40.0,
        length_tier=4,
        ground_truth={
            "correct_ranking": ["PPO", "SAC", "DQN", "A2C"],
            "identifies_theoretical_basis": True,
        },
        paired_task_id="k2_t4_novel_01",
    ),

    # Tier 4 novel inference
    TaskSpec(
        task_id="k2_t4_novel_01",
        regime=KnowledgeRegime.EMPIRICAL_EXTENSION,
        arm="knowledge_quality",
        description=(
            "Given the same 4 RL algorithm implementations and benchmark "
            "results on 3 environments, design and run a new experiment "
            "testing a specific hypothesis about why Algorithm X fails on "
            "Environment Y. Implement the experiment, collect results, and "
            "report whether your hypothesis was supported. Ground truth: "
            "the actual experimental outcome on held-out environment Z."
        ),
        expected_hours=40.0,
        length_tier=4,
        ground_truth={
            "experiment_runs": True,
            "hypothesis_testable": True,
            "held_out_prediction_correct": True,
        },
        paired_task_id="k2_t4_recomb_01",
    ),
]


def validate_task_suite(tasks: List[TaskSpec]) -> List[str]:
    """Check task suite for internal consistency."""
    issues = []
    ids = set()
    for t in tasks:
        if t.task_id in ids:
            issues.append(f"Duplicate task_id: {t.task_id}")
        ids.add(t.task_id)

        if t.length_tier not in LENGTH_TIERS:
            issues.append(f"{t.task_id}: invalid length_tier {t.length_tier}")

        expected = LENGTH_TIERS.get(t.length_tier, 0)
        if abs(t.expected_hours - expected) / expected > 0.5:
            issues.append(
                f"{t.task_id}: expected_hours {t.expected_hours} doesn't match "
                f"tier {t.length_tier} ({expected}h)")

        if t.arm == "knowledge_quality" and t.paired_task_id is None:
            issues.append(f"{t.task_id}: Arm 2 task missing paired_task_id")

        if t.paired_task_id and t.paired_task_id not in ids and t.paired_task_id not in {
            t2.task_id for t2 in tasks
        }:
            issues.append(f"{t.task_id}: paired_task_id {t.paired_task_id} not found")

    # Check that paired tasks have matching tiers
    task_map = {t.task_id: t for t in tasks}
    for t in tasks:
        if t.paired_task_id and t.paired_task_id in task_map:
            pair = task_map[t.paired_task_id]
            if t.length_tier != pair.length_tier:
                issues.append(
                    f"{t.task_id} and {pair.task_id}: paired tasks at "
                    f"different tiers ({t.length_tier} vs {pair.length_tier})")

    return issues


def export_task_suite(tasks: List[TaskSpec], path: str):
    """Export task suite to JSON for reproducibility."""
    data = [asdict(t) for t in tasks]
    # Convert enums to strings
    for d in data:
        d['regime'] = d['regime'].value if hasattr(d['regime'], 'value') else d['regime']
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == '__main__':
    issues = validate_task_suite(EXAMPLE_TASKS)
    if issues:
        print("Task suite issues:")
        for i in issues:
            print(f"  - {i}")
    else:
        print(f"Task suite OK: {len(EXAMPLE_TASKS)} tasks")

    # Summary
    arm1 = [t for t in EXAMPLE_TASKS if t.arm == "kappa"]
    arm2 = [t for t in EXAMPLE_TASKS if t.arm == "knowledge_quality"]
    print(f"  Arm 1 (κ surveillance): {len(arm1)} tasks across "
          f"{len(set(t.length_tier for t in arm1))} tiers")
    print(f"  Arm 2 (knowledge quality): {len(arm2)} tasks, "
          f"{len([t for t in arm2 if t.paired_task_id])} paired")

    export_task_suite(EXAMPLE_TASKS, '/tmp/task_suite.json')
    print("  Exported to /tmp/task_suite.json")

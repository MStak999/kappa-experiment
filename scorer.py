"""
Knowledge quality scorer for Arm 2 (knowledge regime classification).

Two layers of evaluation:
  1. Automated checks — fast, scalable, imperfect
  2. Human rubric — slow, expensive, ground truth (applied to subsample)

The key question: does an agent that "succeeds" at a long task actually
produce genuine new knowledge, or just fast recombination?

This is the q(κ, regime) modifier from the extended ideas production function:
    Ȧ = α · S^λ · A^(1 - β) · q(κ, regime)

If q ≈ 1 for novel inference, agents are genuinely doing science.
If q ≈ 0, they are doing sophisticated recombination regardless of task length.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class NoveltyLevel(Enum):
    """Five-point rubric for knowledge novelty."""
    RETRIEVAL = 1         # Direct lookup / restating known facts
    SYNTHESIS = 2         # Combining known results in known ways
    CREATIVE_RECOMB = 3   # Novel combination, but all components are known
    EMPIRICAL_EXT = 4     # New experiment within known framework, genuine result
    NOVEL_INFERENCE = 5   # New hypothesis, tested, with non-obvious prediction


@dataclass
class AutomatedScore:
    """Automated quality checks on a successful completion."""

    # ── Overlap checks ──
    # How much of the output is retrievable from the training corpus?
    training_corpus_overlap: float    # 0-1, fraction of output n-grams in corpus
    citation_novelty: float           # 0-1, fraction of cited works not in prompt

    # ── Structural checks ──
    has_hypothesis: bool              # Does the output contain a testable claim?
    has_prediction: bool              # Does it derive a quantitative prediction?
    has_falsification: bool           # Does it state what would falsify the claim?
    prediction_accuracy: Optional[float]  # If ground truth exists: |pred - true| / true

    # ── Consistency checks ──
    self_consistent: bool             # No internal contradictions
    uses_provided_data: bool          # Actually references the given data
    goes_beyond_prompt: bool          # Contains substantive claims not in the prompt

    @property
    def automated_novelty_score(self) -> float:
        """
        Rough automated estimate of novelty (0-5 scale).
        This is a heuristic, not a replacement for human evaluation.
        """
        score = 1.0  # Baseline: at least retrieval

        if self.goes_beyond_prompt and self.training_corpus_overlap < 0.5:
            score += 1.0  # Some synthesis
        if self.citation_novelty > 0.3:
            score += 0.5  # Drawing on less obvious sources
        if self.has_hypothesis:
            score += 0.5
        if self.has_prediction:
            score += 0.5
        if self.has_falsification:
            score += 0.5
        if self.prediction_accuracy is not None and self.prediction_accuracy < 0.15:
            score += 1.0  # Accurate novel prediction

        return min(score, 5.0)


@dataclass
class HumanScore:
    """Human expert evaluation of knowledge quality."""
    reviewer_id: str
    novelty_level: NoveltyLevel
    confidence: float                 # 0-1, reviewer's confidence in their rating

    # Free-text justification (required for inter-rater reliability)
    justification: str

    # Specific assessments
    would_publish: bool               # Would this contribute to a real paper?
    hypothesis_quality: Optional[int] # 1-5 if hypothesis present
    methodology_quality: Optional[int] # 1-5 if experiment present
    insight_beyond_training: bool     # Does this contain insight the reviewer
                                      # believes is not in the training data?


@dataclass
class QualityAssessment:
    """Combined quality assessment for one successful run."""
    run_id: str
    task_id: str
    automated: AutomatedScore
    human_scores: List[HumanScore] = field(default_factory=list)

    @property
    def inter_rater_agreement(self) -> Optional[float]:
        """Cohen's kappa between human raters (if 2+ ratings)."""
        if len(self.human_scores) < 2:
            return None
        # Simplified: fraction of pairs that agree on novelty level
        agreements = 0
        pairs = 0
        for i in range(len(self.human_scores)):
            for j in range(i + 1, len(self.human_scores)):
                pairs += 1
                if self.human_scores[i].novelty_level == self.human_scores[j].novelty_level:
                    agreements += 1
        return agreements / pairs if pairs > 0 else None

    @property
    def mean_human_novelty(self) -> Optional[float]:
        """Mean novelty score across human raters."""
        if not self.human_scores:
            return None
        return sum(h.novelty_level.value for h in self.human_scores) / len(self.human_scores)

    @property
    def estimated_q(self) -> float:
        """
        Estimate of q(κ, regime) — the knowledge quality modifier.

        Maps the 1-5 novelty scale to [0, 1]:
          1 (retrieval)        → q = 0.0  (no new knowledge)
          2 (synthesis)        → q = 0.1  (marginal)
          3 (creative recomb)  → q = 0.3  (useful but not science)
          4 (empirical ext)    → q = 0.7  (genuine contribution)
          5 (novel inference)  → q = 1.0  (real science)

        Uses human score if available, otherwise automated estimate.
        """
        q_map = {1: 0.0, 2: 0.1, 3: 0.3, 4: 0.7, 5: 1.0}

        if self.human_scores:
            mean_level = self.mean_human_novelty
            # Interpolate in q_map
            low = int(mean_level)
            high = min(low + 1, 5)
            frac = mean_level - low
            return q_map.get(low, 0) * (1 - frac) + q_map.get(high, 1) * frac
        else:
            # Fall back to automated
            auto = self.automated.automated_novelty_score
            low = int(auto)
            high = min(low + 1, 5)
            frac = auto - low
            return q_map.get(low, 0) * (1 - frac) + q_map.get(high, 1) * frac


# ── Retrieval baseline validator ──
# Q7: "The task validator confirms novel-inference tasks are not solvable
# by pure retrieval (a retrieval-only baseline should fail)."

def validate_task_requires_inference(task_output_retrieval_only: str,
                                     ground_truth: dict) -> bool:
    """
    Check that a retrieval-only baseline fails the task.

    If a pure retrieval approach (no reasoning, just lookup) can solve the
    task, then it is not a valid novel-inference task. This is a necessary
    (not sufficient) condition for Arm 2 validity.

    In practice, this would run a retrieval-only agent (e.g. RAG with no
    reasoning chain) on each Arm 2 novel-inference task and verify it fails.
    """
    # Placeholder: in the real implementation, this would:
    # 1. Run a retrieval-only agent on the task
    # 2. Score its output against ground truth
    # 3. Return True if the retrieval agent FAILS (task requires inference)
    raise NotImplementedError(
        "Requires running a retrieval-only baseline agent. "
        "Implement when agent infrastructure is available."
    )


if __name__ == '__main__':
    # Demo: create an example assessment
    auto = AutomatedScore(
        training_corpus_overlap=0.35,
        citation_novelty=0.6,
        has_hypothesis=True,
        has_prediction=True,
        has_falsification=True,
        prediction_accuracy=0.08,
        self_consistent=True,
        uses_provided_data=True,
        goes_beyond_prompt=True,
    )

    human1 = HumanScore(
        reviewer_id="reviewer_A",
        novelty_level=NoveltyLevel.EMPIRICAL_EXT,
        confidence=0.8,
        justification="Agent formed a testable hypothesis about the observed "
                      "pattern and derived a quantitative prediction that was "
                      "close to the held-out result. The hypothesis mechanism "
                      "is plausible but not conclusively novel.",
        would_publish=True,
        hypothesis_quality=4,
        methodology_quality=3,
        insight_beyond_training=True,
    )

    human2 = HumanScore(
        reviewer_id="reviewer_B",
        novelty_level=NoveltyLevel.CREATIVE_RECOMB,
        confidence=0.6,
        justification="The prediction was accurate but the underlying reasoning "
                      "is a fairly standard application of known theory to the "
                      "specific dataset. I am not convinced this goes beyond "
                      "what a well-read graduate student would produce.",
        would_publish=False,
        hypothesis_quality=3,
        methodology_quality=3,
        insight_beyond_training=False,
    )

    assessment = QualityAssessment(
        run_id="run_001",
        task_id="k2_t3_novel_01",
        automated=auto,
        human_scores=[human1, human2],
    )

    print("Knowledge Quality Assessment Demo")
    print("=" * 45)
    print(f"  Task: {assessment.task_id}")
    print(f"  Automated novelty: {auto.automated_novelty_score:.1f} / 5")
    print(f"  Human novelty (mean): {assessment.mean_human_novelty:.1f} / 5")
    print(f"  Inter-rater agreement: {assessment.inter_rater_agreement:.2f}")
    print(f"  Estimated q: {assessment.estimated_q:.2f}")
    print()
    print("  This q value would multiply the knowledge contribution")
    print("  in the ideas production function. At q = {:.2f}, roughly".format(
        assessment.estimated_q))
    print("  {:.0f}% of the 'discovery' counts as genuine new knowledge.".format(
        assessment.estimated_q * 100))

"""Majority voting (self-consistency) evaluation."""

from typing import Any, Optional
from collections import Counter

from .base import BaseEvaluator, EvaluationResult, AggregatedMetrics
from ..datasets.base import BaseDataset, Problem


class MajorityVotingEvaluator(BaseEvaluator):
    """
    Majority voting / self-consistency evaluation.
    
    Generates multiple responses, extracts answers, and uses
    the most common answer as the prediction.
    
    From "Self-Consistency Improves Chain of Thought Reasoning"
    (Wang et al., 2022)
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        min_votes: int = 1,
        weighted: bool = False,
    ):
        """
        Initialize majority voting evaluator.
        
        Args:
            dataset: Dataset with extract_answer and check_answer methods
            min_votes: Minimum votes needed for a valid prediction
            weighted: If True, weight votes by response score (log prob)
        """
        self.dataset = dataset
        self.min_votes = min_votes
        self.weighted = weighted
    
    @property
    def name(self) -> str:
        if self.weighted:
            return "weighted_majority_voting"
        return "majority_voting"
    
    def evaluate(
        self,
        problems: list[Problem],
        responses: list[list[str]],
        scores: Optional[list[list[float]]] = None,
    ) -> tuple[list[EvaluationResult], AggregatedMetrics]:
        """Evaluate using majority voting."""
        
        results = []
        correct_count = 0
        
        # Track vote distributions
        all_vote_counts = []
        
        for i, (problem, resps) in enumerate(zip(problems, responses)):
            if not resps:
                results.append(EvaluationResult(
                    problem_id=problem.id,
                    correct=False,
                    predicted=None,
                    gold=problem.gold_answer,
                    responses=resps,
                    metadata={"vote_counts": {}}
                ))
                continue
            
            # Extract answers from all responses
            extracted_answers = []
            for resp in resps:
                answer = self.dataset.extract_answer(resp)
                extracted_answers.append(answer)
            
            # Count votes
            if self.weighted and scores and scores[i]:
                # Weighted voting
                vote_counts = self._weighted_vote_count(
                    extracted_answers, scores[i]
                )
            else:
                # Simple counting
                vote_counts = self._simple_vote_count(extracted_answers)
            
            all_vote_counts.append(vote_counts)
            
            # Find majority answer
            if vote_counts:
                majority_answer = max(vote_counts.items(), key=lambda x: x[1])
                predicted = majority_answer[0]
                vote_count = majority_answer[1]
            else:
                predicted = None
                vote_count = 0
            
            # Check if valid and correct
            if vote_count >= self.min_votes and predicted is not None:
                # Need to handle unhashable types for comparison
                correct = self._check_answer(predicted, problem.gold_answer)
            else:
                correct = False
            
            if correct:
                correct_count += 1
            
            results.append(EvaluationResult(
                problem_id=problem.id,
                correct=correct,
                predicted=predicted,
                gold=problem.gold_answer,
                responses=resps,
                scores=scores[i] if scores else None,
                metadata={
                    "vote_counts": vote_counts,
                    "majority_vote_count": vote_count,
                    "num_samples": len(resps),
                    "extracted_answers": [str(a) for a in extracted_answers],
                }
            ))
        
        # Aggregate
        total = len(problems)
        accuracy = correct_count / total if total > 0 else 0.0
        
        # Calculate additional metrics
        avg_samples = sum(len(r) for r in responses) / total if total > 0 else 0
        avg_agreement = self._calculate_agreement(all_vote_counts, responses)
        
        metrics = AggregatedMetrics(
            accuracy=accuracy,
            total=total,
            correct=correct_count,
            metrics={
                "weighted": self.weighted,
                "min_votes": self.min_votes,
                "avg_samples_per_problem": avg_samples,
                "avg_agreement_rate": avg_agreement,
            }
        )
        
        return results, metrics
    
    def _simple_vote_count(self, answers: list[Any]) -> dict:
        """Count votes for each answer."""
        # Handle unhashable types by converting to string
        str_counts = Counter()
        answer_map = {}  # str -> original answer
        
        for ans in answers:
            if ans is None:
                continue
            str_ans = self._to_hashable(ans)
            str_counts[str_ans] += 1
            answer_map[str_ans] = ans
        
        # Convert back
        return {answer_map[k]: v for k, v in str_counts.items()}
    
    def _weighted_vote_count(
        self, answers: list[Any], scores: list[float]
    ) -> dict:
        """Weight votes by score (higher score = more weight)."""
        import math
        
        str_weights = {}
        answer_map = {}
        
        # Convert scores to weights (softmax-like)
        max_score = max(scores) if scores else 0
        weights = [math.exp(s - max_score) for s in scores]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        for ans, weight in zip(answers, weights):
            if ans is None:
                continue
            str_ans = self._to_hashable(ans)
            str_weights[str_ans] = str_weights.get(str_ans, 0) + weight
            answer_map[str_ans] = ans
        
        return {answer_map[k]: v for k, v in str_weights.items()}
    
    def _to_hashable(self, answer: Any) -> str:
        """Convert answer to hashable string representation."""
        if isinstance(answer, (list, dict)):
            import json
            return json.dumps(answer, sort_keys=True)
        return str(answer)
    
    def _check_answer(self, predicted: Any, gold: Any) -> bool:
        """Check answer with proper handling."""
        try:
            return self.dataset.check_answer(predicted, gold)
        except Exception:
            return False
    
    def _calculate_agreement(
        self, vote_counts: list[dict], responses: list[list[str]]
    ) -> float:
        """Calculate average agreement rate across problems."""
        if not vote_counts:
            return 0.0
        
        agreements = []
        for vc, resps in zip(vote_counts, responses):
            if not vc or not resps:
                continue
            # Agreement = max votes / total votes
            max_votes = max(vc.values()) if vc else 0
            total = len(resps)
            if total > 0:
                agreements.append(max_votes / total)
        
        return sum(agreements) / len(agreements) if agreements else 0.0


class SelfConsistencyEvaluator(MajorityVotingEvaluator):
    """Alias for MajorityVotingEvaluator with standard settings."""
    
    def __init__(self, dataset: BaseDataset, n_samples: int = 8):
        super().__init__(dataset, min_votes=1, weighted=False)
        self.n_samples = n_samples
    
    @property
    def name(self) -> str:
        return f"self_consistency_n{self.n_samples}"


class WeightedVotingEvaluator(MajorityVotingEvaluator):
    """Majority voting weighted by log probability."""
    
    def __init__(self, dataset: BaseDataset):
        super().__init__(dataset, min_votes=1, weighted=True)
    
    @property
    def name(self) -> str:
        return "weighted_voting"

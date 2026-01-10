"""Simple accuracy evaluator."""

from typing import Any, Optional

from .base import BaseEvaluator, EvaluationResult, AggregatedMetrics
from ..datasets.base import BaseDataset, Problem


class AccuracyEvaluator(BaseEvaluator):
    """
    Simple accuracy evaluation.
    
    Uses the first response (or best by score) from each problem.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        use_best_score: bool = False,
    ):
        """
        Initialize accuracy evaluator.
        
        Args:
            dataset: Dataset with extract_answer and check_answer methods
            use_best_score: If True, use response with highest score
        """
        self.dataset = dataset
        self.use_best_score = use_best_score
    
    @property
    def name(self) -> str:
        return "accuracy"
    
    def evaluate(
        self,
        problems: list[Problem],
        responses: list[list[str]],
        scores: Optional[list[list[float]]] = None,
    ) -> tuple[list[EvaluationResult], AggregatedMetrics]:
        """Evaluate using simple accuracy."""
        
        results = []
        correct_count = 0
        
        for i, (problem, resps) in enumerate(zip(problems, responses)):
            if not resps:
                # No response
                results.append(EvaluationResult(
                    problem_id=problem.id,
                    correct=False,
                    predicted=None,
                    gold=problem.gold_answer,
                    responses=resps,
                ))
                continue
            
            # Select response
            if self.use_best_score and scores and scores[i]:
                # Use response with highest score
                best_idx = max(range(len(resps)), key=lambda j: scores[i][j])
                response = resps[best_idx]
                response_scores = scores[i]
            else:
                # Use first response
                response = resps[0]
                response_scores = scores[i] if scores else None
            
            # Extract and check answer
            predicted = self.dataset.extract_answer(response)
            correct = self.dataset.check_answer(predicted, problem.gold_answer)
            
            if correct:
                correct_count += 1
            
            results.append(EvaluationResult(
                problem_id=problem.id,
                correct=correct,
                predicted=predicted,
                gold=problem.gold_answer,
                responses=resps,
                scores=response_scores,
            ))
        
        # Aggregate
        total = len(problems)
        accuracy = correct_count / total if total > 0 else 0.0
        
        metrics = AggregatedMetrics(
            accuracy=accuracy,
            total=total,
            correct=correct_count,
            metrics={
                "use_best_score": self.use_best_score,
            }
        )
        
        return results, metrics


class GreedyEvaluator(AccuracyEvaluator):
    """Accuracy evaluator that uses first (greedy) response."""
    
    def __init__(self, dataset: BaseDataset):
        super().__init__(dataset, use_best_score=False)
    
    @property
    def name(self) -> str:
        return "greedy_accuracy"


class BestOfNEvaluator(AccuracyEvaluator):
    """Accuracy evaluator that uses best-scoring response."""
    
    def __init__(self, dataset: BaseDataset):
        super().__init__(dataset, use_best_score=True)
    
    @property
    def name(self) -> str:
        return "best_of_n_accuracy"

"""Base evaluator interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..datasets.base import Problem


@dataclass
class EvaluationResult:
    """Result of evaluating a single problem."""
    problem_id: str
    correct: bool
    predicted: Any
    gold: Any
    responses: list[str]  # All responses (for multi-sample)
    scores: Optional[list[float]] = None  # Log probs or other scores
    metadata: Optional[dict] = None


@dataclass
class AggregatedMetrics:
    """Aggregated evaluation metrics."""
    accuracy: float
    total: int
    correct: int
    metrics: dict  # Additional metrics


class BaseEvaluator(ABC):
    """Abstract base for evaluation strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator identifier."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        problems: list[Problem],
        responses: list[list[str]],
        scores: Optional[list[list[float]]] = None,
    ) -> tuple[list[EvaluationResult], AggregatedMetrics]:
        """
        Evaluate model responses.
        
        Args:
            problems: List of problems
            responses: List of response lists (multiple responses per problem)
            scores: Optional list of score lists (log probs, etc.)
            
        Returns:
            Tuple of (individual results, aggregated metrics)
        """
        pass

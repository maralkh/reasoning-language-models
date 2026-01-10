"""Evaluation metrics and strategies."""

from typing import TYPE_CHECKING

from .base import BaseEvaluator, EvaluationResult, AggregatedMetrics
from .accuracy import AccuracyEvaluator, GreedyEvaluator, BestOfNEvaluator
from .pass_at_k import PassAtKEvaluator, StrictPassAtKEvaluator, pass_at_k
from .majority_voting import (
    MajorityVotingEvaluator,
    SelfConsistencyEvaluator,
    WeightedVotingEvaluator,
)

if TYPE_CHECKING:
    from ..datasets.base import BaseDataset


def load_evaluator(config: dict, dataset: "BaseDataset") -> BaseEvaluator:
    """
    Factory function to load evaluator from config.
    
    Args:
        config: Dict with 'type' and evaluator-specific params
        dataset: Dataset for answer checking
        
    Returns:
        Initialized evaluator
    """
    eval_type = config.get("type", "accuracy")
    
    if eval_type == "accuracy":
        return AccuracyEvaluator(
            dataset=dataset,
            use_best_score=config.get("use_best_score", False),
        )
    elif eval_type == "greedy":
        return GreedyEvaluator(dataset=dataset)
    elif eval_type == "best_of_n":
        return BestOfNEvaluator(dataset=dataset)
    elif eval_type == "pass_at_k":
        return PassAtKEvaluator(
            dataset=dataset,
            k_values=config.get("k_values", [1, 5, 10]),
        )
    elif eval_type == "strict_pass_at_k":
        return StrictPassAtKEvaluator(
            dataset=dataset,
            n_samples=config.get("n_samples", 10),
            k_values=config.get("k_values", [1, 5, 10]),
        )
    elif eval_type == "majority_voting":
        return MajorityVotingEvaluator(
            dataset=dataset,
            min_votes=config.get("min_votes", 1),
            weighted=config.get("weighted", False),
        )
    elif eval_type == "self_consistency":
        return SelfConsistencyEvaluator(
            dataset=dataset,
            n_samples=config.get("n_samples", 8),
        )
    elif eval_type == "weighted_voting":
        return WeightedVotingEvaluator(dataset=dataset)
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")


__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "AggregatedMetrics",
    "AccuracyEvaluator",
    "GreedyEvaluator",
    "BestOfNEvaluator",
    "PassAtKEvaluator",
    "StrictPassAtKEvaluator",
    "pass_at_k",
    "MajorityVotingEvaluator",
    "SelfConsistencyEvaluator",
    "WeightedVotingEvaluator",
    "load_evaluator",
]

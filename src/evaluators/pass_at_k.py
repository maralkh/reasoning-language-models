"""Pass@k evaluation for code generation."""

import math
from typing import Optional
from collections import Counter

from .base import BaseEvaluator, EvaluationResult, AggregatedMetrics
from ..datasets.base import BaseDataset, Problem


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    From the Codex paper: probability of at least one correct sample
    in k samples, given n total samples with c correct.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k
        
    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0
    
    # Use log to avoid overflow
    # 1 - (n-c choose k) / (n choose k)
    # = 1 - product((n-c-i)/(n-i) for i in range(k))
    
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    
    return 1.0 - result


class PassAtKEvaluator(BaseEvaluator):
    """
    Pass@k evaluation for code generation tasks.
    
    Generates n samples and computes pass@k for various k values.
    Standard metric for HumanEval, MBPP, etc.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        k_values: list[int] = [1, 5, 10],
    ):
        """
        Initialize pass@k evaluator.
        
        Args:
            dataset: Dataset with check_answer method for code execution
            k_values: List of k values to compute pass@k for
        """
        self.dataset = dataset
        self.k_values = k_values
    
    @property
    def name(self) -> str:
        return f"pass_at_k_{self.k_values}"
    
    def evaluate(
        self,
        problems: list[Problem],
        responses: list[list[str]],
        scores: Optional[list[list[float]]] = None,
    ) -> tuple[list[EvaluationResult], AggregatedMetrics]:
        """Evaluate using pass@k."""
        
        results = []
        pass_at_k_sums = {k: 0.0 for k in self.k_values}
        
        for problem, resps in zip(problems, responses):
            if not resps:
                results.append(EvaluationResult(
                    problem_id=problem.id,
                    correct=False,
                    predicted=None,
                    gold=problem.gold_answer,
                    responses=resps,
                    metadata={"num_correct": 0, "num_samples": 0}
                ))
                continue
            
            # Check each response
            correct_flags = []
            for resp in resps:
                extracted = self.dataset.extract_answer(resp)
                is_correct = self.dataset.check_answer(extracted, problem.gold_answer)
                correct_flags.append(is_correct)
            
            n = len(resps)
            c = sum(correct_flags)
            
            # Calculate pass@k for each k
            problem_pass_at_k = {}
            for k in self.k_values:
                if k <= n:
                    pak = pass_at_k(n, c, k)
                    problem_pass_at_k[f"pass@{k}"] = pak
                    pass_at_k_sums[k] += pak
            
            results.append(EvaluationResult(
                problem_id=problem.id,
                correct=c > 0,  # At least one correct
                predicted=resps[0] if resps else None,
                gold=problem.gold_answer,
                responses=resps,
                metadata={
                    "num_correct": c,
                    "num_samples": n,
                    "correct_flags": correct_flags,
                    **problem_pass_at_k,
                }
            ))
        
        # Aggregate
        total = len(problems)
        correct_any = sum(1 for r in results if r.correct)
        
        metrics_dict = {}
        for k in self.k_values:
            metrics_dict[f"pass@{k}"] = pass_at_k_sums[k] / total if total > 0 else 0.0
        
        metrics = AggregatedMetrics(
            accuracy=correct_any / total if total > 0 else 0.0,
            total=total,
            correct=correct_any,
            metrics=metrics_dict,
        )
        
        return results, metrics


class StrictPassAtKEvaluator(PassAtKEvaluator):
    """
    Strict pass@k that requires exactly n samples per problem.
    
    If fewer samples are available, marks as failed.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        n_samples: int,
        k_values: list[int] = [1, 5, 10],
    ):
        super().__init__(dataset, k_values)
        self.n_samples = n_samples
        
        # Validate k_values
        self.k_values = [k for k in k_values if k <= n_samples]
    
    @property
    def name(self) -> str:
        return f"strict_pass_at_k_n{self.n_samples}"

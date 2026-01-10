"""Main evaluation runner."""

from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm

from ..models.base import BaseModel
from ..datasets.base import BaseDataset, Problem
from ..samplers.base import BaseSampler, SampleResult
from ..evaluators.base import BaseEvaluator, EvaluationResult, AggregatedMetrics


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""
    batch_size: int = 8
    n_samples: int = 1
    max_problems: Optional[int] = None
    show_progress: bool = True
    verbose: bool = False


class EvaluationRunner:
    """
    Main runner for evaluation.
    
    Coordinates model, sampler, dataset, and evaluator.
    """
    
    def __init__(
        self,
        model: BaseModel,
        sampler: BaseSampler,
        dataset: BaseDataset,
        evaluator: BaseEvaluator,
        config: Optional[RunConfig] = None,
    ):
        """
        Initialize runner.
        
        Args:
            model: Model for generation
            sampler: Sampling strategy
            dataset: Evaluation dataset
            evaluator: Evaluation strategy
            config: Run configuration
        """
        self.model = model
        self.sampler = sampler
        self.dataset = dataset
        self.evaluator = evaluator
        self.config = config or RunConfig()
    
    def run(self) -> tuple[list[EvaluationResult], AggregatedMetrics]:
        """
        Run full evaluation.
        
        Returns:
            Tuple of (individual results, aggregated metrics)
        """
        # Get problems
        problems = self.dataset.get_problems(limit=self.config.max_problems)
        
        if self.config.verbose:
            print(f"Running evaluation on {len(problems)} problems")
            print(f"Model: {self.model.name}")
            print(f"Sampler: {self.sampler.name}")
            print(f"Evaluator: {self.evaluator.name}")
            print(f"Samples per problem: {self.config.n_samples}")
        
        # Generate responses in batches
        all_responses = []
        all_scores = []
        
        batches = self._make_batches(problems, self.config.batch_size)
        
        if self.config.show_progress:
            batches = tqdm(batches, desc="Generating", total=len(batches))
        
        for batch in batches:
            # Format prompts
            prompts = [self.dataset.format_prompt(p) for p in batch]
            
            # Sample responses
            batch_results = self.sampler.sample(
                self.model, prompts, n=self.config.n_samples
            )
            
            # Extract text and scores
            for sample_results in batch_results:
                responses = [sr.text for sr in sample_results]
                scores = [sr.logprob for sr in sample_results if sr.logprob is not None]
                
                all_responses.append(responses)
                all_scores.append(scores if scores else None)
        
        # Evaluate
        if self.config.verbose:
            print("Evaluating responses...")
        
        results, metrics = self.evaluator.evaluate(
            problems, all_responses, all_scores
        )
        
        return results, metrics, all_responses, all_scores
    
    def _make_batches(
        self, items: list, batch_size: int
    ) -> list[list]:
        """Split items into batches."""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches


def run_evaluation(
    model: BaseModel,
    sampler: BaseSampler,
    dataset: BaseDataset,
    evaluator: BaseEvaluator,
    batch_size: int = 8,
    n_samples: int = 1,
    max_problems: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[EvaluationResult], AggregatedMetrics, list[list[str]], list[list[float]]]:
    """
    Convenience function to run evaluation.
    
    Args:
        model: Model for generation
        sampler: Sampling strategy
        dataset: Evaluation dataset
        evaluator: Evaluation strategy
        batch_size: Batch size for generation
        n_samples: Number of samples per problem
        max_problems: Maximum problems to evaluate (None = all)
        verbose: Print progress info
        
    Returns:
        Tuple of (results, metrics, responses, scores)
    """
    config = RunConfig(
        batch_size=batch_size,
        n_samples=n_samples,
        max_problems=max_problems,
        show_progress=verbose,
        verbose=verbose,
    )
    
    runner = EvaluationRunner(model, sampler, dataset, evaluator, config)
    return runner.run()

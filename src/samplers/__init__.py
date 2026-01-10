"""Sampling strategies for generation."""

from .base import BaseSampler, SampleResult
from .standard import (
    StandardSampler,
    GreedySampler,
    NucleusSampler,
    TopKSampler,
    DiverseSampler,
)
from .beam import BeamSearchSampler, SimplifiedBeamSampler
from .tree import (
    BestFirstTreeSearch,
    MCTSTreeSearch,
    GuidedTreeSearch,
    TreeNode,
    make_answer_value_fn,
)


def load_sampler(config: dict) -> BaseSampler:
    """
    Factory function to load sampler from config.
    
    Args:
        config: Dict with 'type' and sampler-specific params
        
    Returns:
        Initialized sampler
    """
    sampler_type = config.get("type", "standard")
    
    if sampler_type == "standard":
        return StandardSampler(
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            top_k=config.get("top_k", -1),
            max_tokens=config.get("max_tokens", 2048),
            stop=config.get("stop"),
        )
    elif sampler_type == "greedy":
        return GreedySampler(
            max_tokens=config.get("max_tokens", 2048),
            stop=config.get("stop"),
        )
    elif sampler_type == "nucleus":
        return NucleusSampler(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 2048),
            stop=config.get("stop"),
        )
    elif sampler_type == "top_k":
        return TopKSampler(
            temperature=config.get("temperature", 0.7),
            top_k=config.get("top_k", 40),
            max_tokens=config.get("max_tokens", 2048),
            stop=config.get("stop"),
        )
    elif sampler_type == "diverse":
        return DiverseSampler(
            temperatures=config.get("temperatures", [0.0, 0.3, 0.5, 0.7, 1.0]),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 2048),
            stop=config.get("stop"),
        )
    elif sampler_type == "beam":
        return BeamSearchSampler(
            beam_width=config.get("beam_width", 5),
            max_tokens=config.get("max_tokens", 512),
            length_penalty=config.get("length_penalty", 1.0),
            early_stopping=config.get("early_stopping", True),
            stop=config.get("stop"),
        )
    elif sampler_type == "simple_beam":
        return SimplifiedBeamSampler(
            beam_width=config.get("beam_width", 5),
            max_tokens=config.get("max_tokens", 512),
            length_penalty=config.get("length_penalty", 1.0),
            stop=config.get("stop"),
        )
    elif sampler_type == "best_first":
        return BestFirstTreeSearch(
            max_expansions=config.get("max_expansions", 50),
            branch_factor=config.get("branch_factor", 3),
            max_tokens=config.get("max_tokens", 512),
            tokens_per_step=config.get("tokens_per_step", 32),
            temperature=config.get("temperature", 0.7),
            stop=config.get("stop"),
        )
    elif sampler_type == "mcts":
        return MCTSTreeSearch(
            max_iterations=config.get("max_iterations", 100),
            branch_factor=config.get("branch_factor", 3),
            max_tokens=config.get("max_tokens", 512),
            tokens_per_step=config.get("tokens_per_step", 32),
            rollout_tokens=config.get("rollout_tokens", 64),
            temperature=config.get("temperature", 0.7),
            exploration_constant=config.get("exploration_constant", 1.414),
            stop=config.get("stop"),
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


__all__ = [
    "BaseSampler",
    "SampleResult",
    "StandardSampler",
    "GreedySampler",
    "NucleusSampler",
    "TopKSampler",
    "DiverseSampler",
    "BeamSearchSampler",
    "SimplifiedBeamSampler",
    "BestFirstTreeSearch",
    "MCTSTreeSearch",
    "GuidedTreeSearch",
    "TreeNode",
    "make_answer_value_fn",
    "load_sampler",
]

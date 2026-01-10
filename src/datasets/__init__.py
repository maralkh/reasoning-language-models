"""Evaluation datasets."""

from .base import BaseDataset, Problem
from .gsm8k import GSM8KDataset
from .aime import AIMEDataset
from .ifeval import IFEvalDataset, IFEvalConstraint
from .code import HumanEvalDataset, MBPPDataset


def load_dataset(config: dict) -> BaseDataset:
    """
    Factory function to load dataset from config.
    
    Args:
        config: Dict with 'name' and dataset-specific params
        
    Returns:
        Initialized dataset
    """
    name = config.get("name", "").lower()
    
    if name == "gsm8k":
        return GSM8KDataset(
            split=config.get("split", "test"),
            use_cot_prompt=config.get("use_cot_prompt", True),
        )
    elif name == "aime":
        return AIMEDataset(
            source=config.get("source", "hf"),
            years=config.get("years"),
            data_path=config.get("data_path"),
        )
    elif name == "ifeval":
        return IFEvalDataset(
            split=config.get("split", "train"),
        )
    elif name == "humaneval":
        return HumanEvalDataset(
            timeout=config.get("timeout", 5.0),
            include_tests_in_prompt=config.get("include_tests_in_prompt", False),
        )
    elif name == "mbpp":
        return MBPPDataset(
            split=config.get("split", "test"),
            timeout=config.get("timeout", 5.0),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "BaseDataset",
    "Problem",
    "GSM8KDataset",
    "AIMEDataset", 
    "IFEvalDataset",
    "IFEvalConstraint",
    "HumanEvalDataset",
    "MBPPDataset",
    "load_dataset",
]

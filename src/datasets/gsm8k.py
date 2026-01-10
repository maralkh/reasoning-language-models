"""GSM8K: Grade School Math dataset."""

import re
from typing import Iterator, Optional

from .base import BaseDataset, Problem


class GSM8KDataset(BaseDataset):
    """
    GSM8K: Grade School Math 8K
    
    Simple arithmetic word problems requiring multi-step reasoning.
    Good baseline for testing reasoning capabilities.
    
    Format: Answer is a single integer.
    """
    
    def __init__(
        self,
        split: str = "test",
        use_cot_prompt: bool = True,
    ):
        """
        Initialize GSM8K dataset.
        
        Args:
            split: Dataset split ("train" or "test")
            use_cot_prompt: Whether to use chain-of-thought prompting
        """
        from datasets import load_dataset
        
        self.split = split
        self.use_cot_prompt = use_cot_prompt
        
        # Load from HuggingFace
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        self._data = list(dataset)
    
    @property
    def name(self) -> str:
        return "gsm8k"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[Problem]:
        for i, item in enumerate(self._data):
            # Extract numeric answer from solution
            # GSM8K format: "#### 42" at the end
            answer_match = re.search(r"####\s*(-?\d+)", item["answer"])
            gold_answer = int(answer_match.group(1)) if answer_match else None
            
            yield Problem(
                id=f"gsm8k_{i}",
                prompt=item["question"],
                gold_answer=gold_answer,
                metadata={
                    "full_solution": item["answer"],
                }
            )
    
    def format_prompt(self, problem: Problem) -> str:
        """Format with optional chain-of-thought."""
        
        if self.use_cot_prompt:
            return (
                f"Solve this math problem step by step. "
                f"After your reasoning, provide the final numerical answer on a new line "
                f"in the format: Answer: [number]\n\n"
                f"Problem: {problem.prompt}"
            )
        else:
            return (
                f"Solve this math problem and provide just the numerical answer.\n\n"
                f"Problem: {problem.prompt}\n"
                f"Answer:"
            )
    
    def extract_answer(self, response: str) -> Optional[int]:
        """Extract integer answer from response."""
        
        # Try common formats
        patterns = [
            r"[Aa]nswer[:\s]*\$?(-?\d+)",  # Answer: 42 or Answer: $42
            r"####\s*(-?\d+)",              # GSM8K format
            r"=\s*\$?(-?\d+)\s*$",          # = 42 at end
            r"(?:is|equals?)\s*\$?(-?\d+)", # is 42, equals 42
            r"\$?(-?\d+)\s*$",              # Just number at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # Last resort: find last number in response
        numbers = re.findall(r"(-?\d+)", response)
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def check_answer(self, predicted: Optional[int], gold: int) -> bool:
        """Check if answers match."""
        if predicted is None:
            return False
        return predicted == gold

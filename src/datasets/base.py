"""Base dataset interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class Problem:
    """A single evaluation problem."""
    id: str
    prompt: str
    gold_answer: Any
    metadata: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Problem(id={self.id!r}, prompt={self.prompt[:50]!r}...)"


class BaseDataset(ABC):
    """Abstract base for evaluation datasets."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of problems."""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[Problem]:
        """Iterate over problems."""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> Any:
        """
        Extract the answer from a model response.
        
        Args:
            response: Raw model output
            
        Returns:
            Extracted answer (type depends on dataset)
        """
        pass
    
    @abstractmethod
    def check_answer(self, predicted: Any, gold: Any) -> bool:
        """
        Check if predicted answer matches gold.
        
        Args:
            predicted: Extracted prediction
            gold: Ground truth answer
            
        Returns:
            True if correct
        """
        pass
    
    def format_prompt(self, problem: Problem) -> str:
        """
        Format a problem into the final prompt for the model.
        Override this to customize prompting strategy.
        
        Args:
            problem: The problem to format
            
        Returns:
            Formatted prompt string
        """
        return problem.prompt
    
    def get_problems(self, limit: Optional[int] = None) -> list[Problem]:
        """
        Get all problems as a list.
        
        Args:
            limit: Maximum number of problems (None = all)
            
        Returns:
            List of problems
        """
        problems = list(self)
        if limit:
            problems = problems[:limit]
        return problems

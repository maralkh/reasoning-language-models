"""Base sampler interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.base import BaseModel


@dataclass
class SampleResult:
    """Result from sampling."""
    text: str
    logprob: Optional[float] = None
    metadata: Optional[dict] = None


class BaseSampler(ABC):
    """Abstract base for all sampling strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Sampler identifier."""
        pass
    
    @abstractmethod
    def sample(
        self,
        model: "BaseModel",
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """
        Generate samples for a batch of prompts.
        
        Args:
            model: The model to sample from
            prompts: List of input prompts
            n: Number of samples per prompt
            
        Returns:
            List of lists - outer list corresponds to prompts,
            inner list contains n SampleResults per prompt
        """
        pass
    
    def sample_single(
        self,
        model: "BaseModel",
        prompt: str,
        n: int = 1,
    ) -> list[SampleResult]:
        """Convenience method for single prompt."""
        results = self.sample(model, [prompt], n=n)
        return results[0]

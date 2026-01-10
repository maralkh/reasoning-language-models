"""Base model interface for all LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationOutput:
    """Single generation output with metadata."""
    text: str
    logprob: Optional[float] = None  # Total log probability
    tokens: Optional[list[str]] = None
    token_logprobs: Optional[list[float]] = None
    finish_reason: Optional[str] = None


@dataclass 
class TokenLogits:
    """Logits for next token prediction."""
    token_ids: list[int]
    logits: list[float]  # Same length as token_ids
    

class BaseModel(ABC):
    """Abstract base for all model backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        n: int = 1,
    ) -> list[list[GenerationOutput]]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (-1 = disabled)
            stop: Stop sequences
            n: Number of completions per prompt
            
        Returns:
            List of lists - outer list corresponds to prompts,
            inner list contains n GenerationOutputs per prompt
        """
        pass
    
    @abstractmethod
    def get_next_token_logits(
        self,
        prompts: list[str],
        top_k: int = 50,
    ) -> list[TokenLogits]:
        """
        Get logits for the next token for each prompt.
        Used for custom decoding (beam search, tree search).
        
        Args:
            prompts: List of input prompts (can be partial generations)
            top_k: Return only top-k tokens per prompt
            
        Returns:
            List of TokenLogits, one per prompt
        """
        pass
    
    def generate_single(
        self,
        prompt: str,
        **kwargs
    ) -> GenerationOutput:
        """Convenience method for single prompt, single output."""
        results = self.generate([prompt], n=1, **kwargs)
        return results[0][0]

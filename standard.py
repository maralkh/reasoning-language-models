"""Standard sampling methods: temperature, top_k, top_p."""

from typing import Optional

from .base import BaseSampler, SampleResult
from ..models.base import BaseModel


class StandardSampler(BaseSampler):
    """
    Standard autoregressive sampling with temperature, top_k, top_p.
    
    This wraps the model's native generate() method.
    """
    
    def __init__(
        self,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ):
        """
        Initialize standard sampler.
        
        Args:
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (-1 = disabled)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
        """
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop = stop
    
    @property
    def name(self) -> str:
        if self.temperature == 0:
            return "greedy"
        parts = [f"temp={self.temperature}"]
        if self.top_p < 1.0:
            parts.append(f"top_p={self.top_p}")
        if self.top_k > 0:
            parts.append(f"top_k={self.top_k}")
        return "_".join(parts)
    
    def sample(
        self,
        model: BaseModel,
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """Generate samples using model's native sampling."""
        
        outputs = model.generate(
            prompts=prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self.stop,
            n=n,
        )
        
        # Convert to SampleResult format
        results = []
        for prompt_outputs in outputs:
            prompt_results = []
            for output in prompt_outputs:
                prompt_results.append(SampleResult(
                    text=output.text,
                    logprob=output.logprob,
                    metadata={
                        "finish_reason": output.finish_reason,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                    }
                ))
            results.append(prompt_results)
        
        return results


class GreedySampler(StandardSampler):
    """Greedy decoding (temperature=0)."""
    
    def __init__(
        self,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ):
        super().__init__(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            stop=stop,
        )
    
    @property
    def name(self) -> str:
        return "greedy"


class NucleusSampler(StandardSampler):
    """Nucleus (top_p) sampling."""
    
    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ):
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            max_tokens=max_tokens,
            stop=stop,
        )
    
    @property
    def name(self) -> str:
        return f"nucleus_p{self.top_p}_t{self.temperature}"


class TopKSampler(StandardSampler):
    """Top-k sampling."""
    
    def __init__(
        self,
        temperature: float = 0.7,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ):
        super().__init__(
            temperature=temperature,
            top_p=1.0,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
        )
    
    @property
    def name(self) -> str:
        return f"topk_{self.top_k}_t{self.temperature}"


class DiverseSampler(BaseSampler):
    """
    Diverse sampling: generates n samples with varied temperatures.
    
    Useful for getting diverse outputs for majority voting.
    """
    
    def __init__(
        self,
        temperatures: list[float] = [0.0, 0.3, 0.5, 0.7, 1.0],
        top_p: float = 0.95,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ):
        """
        Initialize diverse sampler.
        
        Args:
            temperatures: List of temperatures to use
            top_p: Top-p for all samples
            max_tokens: Maximum tokens
            stop: Stop sequences
        """
        self.temperatures = temperatures
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
    
    @property
    def name(self) -> str:
        return f"diverse_temps_{len(self.temperatures)}"
    
    def sample(
        self,
        model: BaseModel,
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """Generate samples with varied temperatures."""
        
        # Determine how many samples per temperature
        n_temps = len(self.temperatures)
        samples_per_temp = [n // n_temps] * n_temps
        
        # Distribute remainder
        remainder = n % n_temps
        for i in range(remainder):
            samples_per_temp[i] += 1
        
        # Generate with each temperature
        all_results = [[] for _ in prompts]
        
        for temp, n_samples in zip(self.temperatures, samples_per_temp):
            if n_samples == 0:
                continue
                
            outputs = model.generate(
                prompts=prompts,
                max_tokens=self.max_tokens,
                temperature=temp,
                top_p=self.top_p,
                stop=self.stop,
                n=n_samples,
            )
            
            for i, prompt_outputs in enumerate(outputs):
                for output in prompt_outputs:
                    all_results[i].append(SampleResult(
                        text=output.text,
                        logprob=output.logprob,
                        metadata={
                            "finish_reason": output.finish_reason,
                            "temperature": temp,
                        }
                    ))
        
        return all_results

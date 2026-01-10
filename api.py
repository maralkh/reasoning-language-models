"""API client for OpenAI-compatible endpoints."""

import os
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base import BaseModel, GenerationOutput, TokenLogits


class APIModel(BaseModel):
    """
    Client for OpenAI-compatible APIs.
    
    Works with:
    - OpenAI
    - Anthropic (via openai-compatible proxy)
    - Together AI
    - Fireworks
    - Local servers (vLLM, llama.cpp)
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        """
        Initialize API client.
        
        Args:
            model_name: Model identifier (e.g., "gpt-4o", "deepseek-chat")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: API base URL (defaults to OpenAI)
            max_concurrent: Max concurrent requests
        """
        from openai import OpenAI
        
        self._model_name = model_name
        self.max_concurrent = max_concurrent
        
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
    
    @property
    def name(self) -> str:
        return self._model_name
    
    def _single_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
        n: int,
        logprobs: bool,
    ) -> list[GenerationOutput]:
        """Generate for a single prompt."""
        
        response = self.client.completions.create(
            model=self._model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            n=n,
            logprobs=5 if logprobs else None,
        )
        
        results = []
        for choice in response.choices:
            total_logprob = None
            token_logprobs = None
            
            if choice.logprobs and choice.logprobs.token_logprobs:
                token_logprobs = choice.logprobs.token_logprobs
                total_logprob = sum(lp for lp in token_logprobs if lp is not None)
            
            results.append(GenerationOutput(
                text=choice.text,
                logprob=total_logprob,
                token_logprobs=token_logprobs,
                finish_reason=choice.finish_reason,
            ))
        
        return results
    
    def _single_chat_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
        n: int,
    ) -> list[GenerationOutput]:
        """Generate using chat completions API (for chat models)."""
        
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            n=n,
        )
        
        results = []
        for choice in response.choices:
            results.append(GenerationOutput(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason,
            ))
        
        return results
    
    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,  # Not supported by OpenAI API
        stop: Optional[list[str]] = None,
        n: int = 1,
        use_chat_api: bool = True,
    ) -> list[list[GenerationOutput]]:
        """Generate completions via API with concurrent requests."""
        
        generate_fn = self._single_chat_generate if use_chat_api else self._single_generate
        
        # Use thread pool for concurrent API calls
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [
                executor.submit(
                    generate_fn,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    n=n,
                )
                for prompt in prompts
            ]
            
            results = [future.result() for future in futures]
        
        return results
    
    def get_next_token_logits(
        self,
        prompts: list[str],
        top_k: int = 50,
    ) -> list[TokenLogits]:
        """
        Get next token logits via API.
        
        Note: Most APIs don't expose raw logits, so this is limited.
        Works best with local vLLM server or text-davinci models.
        """
        # For APIs that support it, generate 1 token with logprobs
        results = []
        
        for prompt in prompts:
            try:
                response = self.client.completions.create(
                    model=self._model_name,
                    prompt=prompt,
                    max_tokens=1,
                    temperature=1.0,
                    logprobs=top_k,
                )
                
                choice = response.choices[0]
                if choice.logprobs and choice.logprobs.top_logprobs:
                    top_logprobs = choice.logprobs.top_logprobs[0]
                    # OpenAI returns token -> logprob dict
                    # We need to convert to token_ids
                    results.append(TokenLogits(
                        token_ids=[],  # API doesn't give token IDs directly
                        logits=list(top_logprobs.values()),
                    ))
                else:
                    results.append(TokenLogits(token_ids=[], logits=[]))
            except Exception:
                # Chat models don't support this
                results.append(TokenLogits(token_ids=[], logits=[]))
        
        return results


class AnthropicModel(BaseModel):
    """Direct Anthropic API client."""
    
    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        import anthropic
        
        self._model_name = model_name
        self.max_concurrent = max_concurrent
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
    
    @property
    def name(self) -> str:
        return self._model_name
    
    def _single_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
    ) -> GenerationOutput:
        """Generate for single prompt."""
        
        response = self.client.messages.create(
            model=self._model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop or [],
            messages=[{"role": "user", "content": prompt}],
        )
        
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        
        return GenerationOutput(
            text=text,
            finish_reason=response.stop_reason,
        )
    
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
        """Generate completions."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            for prompt in prompts:
                # Anthropic doesn't support n > 1 directly
                futures = [
                    executor.submit(
                        self._single_generate,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                    )
                    for _ in range(n)
                ]
                prompt_results = [f.result() for f in futures]
                results.append(prompt_results)
        
        return results
    
    def get_next_token_logits(
        self,
        prompts: list[str],
        top_k: int = 50,
    ) -> list[TokenLogits]:
        """Not supported by Anthropic API."""
        return [TokenLogits(token_ids=[], logits=[]) for _ in prompts]

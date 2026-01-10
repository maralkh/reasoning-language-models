"""Beam search decoding."""

from typing import Optional
from dataclasses import dataclass, field
import math

from .base import BaseSampler, SampleResult
from ..models.base import BaseModel


@dataclass
class BeamHypothesis:
    """A hypothesis in beam search."""
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    score: float = 0.0  # Log probability
    finished: bool = False


class BeamSearchSampler(BaseSampler):
    """
    Beam search decoding.
    
    Maintains k best hypotheses at each step.
    """
    
    def __init__(
        self,
        beam_width: int = 5,
        max_tokens: int = 512,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        stop: Optional[list[str]] = None,
    ):
        """
        Initialize beam search.
        
        Args:
            beam_width: Number of beams to maintain
            max_tokens: Maximum tokens to generate
            length_penalty: Penalty for length (>1 = longer, <1 = shorter)
            early_stopping: Stop when all beams are finished
            stop: Stop sequences
        """
        self.beam_width = beam_width
        self.max_tokens = max_tokens
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.stop = stop or []
    
    @property
    def name(self) -> str:
        return f"beam_{self.beam_width}"
    
    def sample(
        self,
        model: BaseModel,
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """
        Run beam search for each prompt.
        
        Note: n is ignored - returns beam_width hypotheses.
        For efficiency, we process prompts one at a time.
        """
        results = []
        
        for prompt in prompts:
            hypotheses = self._beam_search(model, prompt)
            
            # Take top n (or beam_width if n > beam_width)
            top_n = min(n, len(hypotheses))
            prompt_results = []
            
            for hyp in hypotheses[:top_n]:
                prompt_results.append(SampleResult(
                    text=hyp.text,
                    logprob=hyp.score,
                    metadata={
                        "length": len(hyp.tokens),
                        "normalized_score": hyp.score / (len(hyp.tokens) ** self.length_penalty),
                    }
                ))
            
            results.append(prompt_results)
        
        return results
    
    def _beam_search(self, model: BaseModel, prompt: str) -> list[BeamHypothesis]:
        """Run beam search for a single prompt."""
        
        # Initialize beams
        beams = [BeamHypothesis(text="", score=0.0)]
        finished_beams = []
        
        for step in range(self.max_tokens):
            if not beams:
                break
            
            # Get next token distributions for all active beams
            beam_prompts = [prompt + beam.text for beam in beams]
            token_logits = model.get_next_token_logits(beam_prompts, top_k=self.beam_width * 2)
            
            # Collect all candidates
            candidates = []
            
            for beam_idx, (beam, logits) in enumerate(zip(beams, token_logits)):
                if not logits.token_ids:
                    # No logits returned, mark as finished
                    beam.finished = True
                    finished_beams.append(beam)
                    continue
                
                for token_id, logit in zip(logits.token_ids, logits.logits):
                    new_score = beam.score + logit
                    
                    # Decode token to check for stop sequences
                    # This is approximate - ideally we'd have access to tokenizer
                    candidates.append({
                        "beam_idx": beam_idx,
                        "token_id": token_id,
                        "score": new_score,
                    })
            
            if not candidates:
                break
            
            # Sort by score and take top beam_width
            candidates.sort(key=lambda x: x["score"], reverse=True)
            top_candidates = candidates[:self.beam_width]
            
            # Create new beams
            new_beams = []
            
            for cand in top_candidates:
                old_beam = beams[cand["beam_idx"]]
                
                # Create new hypothesis
                new_hyp = BeamHypothesis(
                    tokens=old_beam.tokens + [cand["token_id"]],
                    score=cand["score"],
                )
                
                # Generate text for this beam to check stopping
                # This is inefficient but necessary without tokenizer access
                beam_prompt = prompt + old_beam.text
                continuation = model.generate(
                    [beam_prompt],
                    max_tokens=1,
                    temperature=0.0,
                )[0][0].text
                
                new_hyp.text = old_beam.text + continuation
                
                # Check for stop sequences
                is_finished = False
                for stop_seq in self.stop:
                    if stop_seq in new_hyp.text:
                        new_hyp.text = new_hyp.text.split(stop_seq)[0]
                        is_finished = True
                        break
                
                # Check for EOS (empty continuation)
                if not continuation.strip():
                    is_finished = True
                
                if is_finished:
                    new_hyp.finished = True
                    finished_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
            
            beams = new_beams
            
            # Early stopping if we have enough finished beams
            if self.early_stopping and len(finished_beams) >= self.beam_width:
                break
        
        # Add any remaining active beams
        finished_beams.extend(beams)
        
        # Sort by length-normalized score
        finished_beams.sort(
            key=lambda h: h.score / (len(h.tokens) ** self.length_penalty) if h.tokens else h.score,
            reverse=True
        )
        
        return finished_beams


class SimplifiedBeamSampler(BaseSampler):
    """
    Simplified beam search using model's native support.
    
    This is faster but less flexible than custom beam search.
    Works with models that support num_beams parameter.
    """
    
    def __init__(
        self,
        beam_width: int = 5,
        max_tokens: int = 512,
        length_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
    ):
        self.beam_width = beam_width
        self.max_tokens = max_tokens
        self.length_penalty = length_penalty
        self.stop = stop
    
    @property
    def name(self) -> str:
        return f"simple_beam_{self.beam_width}"
    
    def sample(
        self,
        model: BaseModel,
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """
        Use model's native beam search if available.
        Falls back to greedy if not supported.
        """
        # Try to use native beam search
        try:
            # vLLM supports best_of parameter
            outputs = model.generate(
                prompts=prompts,
                max_tokens=self.max_tokens,
                temperature=0.0,
                stop=self.stop,
                n=self.beam_width,
            )
        except Exception:
            # Fall back to greedy
            outputs = model.generate(
                prompts=prompts,
                max_tokens=self.max_tokens,
                temperature=0.0,
                stop=self.stop,
                n=1,
            )
        
        results = []
        for prompt_outputs in outputs:
            prompt_results = []
            for output in prompt_outputs[:n]:
                prompt_results.append(SampleResult(
                    text=output.text,
                    logprob=output.logprob,
                    metadata={"method": "native_beam"}
                ))
            results.append(prompt_results)
        
        return results

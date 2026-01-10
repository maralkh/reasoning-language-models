"""Local model using vLLM for fast batched inference."""

from typing import Optional
import torch

from .base import BaseModel, GenerationOutput, TokenLogits


class VLLMModel(BaseModel):
    """
    Local model using vLLM for efficient batched inference.
    
    vLLM provides:
    - Paged attention (lower memory)
    - Continuous batching (faster throughput)
    - Optimized CUDA kernels
    """
    
    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        seed: int = 42,
    ):
        """
        Initialize vLLM model.
        
        Args:
            model_name: HuggingFace model name or local path
            dtype: Data type ("auto", "float16", "bfloat16")
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None = model default)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            seed: Random seed for reproducibility
        """
        from vllm import LLM, SamplingParams
        
        self._model_name = model_name
        self._seed = seed
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams
    
    @property
    def name(self) -> str:
        return self._model_name
    
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
        """Generate completions using vLLM."""
        
        # vLLM uses temperature=0 for greedy
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            stop=stop,
            n=n,
            logprobs=1,  # Get log probabilities
            seed=self._seed,
        )
        
        # Run batched generation
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Convert to our format
        results = []
        for request_output in outputs:
            prompt_results = []
            for completion in request_output.outputs:
                # Sum token log probs for total
                total_logprob = None
                token_logprobs = None
                if completion.logprobs:
                    token_logprobs = [
                        list(lp.values())[0].logprob 
                        for lp in completion.logprobs
                    ]
                    total_logprob = sum(token_logprobs)
                
                prompt_results.append(GenerationOutput(
                    text=completion.text,
                    logprob=total_logprob,
                    token_logprobs=token_logprobs,
                    finish_reason=completion.finish_reason,
                ))
            results.append(prompt_results)
        
        return results
    
    def get_next_token_logits(
        self,
        prompts: list[str],
        top_k: int = 50,
    ) -> list[TokenLogits]:
        """Get next token logits for custom decoding."""
        
        # Generate just 1 token with logprobs to get distribution
        sampling_params = self.SamplingParams(
            max_tokens=1,
            temperature=1.0,  # Need non-zero for logprobs
            logprobs=top_k,
            seed=self._seed,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            if completion.logprobs and len(completion.logprobs) > 0:
                logprob_dict = completion.logprobs[0]
                token_ids = []
                logits = []
                for token_id, logprob_info in logprob_dict.items():
                    token_ids.append(token_id)
                    logits.append(logprob_info.logprob)
                results.append(TokenLogits(token_ids=token_ids, logits=logits))
            else:
                # Fallback if no logprobs
                results.append(TokenLogits(token_ids=[], logits=[]))
        
        return results


class HFModel(BaseModel):
    """
    Fallback using HuggingFace transformers.
    Use when vLLM isn't available or for debugging.
    Slower than vLLM but more compatible.
    """
    
    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._model_name = model_name
        self.device = device
        
        # Determine dtype
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load model with quantization options
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @property
    def name(self) -> str:
        return self._model_name
    
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
        """Generate using HF transformers."""
        
        results = []
        
        # Process prompts (can batch, but simpler to iterate for n > 1)
        for prompt in prompts:
            prompt_results = []
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            
            # Generate n samples
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None,
                    do_sample=temperature > 0,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            # Decode outputs
            for i in range(n):
                output_ids = outputs.sequences[i][input_len:]
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # Calculate log probability from scores
                logprob = None
                if outputs.scores:
                    token_logprobs = []
                    for j, score in enumerate(outputs.scores):
                        if j < len(output_ids):
                            probs = torch.softmax(score[i if n > 1 else 0], dim=-1)
                            token_logprobs.append(
                                torch.log(probs[output_ids[j]]).item()
                            )
                    logprob = sum(token_logprobs)
                
                prompt_results.append(GenerationOutput(
                    text=text,
                    logprob=logprob,
                ))
            
            results.append(prompt_results)
        
        return results
    
    def get_next_token_logits(
        self,
        prompts: list[str],
        top_k: int = 50,
    ) -> list[TokenLogits]:
        """Get next token logits."""
        
        results = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1]  # Last token logits
                
            # Get top-k
            top_logits, top_indices = torch.topk(logits, k=top_k)
            
            results.append(TokenLogits(
                token_ids=top_indices.tolist(),
                logits=top_logits.tolist(),
            ))
        
        return results

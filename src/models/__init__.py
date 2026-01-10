"""Model backends for LLM inference."""

from .base import BaseModel, GenerationOutput, TokenLogits

# Lazy imports for optional backends
_VLLM_AVAILABLE = False
_HF_AVAILABLE = False

try:
    from .local import VLLMModel, HFModel
    _VLLM_AVAILABLE = True
    _HF_AVAILABLE = True
except ImportError:
    VLLMModel = None
    HFModel = None

try:
    from .api import APIModel, AnthropicModel
except ImportError:
    APIModel = None
    AnthropicModel = None


def load_model(config: dict) -> BaseModel:
    """
    Factory function to load model from config.
    
    Args:
        config: Dict with 'type' and model-specific params
        
    Returns:
        Initialized model
    """
    model_type = config.get("type", "vllm")
    
    if model_type == "vllm":
        if VLLMModel is None:
            raise ImportError("vLLM not available. Install with: pip install vllm torch")
        return VLLMModel(
            model_name=config["name"],
            dtype=config.get("dtype", "auto"),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            max_model_len=config.get("max_model_len"),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
        )
    elif model_type == "hf":
        if HFModel is None:
            raise ImportError("HuggingFace not available. Install with: pip install transformers torch")
        return HFModel(
            model_name=config["name"],
            dtype=config.get("dtype", "auto"),
            device=config.get("device", "cuda"),
            load_in_8bit=config.get("load_in_8bit", False),
            load_in_4bit=config.get("load_in_4bit", False),
        )
    elif model_type == "api":
        if APIModel is None:
            raise ImportError("OpenAI client not available. Install with: pip install openai")
        return APIModel(
            model_name=config["name"],
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            max_concurrent=config.get("max_concurrent", 10),
        )
    elif model_type == "anthropic":
        if AnthropicModel is None:
            raise ImportError("Anthropic client not available. Install with: pip install anthropic")
        return AnthropicModel(
            model_name=config.get("name", "claude-sonnet-4-20250514"),
            api_key=config.get("api_key"),
            max_concurrent=config.get("max_concurrent", 10),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "BaseModel",
    "GenerationOutput", 
    "TokenLogits",
    "VLLMModel",
    "HFModel",
    "APIModel",
    "AnthropicModel",
    "load_model",
]

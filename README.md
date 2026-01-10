# LLM Reasoning Evaluation Framework

A simple, flexible framework for evaluating LLMs on reasoning tasks like math and coding.

## Features

- **Models**: Local models via vLLM (fast) or HuggingFace, plus API support
- **Sampling**: Greedy, nucleus, top-k, beam search, tree search (MCTS)
- **Datasets**: GSM8K, AIME, IFEval, HumanEval, MBPP
- **Evaluation**: Accuracy, pass@k, majority voting, self-consistency

## Installation

```bash
pip install -r requirements.txt
```

For best performance with local models, ensure you have a CUDA-capable GPU.

## Quick Start

### Basic Evaluation

```bash
# Greedy evaluation on GSM8K
python scripts/run_eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dataset gsm8k

# With config file
python scripts/run_eval.py --config config/gsm8k_greedy.yaml
```

### Self-Consistency (Majority Voting)

```bash
python scripts/run_eval.py --config config/gsm8k_self_consistency.yaml
```

### Tree Search (MCTS)

```bash
python scripts/run_eval.py --config config/aime_mcts.yaml
```

### Pass@k for Code

```bash
python scripts/run_eval.py --config config/humaneval_pass_at_k.yaml
```

## Configuration

Create a YAML config file (see `config/` for examples):

```yaml
model:
  type: vllm  # vllm, hf, api, anthropic
  name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  gpu_memory_utilization: 0.9

sampler:
  type: greedy  # greedy, nucleus, mcts, etc.
  max_tokens: 2048

dataset:
  name: gsm8k

evaluator:
  type: accuracy  # accuracy, pass_at_k, majority_voting

run:
  batch_size: 8
  n_samples: 1
```

## Supported Components

### Models

| Type | Description |
|------|-------------|
| `vllm` | vLLM for fast local inference (recommended) |
| `hf` | HuggingFace transformers (fallback) |
| `api` | OpenAI-compatible API |
| `anthropic` | Anthropic API |

### Samplers

| Type | Description |
|------|-------------|
| `greedy` | Temperature=0, deterministic |
| `nucleus` | Top-p sampling |
| `top_k` | Top-k sampling |
| `diverse` | Multiple temperatures for diversity |
| `beam` | Beam search |
| `best_first` | Best-first tree search |
| `mcts` | Monte Carlo Tree Search |

### Datasets

| Name | Type | Description |
|------|------|-------------|
| `gsm8k` | Math | Grade school math |
| `aime` | Math | Competition math |
| `ifeval` | Instructions | Instruction following |
| `humaneval` | Code | Python coding |
| `mbpp` | Code | Basic Python |

### Evaluators

| Type | Description |
|------|-------------|
| `accuracy` | Simple accuracy |
| `best_of_n` | Best by log probability |
| `pass_at_k` | Code generation metric |
| `majority_voting` | Self-consistency |
| `weighted_voting` | Probability-weighted voting |

## Project Structure

```
llm-reasoning-eval/
├── config/              # Example configurations
├── src/
│   ├── models/          # Model backends
│   ├── samplers/        # Sampling strategies
│   ├── datasets/        # Evaluation datasets
│   ├── evaluators/      # Evaluation metrics
│   ├── runners/         # Orchestration
│   └── utils/           # IO utilities
├── scripts/
│   └── run_eval.py      # Main entry point
├── results/             # Saved results
└── requirements.txt
```

## Output

Results are saved to `results/<run_name>_<timestamp>/`:

- `config.json` - Run configuration
- `metrics.json` - Aggregated metrics
- `results.json` - Per-problem results
- `responses.json` - Raw model responses

## Example: Comparing Methods

```python
from src.utils import compare_runs

comparison = compare_runs([
    "results/gsm8k_greedy_20240101_120000",
    "results/gsm8k_self_consistency_20240101_130000",
    "results/gsm8k_mcts_20240101_140000",
])
print(comparison)
```

## Adding Custom Components

### Custom Dataset

```python
from src.datasets.base import BaseDataset, Problem

class MyDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "my_dataset"
    
    def __iter__(self):
        for item in self._data:
            yield Problem(
                id=item["id"],
                prompt=item["question"],
                gold_answer=item["answer"],
            )
    
    def extract_answer(self, response: str):
        # Parse model output
        ...
    
    def check_answer(self, predicted, gold) -> bool:
        return predicted == gold
```

### Custom Sampler

```python
from src.samplers.base import BaseSampler, SampleResult

class MySampler(BaseSampler):
    @property
    def name(self) -> str:
        return "my_sampler"
    
    def sample(self, model, prompts, n=1):
        # Custom sampling logic
        ...
```

## Tips

1. **Memory**: Use `gpu_memory_utilization: 0.9` for vLLM
2. **Speed**: vLLM is 5-10x faster than HuggingFace for batched inference
3. **4-bit**: Use `--load-in-4bit` with HF backend for low memory
4. **Batching**: Larger batches = faster, but more memory
5. **MCTS**: Slower but can improve accuracy on hard problems

## License

MIT

#!/usr/bin/env python3
"""
Main entry point for running evaluations.

Usage:
    python scripts/run_eval.py --config config/example.yaml
    python scripts/run_eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dataset gsm8k
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.models import load_model
from src.datasets import load_dataset
from src.samplers import load_sampler
from src.evaluators import load_evaluator
from src.runners import run_evaluation
from src.utils import save_results


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_config_from_args(args) -> dict:
    """Build configuration from command line arguments."""
    config = {
        "model": {
            "type": args.model_type,
            "name": args.model,
        },
        "sampler": {
            "type": args.sampler,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "dataset": {
            "name": args.dataset,
        },
        "evaluator": {
            "type": args.evaluator,
        },
        "run": {
            "batch_size": args.batch_size,
            "n_samples": args.n_samples,
            "max_problems": args.max_problems,
        },
        "output": {
            "dir": args.output_dir,
            "run_name": args.run_name or f"{args.dataset}_{args.model.split('/')[-1]}",
        },
    }
    
    # Add model-specific options
    if args.model_type == "vllm":
        config["model"]["gpu_memory_utilization"] = args.gpu_memory
    elif args.model_type == "hf":
        config["model"]["load_in_4bit"] = args.load_in_4bit
        config["model"]["load_in_8bit"] = args.load_in_8bit
    elif args.model_type == "api":
        config["model"]["base_url"] = args.api_base_url
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM reasoning evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file (overrides other args)
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file",
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vllm",
        choices=["vllm", "hf", "api", "anthropic"],
        help="Model backend type",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (vLLM)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (HF)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (HF)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        help="API base URL (for API models)",
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "aime", "ifeval", "humaneval", "mbpp"],
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum problems to evaluate",
    )
    
    # Sampler options
    parser.add_argument(
        "--sampler", "-s",
        type=str,
        default="greedy",
        choices=["greedy", "standard", "nucleus", "top_k", "diverse", 
                 "beam", "simple_beam", "best_first", "mcts"],
        help="Sampling strategy",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per problem",
    )
    
    # Evaluator options
    parser.add_argument(
        "--evaluator", "-e",
        type=str,
        default="accuracy",
        choices=["accuracy", "greedy", "best_of_n", "pass_at_k", 
                 "majority_voting", "self_consistency", "weighted_voting"],
        help="Evaluation strategy",
    )
    
    # Run options
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Load or build config
    if args.config:
        config = load_config(args.config)
    else:
        config = build_config_from_args(args)
    
    print("=" * 60)
    print("LLM Reasoning Evaluation")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)
    
    # Initialize components
    print("\nInitializing model...")
    model = load_model(config["model"])
    
    print("Loading dataset...")
    dataset = load_dataset(config["dataset"])
    
    print("Setting up sampler...")
    sampler = load_sampler(config["sampler"])
    
    print("Setting up evaluator...")
    evaluator = load_evaluator(config["evaluator"], dataset)
    
    # Run evaluation
    print("\nStarting evaluation...")
    run_config = config.get("run", {})
    
    results, metrics, responses, scores = run_evaluation(
        model=model,
        sampler=sampler,
        dataset=dataset,
        evaluator=evaluator,
        batch_size=run_config.get("batch_size", 8),
        n_samples=run_config.get("n_samples", 1),
        max_problems=run_config.get("max_problems"),
        verbose=args.verbose or run_config.get("verbose", True),
    )
    
    # Save results
    output_config = config.get("output", {})
    save_results(
        output_dir=output_config.get("dir", "results"),
        run_name=output_config.get("run_name", "eval"),
        results=results,
        metrics=metrics,
        config=config,
        responses=responses,
        scores=scores,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

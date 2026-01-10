"""Utilities for saving and loading results."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import asdict

from ..evaluators.base import EvaluationResult, AggregatedMetrics


def save_results(
    output_dir: str,
    run_name: str,
    results: list[EvaluationResult],
    metrics: AggregatedMetrics,
    config: dict,
    responses: Optional[list[list[str]]] = None,
    scores: Optional[list[list[float]]] = None,
) -> str:
    """
    Save evaluation results to disk.
    
    Args:
        output_dir: Output directory
        run_name: Name for this run
        results: List of evaluation results
        metrics: Aggregated metrics
        config: Run configuration
        responses: Optional raw responses
        scores: Optional response scores
        
    Returns:
        Path to the saved results directory
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    # Save metrics summary
    metrics_dict = {
        "accuracy": metrics.accuracy,
        "total": metrics.total,
        "correct": metrics.correct,
        **metrics.metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save detailed results
    results_data = []
    for r in results:
        result_dict = {
            "problem_id": r.problem_id,
            "correct": r.correct,
            "predicted": _serialize(r.predicted),
            "gold": _serialize(r.gold),
            "num_responses": len(r.responses) if r.responses else 0,
        }
        if r.metadata:
            result_dict["metadata"] = _serialize(r.metadata)
        if r.scores:
            result_dict["scores"] = r.scores
        results_data.append(result_dict)
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Save full responses if provided
    if responses:
        responses_data = []
        for i, (result, resps) in enumerate(zip(results, responses)):
            resp_entry = {
                "problem_id": result.problem_id,
                "responses": resps,
            }
            if scores and i < len(scores):
                resp_entry["scores"] = scores[i]
            responses_data.append(resp_entry)
        
        with open(run_dir / "responses.json", "w") as f:
            json.dump(responses_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*50}")
    print(f"Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
    for key, value in metrics.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"{'='*50}\n")
    
    return str(run_dir)


def load_results(run_dir: str) -> tuple[list[dict], dict, dict]:
    """
    Load results from a previous run.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tuple of (results, metrics, config)
    """
    run_path = Path(run_dir)
    
    with open(run_path / "config.json") as f:
        config = json.load(f)
    
    with open(run_path / "metrics.json") as f:
        metrics = json.load(f)
    
    with open(run_path / "results.json") as f:
        results = json.load(f)
    
    return results, metrics, config


def load_responses(run_dir: str) -> list[dict]:
    """
    Load raw responses from a previous run.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        List of response entries
    """
    run_path = Path(run_dir)
    responses_path = run_path / "responses.json"
    
    if not responses_path.exists():
        raise FileNotFoundError(f"No responses.json in {run_dir}")
    
    with open(responses_path) as f:
        return json.load(f)


def _serialize(obj: Any) -> Any:
    """Serialize object for JSON."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    # Try dataclass
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    # Fallback
    return str(obj)


def compare_runs(
    run_dirs: list[str],
    metric_keys: Optional[list[str]] = None,
) -> dict:
    """
    Compare metrics across multiple runs.
    
    Args:
        run_dirs: List of run directory paths
        metric_keys: Specific metrics to compare (None = all)
        
    Returns:
        Comparison dict with run names as keys
    """
    comparison = {}
    
    for run_dir in run_dirs:
        try:
            _, metrics, config = load_results(run_dir)
            run_name = Path(run_dir).name
            
            run_data = {
                "model": config.get("model", {}).get("name", "unknown"),
                "sampler": config.get("sampler", {}).get("type", "unknown"),
                "evaluator": config.get("evaluator", {}).get("type", "unknown"),
                "accuracy": metrics.get("accuracy", 0),
            }
            
            # Add additional metrics
            if metric_keys:
                for key in metric_keys:
                    if key in metrics:
                        run_data[key] = metrics[key]
            else:
                run_data.update(metrics)
            
            comparison[run_name] = run_data
        except Exception as e:
            print(f"Warning: Could not load {run_dir}: {e}")
    
    return comparison

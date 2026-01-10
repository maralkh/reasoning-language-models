"""HumanEval: Code generation evaluation dataset."""

import re
from typing import Any, Iterator, Optional
import tempfile
import os

from .base import BaseDataset, Problem


class HumanEvalDataset(BaseDataset):
    """
    HumanEval: Hand-Written Evaluation Set for Code Generation
    
    164 Python programming problems with test cases.
    Standard benchmark for pass@k evaluation.
    
    Source: openai/human-eval on HuggingFace
    """
    
    def __init__(
        self,
        timeout: float = 5.0,
        include_tests_in_prompt: bool = False,
    ):
        """
        Initialize HumanEval dataset.
        
        Args:
            timeout: Execution timeout in seconds
            include_tests_in_prompt: Whether to include example tests in prompt
        """
        from datasets import load_dataset
        
        self.timeout = timeout
        self.include_tests_in_prompt = include_tests_in_prompt
        
        dataset = load_dataset("openai_humaneval", split="test")
        self._data = list(dataset)
    
    @property
    def name(self) -> str:
        return "humaneval"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[Problem]:
        for item in self._data:
            yield Problem(
                id=item["task_id"],
                prompt=item["prompt"],  # Function signature + docstring
                gold_answer={
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                    "canonical_solution": item.get("canonical_solution", ""),
                },
                metadata={
                    "task_id": item["task_id"],
                    "entry_point": item["entry_point"],
                }
            )
    
    def format_prompt(self, problem: Problem) -> str:
        """Format as code completion task."""
        
        base_prompt = (
            f"Complete the following Python function. "
            f"Only output the function body, no explanations.\n\n"
            f"{problem.prompt}"
        )
        
        if self.include_tests_in_prompt:
            # Extract example from docstring if present
            docstring_match = re.search(r'"""(.+?)"""', problem.prompt, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1)
                examples = re.findall(r'>>>.+', docstring)
                if examples:
                    base_prompt += f"\n\n# Examples from docstring:\n"
                    for ex in examples[:3]:  # Limit examples
                        base_prompt += f"# {ex}\n"
        
        return base_prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract code from response."""
        
        # Try to find code block
        code_block = re.search(r'```(?:python)?\n(.+?)```', response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Try to find function definition
        func_match = re.search(r'(def .+?)(?:\n\n|\Z)', response, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()
        
        # Return as-is if no patterns match
        return response.strip()
    
    def check_answer(self, predicted: str, gold: dict) -> bool:
        """
        Check if code passes test cases.
        
        WARNING: This executes code! Only run on trusted inputs.
        """
        return self._execute_tests(predicted, gold)
    
    def _execute_tests(self, code: str, gold: dict) -> bool:
        """Execute code against test cases."""
        from multiprocess import Process, Queue
        
        def run_tests(code: str, test: str, entry_point: str, queue: Queue):
            try:
                # Create execution namespace
                namespace = {}
                
                # Execute the generated code
                exec(code, namespace)
                
                # Execute tests
                exec(test, namespace)
                
                # Run the check function
                check_fn = namespace.get("check")
                if check_fn:
                    check_fn(namespace[entry_point])
                
                queue.put(True)
            except Exception as e:
                queue.put(False)
        
        # Combine prompt (for function signature) with generated code
        # This handles cases where model only outputs function body
        full_code = self._prepare_code(code, gold)
        
        queue = Queue()
        process = Process(
            target=run_tests,
            args=(full_code, gold["test"], gold["entry_point"], queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return False
        
        try:
            return queue.get_nowait()
        except Exception:
            return False
    
    def _prepare_code(self, generated: str, gold: dict) -> str:
        """Prepare full executable code."""
        
        entry_point = gold["entry_point"]
        
        # If generated code contains the function definition, use as-is
        if f"def {entry_point}" in generated:
            return generated
        
        # Otherwise, assume it's just the function body
        # Would need the original prompt to reconstruct
        return generated


class MBPPDataset(BaseDataset):
    """
    MBPP: Mostly Basic Python Programming
    
    974 crowd-sourced Python programming problems.
    Simpler than HumanEval, good for baseline evaluation.
    
    Source: google-research/mbpp on HuggingFace
    """
    
    def __init__(
        self,
        split: str = "test",
        timeout: float = 5.0,
    ):
        """
        Initialize MBPP dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
            timeout: Execution timeout in seconds
        """
        from datasets import load_dataset
        
        self.timeout = timeout
        
        # MBPP has sanitized version which is cleaner
        try:
            dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception:
            dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
        
        self._data = list(dataset)
    
    @property
    def name(self) -> str:
        return "mbpp"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[Problem]:
        for item in self._data:
            yield Problem(
                id=f"mbpp_{item['task_id']}",
                prompt=item["text"],  # Natural language description
                gold_answer={
                    "test_list": item.get("test_list", []),
                    "code": item.get("code", ""),
                },
                metadata={
                    "task_id": item["task_id"],
                    "test_setup_code": item.get("test_setup_code", ""),
                    "challenge_test_list": item.get("challenge_test_list", []),
                }
            )
    
    def format_prompt(self, problem: Problem) -> str:
        """Format as natural language to code."""
        return (
            f"Write a Python function to solve the following problem. "
            f"Only output the function definition and body, no explanations.\n\n"
            f"Problem: {problem.prompt}\n\n"
            f"```python\n"
        )
    
    def extract_answer(self, response: str) -> str:
        """Extract code from response."""
        
        # Remove markdown code blocks if present
        code = re.sub(r'```python\n?', '', response)
        code = re.sub(r'```\n?', '', code)
        
        return code.strip()
    
    def check_answer(self, predicted: str, gold: dict) -> bool:
        """Check if code passes test cases."""
        from multiprocess import Process, Queue
        
        def run_tests(code: str, tests: list, queue: Queue):
            try:
                namespace = {}
                exec(code, namespace)
                
                for test in tests:
                    exec(test, namespace)
                
                queue.put(True)
            except Exception:
                queue.put(False)
        
        queue = Queue()
        process = Process(
            target=run_tests,
            args=(predicted, gold.get("test_list", []), queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return False
        
        try:
            return queue.get_nowait()
        except Exception:
            return False

"""AIME: American Invitational Mathematics Examination dataset."""

import re
from typing import Iterator, Optional

from .base import BaseDataset, Problem


class AIMEDataset(BaseDataset):
    """
    AIME: American Invitational Mathematics Examination
    
    Challenging competition math problems.
    Answers are always integers from 0-999.
    
    Uses multiple sources:
    1. HuggingFace datasets (various AIME collections)
    2. Can also load from local JSONL files
    """
    
    def __init__(
        self,
        source: str = "hf",
        years: Optional[list[int]] = None,
        data_path: Optional[str] = None,
    ):
        """
        Initialize AIME dataset.
        
        Args:
            source: "hf" for HuggingFace, "local" for local file
            years: Filter to specific years (None = all)
            data_path: Path to local JSONL file (if source="local")
        """
        self.source = source
        self.years = years
        self._data = []
        
        if source == "hf":
            self._load_from_hf()
        elif source == "local" and data_path:
            self._load_from_local(data_path)
        else:
            # Try multiple HF datasets
            self._load_from_hf()
    
    def _load_from_hf(self):
        """Load from HuggingFace datasets."""
        from datasets import load_dataset
        
        try:
            # Try the AI-MO/aimo-validation-aime dataset
            dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
            
            for i, item in enumerate(dataset):
                year = item.get("year", 0)
                if self.years and year not in self.years:
                    continue
                    
                self._data.append({
                    "id": f"aime_{year}_{item.get('problem_number', i)}",
                    "problem": item["problem"],
                    "answer": item.get("answer"),
                    "year": year,
                    "number": item.get("problem_number", i),
                })
        except Exception:
            # Fallback: try another source
            try:
                dataset = load_dataset("Maxwell-Jian/AIME_1983_2024", split="train")
                for i, item in enumerate(dataset):
                    year = item.get("Year", 0)
                    if self.years and year not in self.years:
                        continue
                    
                    self._data.append({
                        "id": f"aime_{year}_{i}",
                        "problem": item["Question"],
                        "answer": item.get("Answer"),
                        "year": year,
                        "number": i,
                    })
            except Exception as e:
                print(f"Warning: Could not load AIME from HuggingFace: {e}")
                print("Consider using source='local' with a data_path")
    
    def _load_from_local(self, path: str):
        """Load from local JSONL file."""
        import json
        
        with open(path) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                year = item.get("year", 0)
                if self.years and year not in self.years:
                    continue
                    
                self._data.append({
                    "id": item.get("id", f"aime_{i}"),
                    "problem": item["problem"],
                    "answer": item.get("answer"),
                    "year": year,
                    "number": item.get("number", i),
                })
    
    @property
    def name(self) -> str:
        return "aime"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[Problem]:
        for item in self._data:
            # Parse answer - should be integer 0-999
            answer = item.get("answer")
            if isinstance(answer, str):
                try:
                    answer = int(answer)
                except ValueError:
                    answer = None
            
            yield Problem(
                id=item["id"],
                prompt=item["problem"],
                gold_answer=answer,
                metadata={
                    "year": item.get("year"),
                    "number": item.get("number"),
                }
            )
    
    def format_prompt(self, problem: Problem) -> str:
        """Format AIME problem with instructions."""
        return (
            f"Solve this AIME (American Invitational Mathematics Examination) problem. "
            f"Show your work step by step. The answer is always an integer from 0 to 999.\n\n"
            f"Problem: {problem.prompt}\n\n"
            f"After your solution, provide the final answer in the format: Answer: [number]"
        )
    
    def extract_answer(self, response: str) -> Optional[int]:
        """Extract integer answer (0-999) from response."""
        
        patterns = [
            r"[Aa]nswer[:\s]*(\d+)",
            r"[Ff]inal [Aa]nswer[:\s]*(\d+)",
            r"=\s*(\d+)\s*$",
            r"\*\*(\d+)\*\*",  # Bold in markdown
            r"\\boxed\{(\d+)\}",  # LaTeX boxed
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                try:
                    val = int(match.group(1))
                    # AIME answers are 0-999
                    if 0 <= val <= 999:
                        return val
                except ValueError:
                    continue
        
        # Find last 1-3 digit number
        numbers = re.findall(r"\b(\d{1,3})\b", response)
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    def check_answer(self, predicted: Optional[int], gold: int) -> bool:
        """Check exact match."""
        if predicted is None or gold is None:
            return False
        return predicted == gold

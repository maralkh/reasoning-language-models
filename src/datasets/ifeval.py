"""IFEval: Instruction Following Evaluation dataset."""

import re
from typing import Any, Iterator, Optional
from dataclasses import dataclass

from .base import BaseDataset, Problem


@dataclass
class IFEvalConstraint:
    """A single instruction constraint to check."""
    type: str  # Constraint type (e.g., "length", "format", "content")
    params: dict  # Type-specific parameters


class IFEvalDataset(BaseDataset):
    """
    IFEval: Instruction Following Evaluation
    
    Tests whether models follow specific formatting/content instructions.
    Each problem has verifiable constraints.
    
    Source: google/IFEval on HuggingFace
    """
    
    def __init__(self, split: str = "train"):
        """
        Initialize IFEval dataset.
        
        Args:
            split: Dataset split (IFEval only has "train")
        """
        from datasets import load_dataset
        
        dataset = load_dataset("google/IFEval", split=split)
        self._data = list(dataset)
    
    @property
    def name(self) -> str:
        return "ifeval"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[Problem]:
        for i, item in enumerate(self._data):
            # Parse constraints from the instruction_id_list
            constraints = self._parse_constraints(item)
            
            yield Problem(
                id=f"ifeval_{i}",
                prompt=item["prompt"],
                gold_answer=constraints,  # Constraints are the "answer"
                metadata={
                    "instruction_id_list": item.get("instruction_id_list", []),
                    "kwargs": item.get("kwargs", []),
                }
            )
    
    def _parse_constraints(self, item: dict) -> list[IFEvalConstraint]:
        """Parse constraint specifications from item."""
        constraints = []
        
        instruction_ids = item.get("instruction_id_list", [])
        kwargs_list = item.get("kwargs", [])
        
        for inst_id, kwargs in zip(instruction_ids, kwargs_list):
            constraints.append(IFEvalConstraint(
                type=inst_id,
                params=kwargs if kwargs else {},
            ))
        
        return constraints
    
    def format_prompt(self, problem: Problem) -> str:
        """IFEval prompts are used as-is."""
        return problem.prompt
    
    def extract_answer(self, response: str) -> str:
        """For IFEval, the response itself is what we check."""
        return response
    
    def check_answer(self, predicted: str, gold: list[IFEvalConstraint]) -> bool:
        """Check if response satisfies all constraints."""
        if not gold:
            return True
            
        for constraint in gold:
            if not self._check_constraint(predicted, constraint):
                return False
        return True
    
    def check_answer_detailed(
        self, predicted: str, gold: list[IFEvalConstraint]
    ) -> dict[str, bool]:
        """Check each constraint individually."""
        results = {}
        for constraint in gold:
            results[constraint.type] = self._check_constraint(predicted, constraint)
        return results
    
    def _check_constraint(self, response: str, constraint: IFEvalConstraint) -> bool:
        """Check a single constraint."""
        
        ctype = constraint.type.lower()
        params = constraint.params
        
        # Word count constraints
        if "number_words" in ctype:
            word_count = len(response.split())
            if "at_least" in ctype:
                return word_count >= params.get("num_words", 0)
            elif "at_most" in ctype:
                return word_count <= params.get("num_words", float("inf"))
        
        # Sentence count constraints
        if "number_sentences" in ctype:
            # Simple sentence count by periods
            sentences = len([s for s in re.split(r'[.!?]+', response) if s.strip()])
            if "at_least" in ctype:
                return sentences >= params.get("num_sentences", 0)
            elif "at_most" in ctype:
                return sentences <= params.get("num_sentences", float("inf"))
        
        # Paragraph constraints
        if "number_paragraphs" in ctype:
            paragraphs = len([p for p in response.split("\n\n") if p.strip()])
            if "at_least" in ctype:
                return paragraphs >= params.get("num_paragraphs", 0)
            elif "at_most" in ctype:
                return paragraphs <= params.get("num_paragraphs", float("inf"))
        
        # Letter/character constraints
        if "number_letters" in ctype or "letter_count" in ctype:
            letter_count = sum(1 for c in response if c.isalpha())
            target = params.get("num_letters", params.get("letter_count", 0))
            if "at_least" in ctype:
                return letter_count >= target
            elif "at_most" in ctype:
                return letter_count <= target
        
        # Keyword inclusion
        if "include_keywords" in ctype or "keywords:existence" in ctype:
            keywords = params.get("keywords", [])
            response_lower = response.lower()
            return all(kw.lower() in response_lower for kw in keywords)
        
        # Keyword exclusion
        if "exclude_keywords" in ctype or "keywords:forbidden" in ctype:
            forbidden = params.get("keywords", [])
            response_lower = response.lower()
            return not any(kw.lower() in response_lower for kw in forbidden)
        
        # Frequency constraints
        if "keyword_frequency" in ctype or "keywords:frequency" in ctype:
            keyword = params.get("keyword", "")
            min_count = params.get("frequency", params.get("min_count", 0))
            max_count = params.get("max_count", float("inf"))
            actual = response.lower().count(keyword.lower())
            return min_count <= actual <= max_count
        
        # Format constraints
        if "response_format" in ctype:
            fmt = params.get("format", "").lower()
            if fmt == "json":
                return self._is_valid_json(response)
            elif fmt == "bullet":
                return bool(re.search(r"^[\s]*[-*•]", response, re.MULTILINE))
            elif fmt == "numbered":
                return bool(re.search(r"^[\s]*\d+[.\)]", response, re.MULTILINE))
        
        # Postscript
        if "postscript" in ctype:
            return bool(re.search(r"\bP\.?S\.?\b", response, re.IGNORECASE))
        
        # Placeholder
        if "placeholder" in ctype:
            return "[" in response and "]" in response
        
        # All caps
        if "all_caps" in ctype or "change_case:capital" in ctype:
            words = response.split()
            if words:
                return all(w.isupper() for w in words if w.isalpha())
        
        # All lowercase
        if "all_lowercase" in ctype or "change_case:lower" in ctype:
            words = response.split()
            if words:
                return all(w.islower() for w in words if w.isalpha())
        
        # Title case
        if "title_case" in ctype or "change_case:title" in ctype:
            # Check if response looks like title case
            words = response.split()
            if words:
                return all(w[0].isupper() for w in words if w and w[0].isalpha())
        
        # Quotation marks
        if "quotation" in ctype:
            return '"' in response or "'" in response or '"' in response or '"' in response
        
        # Section constraints
        if "section" in ctype:
            section_marker = params.get("section_spliter", params.get("section_marker", "---"))
            num_sections = params.get("num_sections", 1)
            sections = response.split(section_marker)
            return len([s for s in sections if s.strip()]) >= num_sections
        
        # Constraint not recognized - default to pass
        return True
    
    def _is_valid_json(self, text: str) -> bool:
        """Check if text contains valid JSON."""
        import json
        
        # Try to find JSON in the response
        # Look for content between { } or [ ]
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
            r'\{.*\}',  # Full object (greedy)
            r'\[.*\]',  # Full array (greedy)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return True
                except json.JSONDecodeError:
                    continue
        
        # Try the whole response
        try:
            json.loads(text.strip())
            return True
        except json.JSONDecodeError:
            return False

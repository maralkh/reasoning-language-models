"""Tree search decoding: Best-first and MCTS."""

from typing import Optional, Callable
from dataclasses import dataclass, field
import math
import random

from .base import BaseSampler, SampleResult
from ..models.base import BaseModel


@dataclass
class TreeNode:
    """A node in the search tree."""
    text: str  # Full text from root to this node
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    
    # Scores
    logprob: float = 0.0  # Log probability of path
    value: float = 0.0  # Value estimate (for MCTS)
    
    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0
    
    # State
    is_terminal: bool = False
    depth: int = 0
    
    @property
    def q_value(self) -> float:
        """Average value from this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def ucb_score(self, c: float = 1.414) -> float:
        """Upper Confidence Bound score for MCTS."""
        if self.visits == 0:
            return float("inf")
        
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return self.q_value + exploration


class BestFirstTreeSearch(BaseSampler):
    """
    Simple best-first tree search.
    
    Expands the highest-scoring frontier node at each step.
    Uses log probability as the scoring function.
    """
    
    def __init__(
        self,
        max_expansions: int = 50,
        branch_factor: int = 3,
        max_tokens: int = 512,
        tokens_per_step: int = 32,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ):
        """
        Initialize best-first search.
        
        Args:
            max_expansions: Maximum number of node expansions
            branch_factor: Number of children per expansion
            max_tokens: Maximum total tokens
            tokens_per_step: Tokens to generate per expansion
            temperature: Sampling temperature for expansions
            stop: Stop sequences
        """
        self.max_expansions = max_expansions
        self.branch_factor = branch_factor
        self.max_tokens = max_tokens
        self.tokens_per_step = tokens_per_step
        self.temperature = temperature
        self.stop = stop or []
    
    @property
    def name(self) -> str:
        return f"best_first_b{self.branch_factor}_e{self.max_expansions}"
    
    def sample(
        self,
        model: BaseModel,
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """Run best-first search for each prompt."""
        results = []
        
        for prompt in prompts:
            completed = self._search(model, prompt)
            
            # Sort by score and take top n
            completed.sort(key=lambda x: x.logprob, reverse=True)
            
            prompt_results = []
            for node in completed[:n]:
                prompt_results.append(SampleResult(
                    text=node.text,
                    logprob=node.logprob,
                    metadata={
                        "depth": node.depth,
                        "method": "best_first",
                    }
                ))
            
            # Pad with empty results if needed
            while len(prompt_results) < n:
                prompt_results.append(SampleResult(text="", logprob=float("-inf")))
            
            results.append(prompt_results)
        
        return results
    
    def _search(self, model: BaseModel, prompt: str) -> list[TreeNode]:
        """Run best-first search."""
        
        # Initialize root
        root = TreeNode(text="", logprob=0.0, depth=0)
        frontier = [root]  # Priority queue (we'll sort)
        completed = []
        
        for _ in range(self.max_expansions):
            if not frontier:
                break
            
            # Sort frontier by score (highest first)
            frontier.sort(key=lambda n: n.logprob, reverse=True)
            
            # Expand best node
            node = frontier.pop(0)
            
            # Check token limit
            approx_tokens = len(node.text.split())
            if approx_tokens >= self.max_tokens:
                node.is_terminal = True
                completed.append(node)
                continue
            
            # Generate children
            children = self._expand(model, prompt, node)
            
            for child in children:
                if child.is_terminal:
                    completed.append(child)
                else:
                    frontier.append(child)
        
        # Add remaining frontier nodes as completed
        completed.extend(frontier)
        
        return completed
    
    def _expand(self, model: BaseModel, prompt: str, node: TreeNode) -> list[TreeNode]:
        """Expand a node by generating continuations."""
        
        full_prompt = prompt + node.text
        
        # Generate multiple continuations
        outputs = model.generate(
            [full_prompt],
            max_tokens=self.tokens_per_step,
            temperature=self.temperature,
            n=self.branch_factor,
        )[0]
        
        children = []
        for output in outputs:
            continuation = output.text
            new_text = node.text + continuation
            
            # Check for stop sequences
            is_terminal = False
            for stop_seq in self.stop:
                if stop_seq in new_text:
                    new_text = new_text.split(stop_seq)[0]
                    is_terminal = True
                    break
            
            # Check for natural end
            if not continuation.strip():
                is_terminal = True
            
            child = TreeNode(
                text=new_text,
                parent=node,
                logprob=node.logprob + (output.logprob or 0.0),
                depth=node.depth + 1,
                is_terminal=is_terminal,
            )
            
            node.children.append(child)
            children.append(child)
        
        return children


class MCTSTreeSearch(BaseSampler):
    """
    Monte Carlo Tree Search for text generation.
    
    Uses UCB1 for selection, random rollouts for evaluation.
    Better exploration/exploitation balance than best-first.
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        branch_factor: int = 3,
        max_tokens: int = 512,
        tokens_per_step: int = 32,
        rollout_tokens: int = 64,
        temperature: float = 0.7,
        exploration_constant: float = 1.414,
        value_fn: Optional[Callable[[str], float]] = None,
        stop: Optional[list[str]] = None,
    ):
        """
        Initialize MCTS.
        
        Args:
            max_iterations: Number of MCTS iterations
            branch_factor: Number of children per expansion
            max_tokens: Maximum total tokens
            tokens_per_step: Tokens per expansion step
            rollout_tokens: Tokens for rollout simulation
            temperature: Sampling temperature
            exploration_constant: UCB exploration parameter (c)
            value_fn: Optional function to evaluate leaf nodes
                     If None, uses log probability as value
            stop: Stop sequences
        """
        self.max_iterations = max_iterations
        self.branch_factor = branch_factor
        self.max_tokens = max_tokens
        self.tokens_per_step = tokens_per_step
        self.rollout_tokens = rollout_tokens
        self.temperature = temperature
        self.c = exploration_constant
        self.value_fn = value_fn
        self.stop = stop or []
    
    @property
    def name(self) -> str:
        return f"mcts_i{self.max_iterations}_b{self.branch_factor}"
    
    def sample(
        self,
        model: BaseModel,
        prompts: list[str],
        n: int = 1,
    ) -> list[list[SampleResult]]:
        """Run MCTS for each prompt."""
        results = []
        
        for prompt in prompts:
            root, completed = self._search(model, prompt)
            
            # Get best paths
            best_nodes = self._get_best_paths(root, n)
            
            prompt_results = []
            for node in best_nodes:
                prompt_results.append(SampleResult(
                    text=node.text,
                    logprob=node.logprob,
                    metadata={
                        "visits": node.visits,
                        "q_value": node.q_value,
                        "depth": node.depth,
                        "method": "mcts",
                    }
                ))
            
            # Pad if needed
            while len(prompt_results) < n:
                prompt_results.append(SampleResult(text="", logprob=float("-inf")))
            
            results.append(prompt_results)
        
        return results
    
    def _search(self, model: BaseModel, prompt: str) -> tuple[TreeNode, list[TreeNode]]:
        """Run MCTS search."""
        
        root = TreeNode(text="", logprob=0.0, depth=0, visits=1)
        completed = []
        
        for _ in range(self.max_iterations):
            # Selection: traverse tree using UCB
            node = self._select(root)
            
            # Check if terminal
            if node.is_terminal:
                value = node.value
            else:
                # Expansion: add children
                if not node.children:
                    children = self._expand(model, prompt, node)
                    
                    # Check if any children are terminal
                    for child in children:
                        if child.is_terminal:
                            completed.append(child)
                    
                    if children:
                        node = random.choice(children)
                
                # Simulation: rollout to estimate value
                value = self._rollout(model, prompt, node)
            
            # Backpropagation: update statistics
            self._backpropagate(node, value)
        
        return root, completed
    
    def _select(self, root: TreeNode) -> TreeNode:
        """Select a node to expand using UCB1."""
        node = root
        
        while node.children:
            # If any child is unexplored, select it
            unexplored = [c for c in node.children if c.visits == 0]
            if unexplored:
                return random.choice(unexplored)
            
            # Otherwise, select by UCB score
            node = max(node.children, key=lambda c: c.ucb_score(self.c))
        
        return node
    
    def _expand(self, model: BaseModel, prompt: str, node: TreeNode) -> list[TreeNode]:
        """Expand a node by generating continuations."""
        
        # Check token limit
        approx_tokens = len(node.text.split())
        if approx_tokens >= self.max_tokens:
            node.is_terminal = True
            return []
        
        full_prompt = prompt + node.text
        
        # Generate continuations
        outputs = model.generate(
            [full_prompt],
            max_tokens=self.tokens_per_step,
            temperature=self.temperature,
            n=self.branch_factor,
        )[0]
        
        children = []
        for output in outputs:
            continuation = output.text
            new_text = node.text + continuation
            
            # Check stop sequences
            is_terminal = False
            for stop_seq in self.stop:
                if stop_seq in new_text:
                    new_text = new_text.split(stop_seq)[0]
                    is_terminal = True
                    break
            
            if not continuation.strip():
                is_terminal = True
            
            child = TreeNode(
                text=new_text,
                parent=node,
                logprob=node.logprob + (output.logprob or 0.0),
                depth=node.depth + 1,
                is_terminal=is_terminal,
            )
            
            node.children.append(child)
            children.append(child)
        
        return children
    
    def _rollout(self, model: BaseModel, prompt: str, node: TreeNode) -> float:
        """
        Simulate from node to estimate value.
        
        Returns a value between 0 and 1.
        """
        if node.is_terminal:
            # Use custom value function or log prob
            if self.value_fn:
                return self.value_fn(node.text)
            # Normalize log prob to [0, 1] range
            return self._normalize_logprob(node.logprob, node.depth)
        
        # Do a quick rollout
        full_prompt = prompt + node.text
        
        outputs = model.generate(
            [full_prompt],
            max_tokens=self.rollout_tokens,
            temperature=self.temperature,
            n=1,
        )[0]
        
        if outputs:
            rollout_text = node.text + outputs[0].text
            rollout_logprob = node.logprob + (outputs[0].logprob or 0.0)
            
            if self.value_fn:
                return self.value_fn(rollout_text)
            
            # Use normalized log probability
            depth = node.depth + len(outputs[0].text.split())
            return self._normalize_logprob(rollout_logprob, depth)
        
        return 0.5  # Default value
    
    def _normalize_logprob(self, logprob: float, length: int) -> float:
        """Normalize log probability to [0, 1] range."""
        if length == 0:
            return 0.5
        
        # Average log prob per token
        avg_logprob = logprob / max(length, 1)
        
        # Map to [0, 1] using sigmoid-like function
        # Typical log probs are in [-10, 0] range
        return 1 / (1 + math.exp(-avg_logprob - 2))
    
    def _backpropagate(self, node: TreeNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
    
    def _get_best_paths(self, root: TreeNode, n: int) -> list[TreeNode]:
        """Get n best complete paths from root."""
        
        # Collect all terminal nodes
        terminals = []
        
        def collect_terminals(node: TreeNode):
            if node.is_terminal or not node.children:
                terminals.append(node)
            for child in node.children:
                collect_terminals(child)
        
        collect_terminals(root)
        
        # Sort by Q-value (average returns)
        terminals.sort(key=lambda n: n.q_value, reverse=True)
        
        return terminals[:n]


class GuidedTreeSearch(MCTSTreeSearch):
    """
    MCTS with a value/reward model for guidance.
    
    Uses an external scoring function to evaluate partial generations.
    This is useful for steering generation toward desired properties.
    """
    
    def __init__(
        self,
        value_fn: Callable[[str], float],
        **kwargs
    ):
        """
        Initialize guided tree search.
        
        Args:
            value_fn: Function that takes text and returns a value in [0, 1].
                      Higher values = better generations.
            **kwargs: Additional arguments passed to MCTSTreeSearch
        """
        super().__init__(value_fn=value_fn, **kwargs)
    
    @property
    def name(self) -> str:
        return f"guided_mcts_i{self.max_iterations}"


# Convenience function to create a simple correctness-based value function
def make_answer_value_fn(
    extract_fn: Callable[[str], any],
    check_fn: Callable[[any, any], bool],
    gold_answer: any,
) -> Callable[[str], float]:
    """
    Create a value function based on answer correctness.
    
    Args:
        extract_fn: Function to extract answer from text
        check_fn: Function to check if answer is correct
        gold_answer: The correct answer
        
    Returns:
        Value function that returns 1.0 if correct, 0.0 otherwise
    """
    def value_fn(text: str) -> float:
        try:
            predicted = extract_fn(text)
            if check_fn(predicted, gold_answer):
                return 1.0
        except Exception:
            pass
        return 0.0
    
    return value_fn

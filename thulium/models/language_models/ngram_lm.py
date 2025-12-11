"""
N-gram Language Model for HTR decoding.

This module provides n-gram based language models that score
character or word sequences based on their statistical likelihood
in training text. N-gram models are simple, interpretable, and
require no GPU for inference, making them suitable for lightweight
deployment scenarios.

N-gram Order Guidelines
-----------------------
- Unigram (n=1): P(c) - character frequency only, no context
- Bigram (n=2): P(c_i | c_{i-1}) - single character context
- Trigram (n=3): P(c_i | c_{i-2}, c_{i-1}) - two character context
- 4-gram/5-gram: Useful for word-level modeling

For character-level HTR, trigrams to 5-grams are typically effective.

Smoothing
---------
Raw n-gram counts produce zero probabilities for unseen sequences.
This module implements several smoothing techniques:
- Add-k smoothing: Add constant to all counts
- Interpolation: Weighted combination of different orders
- Backoff: Fall back to lower-order models for unseen contexts
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math
import json
from pathlib import Path


class CharacterNGramLM:
    """
    Character-level n-gram language model with smoothing.
    
    This model estimates P(c_i | c_{i-n+1}, ..., c_{i-1}) using
    counts from training text and applies smoothing for robustness.
    
    Attributes:
        n: Maximum n-gram order.
        vocab: Set of known characters.
        counts: N-gram count dictionaries.
    
    Example:
        >>> lm = CharacterNGramLM(n=3)
        >>> lm.train(["hello world", "hello there"])
        >>> log_prob = lm.score([ord(c) for c in "hello"])
    """
    
    def __init__(
        self,
        n: int = 3,
        smoothing: str = 'interpolation',
        add_k: float = 0.01,
        interpolation_weights: Optional[List[float]] = None
    ):
        """
        Initialize n-gram language model.
        
        Args:
            n: Maximum n-gram order (e.g., 3 for trigrams).
            smoothing: Smoothing method ('add_k', 'interpolation', 'backoff').
            add_k: Constant for add-k smoothing.
            interpolation_weights: Weights for interpolation smoothing.
        """
        self.n = n
        self.smoothing = smoothing
        self.add_k = add_k
        
        if interpolation_weights is None:
            # Default: give more weight to higher-order n-grams
            interpolation_weights = [0.1 * (i + 1) for i in range(n)]
            total = sum(interpolation_weights)
            interpolation_weights = [w / total for w in interpolation_weights]
        self.interpolation_weights = interpolation_weights
        
        # Count dictionaries for each n-gram order
        # counts[order][context] = {token: count}
        self.counts: List[Dict] = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
        self.context_totals: List[Dict] = [defaultdict(int) for _ in range(n)]
        
        self.vocab: set = set()
        self.total_chars = 0
        
        # Special tokens
        self.BOS = -1  # Beginning of sequence
        self.EOS = -2  # End of sequence
    
    def train(self, texts: List[str]) -> None:
        """
        Train the n-gram model on a list of text strings.
        
        Args:
            texts: List of training text strings.
        """
        for text in texts:
            self._train_single(text)
    
    def _train_single(self, text: str) -> None:
        """Train on a single text string."""
        # Convert to token indices (using character codes)
        tokens = [ord(c) for c in text]
        self.vocab.update(tokens)
        self.total_chars += len(tokens)
        
        # Add BOS padding
        padded = [self.BOS] * (self.n - 1) + tokens + [self.EOS]
        
        # Count n-grams of each order
        for order in range(1, self.n + 1):
            for i in range(len(padded) - order + 1):
                context = tuple(padded[i:i + order - 1]) if order > 1 else ()
                token = padded[i + order - 1]
                
                self.counts[order - 1][context][token] += 1
                self.context_totals[order - 1][context] += 1
    
    def log_prob(self, token: int, context: Tuple[int, ...]) -> float:
        """
        Compute log probability of a token given context.
        
        Args:
            token: Token to score.
            context: Preceding tokens (tuple).
            
        Returns:
            Log probability.
        """
        if self.smoothing == 'add_k':
            return self._log_prob_add_k(token, context)
        elif self.smoothing == 'interpolation':
            return self._log_prob_interpolation(token, context)
        else:  # backoff
            return self._log_prob_backoff(token, context)
    
    def _log_prob_add_k(self, token: int, context: Tuple[int, ...]) -> float:
        """Add-k smoothed probability."""
        order = min(len(context) + 1, self.n)
        context = context[-(order - 1):] if order > 1 else ()
        
        count = self.counts[order - 1][context].get(token, 0)
        total = self.context_totals[order - 1].get(context, 0)
        vocab_size = len(self.vocab) + 2  # +2 for BOS/EOS
        
        prob = (count + self.add_k) / (total + self.add_k * vocab_size)
        return math.log(max(prob, 1e-10))
    
    def _log_prob_interpolation(self, token: int, context: Tuple[int, ...]) -> float:
        """Interpolation smoothed probability."""
        prob = 0.0
        
        for order in range(1, self.n + 1):
            ctx = context[-(order - 1):] if order > 1 else ()
            
            count = self.counts[order - 1][ctx].get(token, 0)
            total = self.context_totals[order - 1].get(ctx, 0)
            
            if total > 0:
                order_prob = count / total
            else:
                order_prob = 1.0 / (len(self.vocab) + 2)
            
            weight = self.interpolation_weights[order - 1]
            prob += weight * order_prob
        
        return math.log(max(prob, 1e-10))
    
    def _log_prob_backoff(self, token: int, context: Tuple[int, ...]) -> float:
        """Backoff smoothed probability."""
        for order in range(self.n, 0, -1):
            ctx = context[-(order - 1):] if order > 1 else ()
            
            if ctx in self.counts[order - 1] and token in self.counts[order - 1][ctx]:
                count = self.counts[order - 1][ctx][token]
                total = self.context_totals[order - 1][ctx]
                return math.log(count / total)
        
        # Fall back to uniform
        return math.log(1.0 / (len(self.vocab) + 2))
    
    def score(self, token_sequence: List[int], language: str = "en") -> float:
        """
        Compute log-probability of a token sequence.
        
        Args:
            token_sequence: List of token indices.
            language: Language code (unused, for interface compatibility).
            
        Returns:
            Log-probability of the sequence.
        """
        if len(token_sequence) == 0:
            return 0.0
        
        # Pad with BOS
        padded = [self.BOS] * (self.n - 1) + token_sequence
        
        total_log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[max(0, i - self.n + 1):i])
            token = padded[i]
            total_log_prob += self.log_prob(token, context)
        
        return total_log_prob
    
    def score_partial(
        self,
        prefix: List[int],
        next_token: int,
        state: Optional[Tuple[int, ...]] = None,
        language: str = "en"
    ) -> Tuple[float, Tuple[int, ...]]:
        """
        Score a partial sequence incrementally.
        
        Args:
            prefix: Current token sequence.
            next_token: Token to score.
            state: Previous context (last n-1 tokens).
            language: Language code.
            
        Returns:
            Tuple of (log-probability, new context state).
        """
        if state is None:
            # Build context from prefix
            if len(prefix) >= self.n - 1:
                context = tuple(prefix[-(self.n - 1):])
            else:
                context = tuple([self.BOS] * (self.n - 1 - len(prefix)) + prefix)
        else:
            context = state
        
        log_prob = self.log_prob(next_token, context)
        
        # Update context for next call
        new_context = tuple(list(context)[1:] + [next_token])
        
        return log_prob, new_context
    
    def save(self, path: str) -> None:
        """Save model to JSON file."""
        data = {
            'n': self.n,
            'smoothing': self.smoothing,
            'add_k': self.add_k,
            'interpolation_weights': self.interpolation_weights,
            'vocab': list(self.vocab),
            'total_chars': self.total_chars,
            'counts': [
                {str(k): dict(v) for k, v in order.items()}
                for order in self.counts
            ],
            'context_totals': [dict(order) for order in self.context_totals]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'CharacterNGramLM':
        """Load model from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lm = cls(
            n=data['n'],
            smoothing=data['smoothing'],
            add_k=data['add_k'],
            interpolation_weights=data['interpolation_weights']
        )
        
        lm.vocab = set(data['vocab'])
        lm.total_chars = data['total_chars']
        
        # Restore counts
        for order, counts in enumerate(data['counts']):
            for ctx_str, token_counts in counts.items():
                ctx = eval(ctx_str) if ctx_str else ()
                for token, count in token_counts.items():
                    lm.counts[order][ctx][int(token)] = count
        
        for order, totals in enumerate(data['context_totals']):
            for ctx_str, total in totals.items():
                ctx = eval(ctx_str) if ctx_str else ()
                lm.context_totals[order][ctx] = total
        
        return lm


class WordNGramLM:
    """
    Word-level n-gram language model.
    
    Similar to CharacterNGramLM but operates on word tokens.
    Useful for post-processing or word-level beam search.
    """
    
    def __init__(
        self,
        n: int = 3,
        smoothing: str = 'interpolation',
        add_k: float = 0.01
    ):
        self.n = n
        self.smoothing = smoothing
        self.add_k = add_k
        
        self.counts: List[Dict] = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
        self.context_totals: List[Dict] = [defaultdict(int) for _ in range(n)]
        self.vocab: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.BOS = '<s>'
        self.EOS = '</s>'
        self.UNK = '<unk>'
    
    def train(self, texts: List[str]) -> None:
        """Train on tokenized texts."""
        for text in texts:
            words = text.split()
            self._train_single(words)
    
    def _train_single(self, words: List[str]) -> None:
        """Train on a single word sequence."""
        for word in words:
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.idx_to_word[idx] = word
        
        # Add special tokens
        if self.BOS not in self.vocab:
            self.vocab[self.BOS] = len(self.vocab)
            self.idx_to_word[self.vocab[self.BOS]] = self.BOS
        if self.EOS not in self.vocab:
            self.vocab[self.EOS] = len(self.vocab)
            self.idx_to_word[self.vocab[self.EOS]] = self.EOS
        
        # Convert to indices
        tokens = [self.vocab[w] for w in words]
        padded = [self.vocab[self.BOS]] * (self.n - 1) + tokens + [self.vocab[self.EOS]]
        
        for order in range(1, self.n + 1):
            for i in range(len(padded) - order + 1):
                context = tuple(padded[i:i + order - 1]) if order > 1 else ()
                token = padded[i + order - 1]
                
                self.counts[order - 1][context][token] += 1
                self.context_totals[order - 1][context] += 1
    
    def score(self, words: List[str], language: str = "en") -> float:
        """Score a word sequence."""
        tokens = [self.vocab.get(w, self.vocab.get(self.UNK, 0)) for w in words]
        
        if len(tokens) == 0:
            return 0.0
        
        bos_idx = self.vocab.get(self.BOS, 0)
        padded = [bos_idx] * (self.n - 1) + tokens
        
        total_log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[max(0, i - self.n + 1):i])
            token = padded[i]
            
            # Simple add-k smoothing
            count = self.counts[min(len(context) + 1, self.n) - 1][context].get(token, 0)
            total = self.context_totals[min(len(context) + 1, self.n) - 1].get(context, 0)
            
            prob = (count + self.add_k) / (total + self.add_k * len(self.vocab))
            total_log_prob += math.log(max(prob, 1e-10))
        
        return total_log_prob

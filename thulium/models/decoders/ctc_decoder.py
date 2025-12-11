"""
CTC Decoder module for Connectionist Temporal Classification decoding.

This module provides neural network layers and decoding algorithms for CTC-based
handwriting text recognition, including:
- CTCDecoder: Linear projection layer with log-softmax for CTC loss computation
- CTCLanguageModelScorer: Abstract interface for language model integration
- Greedy decoding: Fast, approximate decoding
- Beam search decoding: Higher accuracy with optional language model rescoring

Mathematical Background
-----------------------
The CTC loss enables alignment-free training by marginalizing over all valid
alignments between input frames and target sequences. During decoding, we seek:

    y* = argmax_y P(y | x)

For beam search with language model integration:

    score(y) = log P_CTC(y | x) + alpha * log P_LM(y) + beta * |y|

where:
- P_CTC(y | x) is the CTC posterior probability
- P_LM(y) is the language model probability
- alpha is the LM weight coefficient
- beta is the word insertion bonus/penalty
- |y| is the sequence length
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLanguageModelScorer(Protocol):
    """
    Protocol defining the interface for language model scorers.
    
    Language model scorers provide log-probability scores for token sequences,
    enabling beam search decoding to incorporate linguistic knowledge.
    
    Implementations should handle:
    - Character-level scoring
    - Incremental state updates for efficient beam search
    - Language-specific vocabulary and tokenization
    """
    
    def score(self, token_sequence: List[int], language: str = "en") -> float:
        """
        Compute the log-probability score for a token sequence.
        
        Args:
            token_sequence: List of token indices representing the sequence.
            language: ISO 639-1 language code for language-specific models.
            
        Returns:
            Log-probability score (higher is better, typically negative).
        """
        ...
    
    def score_partial(
        self,
        prefix: List[int],
        next_token: int,
        state: Optional[object] = None,
        language: str = "en"
    ) -> Tuple[float, object]:
        """
        Score a partial sequence incrementally for efficient beam search.
        
        Args:
            prefix: Current token sequence (already scored).
            next_token: Next token to append.
            state: Optional cached state from previous scoring.
            language: ISO 639-1 language code.
            
        Returns:
            Tuple of (log-probability for next_token, updated state).
        """
        ...


@dataclass
class BeamSearchConfig:
    """
    Configuration for beam search decoding.
    
    Attributes:
        beam_width: Number of hypotheses to maintain at each step. Larger values
            improve accuracy but increase computation. Typical values: 5-100.
        lm_alpha: Language model weight coefficient. Controls the influence of
            the language model on decoding. Typical values: 0.5-2.0.
        lm_beta: Length penalty/bonus coefficient. Positive values encourage
            longer sequences. Typical values: 0.0-1.0.
        blank_index: Index of the CTC blank token in the vocabulary.
        length_normalization: If True, normalize final scores by sequence length
            to avoid preference for shorter sequences.
        min_score: Minimum log-probability threshold for pruning beams.
    """
    beam_width: int = 10
    lm_alpha: float = 1.0
    lm_beta: float = 0.0
    blank_index: int = 0
    length_normalization: bool = True
    min_score: float = -float('inf')


@dataclass
class BeamHypothesis:
    """
    Represents a single hypothesis in beam search.
    
    Attributes:
        tokens: Decoded token sequence (excluding blanks).
        score: Accumulated log-probability score.
        lm_state: Cached language model state for incremental scoring.
        prob_blank: Probability of ending in blank at current timestep.
        prob_non_blank: Probability of ending in non-blank at current timestep.
    """
    tokens: List[int]
    score: float
    lm_state: Optional[object] = None
    prob_blank: float = 0.0
    prob_non_blank: float = -float('inf')
    
    @property
    def combined_prob(self) -> float:
        """Log-sum-exp of blank and non-blank ending probabilities."""
        return _log_sum_exp(self.prob_blank, self.prob_non_blank)
    
    def __hash__(self):
        return hash(tuple(self.tokens))
    
    def __eq__(self, other):
        if isinstance(other, BeamHypothesis):
            return self.tokens == other.tokens
        return False


def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log-sum-exp for two values."""
    if a == -float('inf'):
        return b
    if b == -float('inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class CTCDecoder(nn.Module):
    """
    CTC Decoder layer with advanced decoding algorithms.
    
    This module provides:
    1. A linear projection from hidden states to vocabulary logits
    2. Log-softmax normalization for CTC loss computation
    3. Greedy decoding for fast inference
    4. Beam search decoding with optional language model integration
    
    The decoder supports configurable beam search parameters and can integrate
    with any language model implementing the CTCLanguageModelScorer protocol.
    
    Architecture:
        Input (B, T, H) -> Linear (H, V+1) -> LogSoftmax -> Output (B, T, V+1)
        
    where B=batch, T=sequence length, H=hidden size, V=vocabulary size.
    The +1 accounts for the CTC blank token.
    
    Example:
        >>> decoder = CTCDecoder(input_size=256, num_classes=100)
        >>> hidden_states = torch.randn(4, 50, 256)  # batch=4, seq=50
        >>> log_probs = decoder(hidden_states)  # (4, 50, 101)
        >>> decoded = decoder.decode_greedy(log_probs)  # List of token lists
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        blank_index: int = 0,
        dropout: float = 0.0
    ):
        """
        Initialize the CTC decoder.
        
        Args:
            input_size: Dimension of input hidden states.
            num_classes: Number of output classes (excluding blank).
            blank_index: Index for the CTC blank token. Default: 0.
            dropout: Dropout probability before projection. Default: 0.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.blank_index = blank_index
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # +1 for the blank token
        self.fc = nn.Linear(input_size, num_classes + 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for stable training."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probabilities for CTC loss.
        
        Args:
            x: Hidden states tensor of shape (B, T, H) where:
                B = batch size
                T = sequence length (time steps)
                H = hidden dimension
                
        Returns:
            Log-probabilities tensor of shape (B, T, V+1) where V is num_classes.
        """
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)
    
    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        blank_idx: Optional[int] = None
    ) -> List[List[int]]:
        """
        Perform greedy (best-path) CTC decoding.
        
        Greedy decoding selects the most probable token at each timestep,
        then collapses repeated tokens and removes blanks. This is fast
        but may not find the globally optimal sequence.
        
        Algorithm:
            1. Take argmax at each timestep
            2. Remove consecutive duplicates
            3. Remove blank tokens
        
        Args:
            log_probs: Log-probabilities of shape (B, T, V+1).
            blank_idx: Override blank index. Uses self.blank_index if None.
            
        Returns:
            List of decoded token sequences (one per batch element).
        """
        if blank_idx is None:
            blank_idx = self.blank_index
            
        predictions = torch.argmax(log_probs, dim=2)
        batch_results = []
        
        for batch_idx in range(predictions.size(0)):
            seq = predictions[batch_idx].cpu().numpy()
            decoded_tokens = []
            prev = None
            
            for token in seq:
                # Skip blanks and consecutive duplicates
                if token != prev and token != blank_idx:
                    decoded_tokens.append(int(token))
                prev = token
                
            batch_results.append(decoded_tokens)
            
        return batch_results
    
    def decode_beam_search(
        self,
        log_probs: torch.Tensor,
        config: Optional[BeamSearchConfig] = None,
        lm_scorer: Optional[CTCLanguageModelScorer] = None,
        language: str = "en"
    ) -> List[List[int]]:
        """
        Perform beam search CTC decoding with optional language model.
        
        Beam search maintains multiple hypotheses at each timestep, exploring
        a larger portion of the search space than greedy decoding. With
        language model integration, the decoder can incorporate linguistic
        constraints to improve accuracy.
        
        The scoring function combines CTC probabilities with LM scores:
        
            score = log P_CTC + alpha * log P_LM + beta * length
        
        Args:
            log_probs: Log-probabilities of shape (B, T, V+1).
            config: Beam search configuration. Uses defaults if None.
            lm_scorer: Optional language model scorer implementing
                CTCLanguageModelScorer protocol.
            language: Language code for LM scoring.
            
        Returns:
            List of decoded token sequences (one per batch element).
        """
        if config is None:
            config = BeamSearchConfig()
        
        batch_size = log_probs.size(0)
        results = []
        
        for batch_idx in range(batch_size):
            seq_log_probs = log_probs[batch_idx].cpu().numpy()
            decoded = self._beam_search_single(
                seq_log_probs, config, lm_scorer, language
            )
            results.append(decoded)
        
        return results
    
    def _beam_search_single(
        self,
        log_probs,  # numpy array (T, V+1)
        config: BeamSearchConfig,
        lm_scorer: Optional[CTCLanguageModelScorer],
        language: str
    ) -> List[int]:
        """
        Beam search decoding for a single sequence.
        
        Implements the CTC prefix beam search algorithm with language model
        integration. Uses log-space arithmetic for numerical stability.
        """
        T, V = log_probs.shape
        blank = config.blank_index
        
        # Initialize with empty hypothesis
        beams = {
            tuple(): BeamHypothesis(
                tokens=[],
                score=0.0,
                prob_blank=0.0,
                prob_non_blank=-float('inf')
            )
        }
        
        for t in range(T):
            new_beams = {}
            
            for prefix_tuple, hyp in beams.items():
                prefix = list(prefix_tuple)
                
                # Probability of staying in current state (blank)
                p_blank = log_probs[t, blank]
                
                # Extend with blank: can follow either blank or non-blank
                new_prob_blank = _log_sum_exp(
                    hyp.prob_blank + p_blank,
                    hyp.prob_non_blank + p_blank
                )
                
                key = tuple(prefix)
                if key not in new_beams:
                    new_beams[key] = BeamHypothesis(
                        tokens=prefix,
                        score=0.0,
                        lm_state=hyp.lm_state,
                        prob_blank=new_prob_blank,
                        prob_non_blank=-float('inf')
                    )
                else:
                    new_beams[key].prob_blank = _log_sum_exp(
                        new_beams[key].prob_blank, new_prob_blank
                    )
                
                # Extend with each non-blank token
                for c in range(V):
                    if c == blank:
                        continue
                    
                    p_char = log_probs[t, c]
                    
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same character as last: can only follow blank
                        new_prob_non_blank = hyp.prob_blank + p_char
                    else:
                        # Different character: can follow either
                        new_prob_non_blank = _log_sum_exp(
                            hyp.prob_blank + p_char,
                            hyp.prob_non_blank + p_char
                        )
                    
                    new_prefix = prefix + [c]
                    new_key = tuple(new_prefix)
                    
                    # Apply language model scoring if available
                    lm_score = 0.0
                    new_lm_state = hyp.lm_state
                    if lm_scorer is not None:
                        try:
                            lm_score, new_lm_state = lm_scorer.score_partial(
                                prefix, c, hyp.lm_state, language
                            )
                            lm_score *= config.lm_alpha
                        except (AttributeError, NotImplementedError):
                            # Fallback to full sequence scoring
                            lm_score = lm_scorer.score(new_prefix, language)
                            lm_score *= config.lm_alpha
                    
                    # Add length bonus
                    length_bonus = config.lm_beta
                    
                    adjusted_prob = new_prob_non_blank + lm_score + length_bonus
                    
                    if new_key not in new_beams:
                        new_beams[new_key] = BeamHypothesis(
                            tokens=new_prefix,
                            score=0.0,
                            lm_state=new_lm_state,
                            prob_blank=-float('inf'),
                            prob_non_blank=adjusted_prob
                        )
                    else:
                        new_beams[new_key].prob_non_blank = _log_sum_exp(
                            new_beams[new_key].prob_non_blank,
                            adjusted_prob
                        )
            
            # Compute combined scores and prune to beam width
            for key, hyp in new_beams.items():
                hyp.score = hyp.combined_prob
            
            # Sort by score and keep top beams
            sorted_beams = sorted(
                new_beams.items(),
                key=lambda x: x[1].score,
                reverse=True
            )
            
            beams = dict(sorted_beams[:config.beam_width])
        
        # Return best hypothesis
        if not beams:
            return []
        
        best_key = max(beams.keys(), key=lambda k: beams[k].score)
        best_hyp = beams[best_key]
        
        # Apply length normalization if configured
        if config.length_normalization and len(best_hyp.tokens) > 0:
            # Re-rank with length normalization
            best_score = -float('inf')
            best_tokens = []
            for key, hyp in beams.items():
                normalized_score = hyp.score / max(1, len(hyp.tokens))
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_tokens = hyp.tokens
            return best_tokens
        
        return best_hyp.tokens


class NullLanguageModelScorer:
    """
    Null implementation of language model scorer.
    
    Returns zero scores for all sequences, effectively disabling
    language model influence in beam search. Useful for ablation
    studies or when no language model is available.
    """
    
    def score(self, token_sequence: List[int], language: str = "en") -> float:
        """Return zero score (no language model influence)."""
        return 0.0
    
    def score_partial(
        self,
        prefix: List[int],
        next_token: int,
        state: Optional[object] = None,
        language: str = "en"
    ) -> Tuple[float, object]:
        """Return zero score with no state."""
        return 0.0, None

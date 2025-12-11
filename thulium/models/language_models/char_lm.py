"""
Character-level Language Model for HTR decoding.

This module provides character-level language models that can be
integrated with CTC beam search decoding to improve recognition
accuracy by incorporating linguistic knowledge.

Language models provide P(y) - the prior probability of character
sequences - which is combined with acoustic/visual model outputs
P(x|y) during decoding via Bayes' rule or log-linear interpolation.

Mathematical Formulation
------------------------
During beam search, the combined score is:

    score(y) = log P_HTR(y|x) + alpha * log P_LM(y) + beta * |y|

where:
- P_HTR(y|x) is the HTR model probability (from CTC)
- P_LM(y) is the language model probability
- alpha is the LM weight (hyperparameter)
- beta is the word/character insertion bonus
- |y| is the sequence length
"""

from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelScorer(ABC):
    """
    Abstract base class for language model scorers.
    
    Language model scorers provide log-probability scores for token
    sequences, enabling beam search decoding to incorporate
    linguistic knowledge.
    """
    
    @abstractmethod
    def score(self, token_sequence: List[int], language: str = "en") -> float:
        """
        Compute the log-probability score for a token sequence.
        
        Args:
            token_sequence: List of token indices.
            language: Language code for language-specific models.
            
        Returns:
            Log-probability score (typically negative).
        """
        pass
    
    @abstractmethod
    def score_partial(
        self,
        prefix: List[int],
        next_token: int,
        state: Optional[object] = None,
        language: str = "en"
    ) -> Tuple[float, object]:
        """
        Score a partial sequence incrementally.
        
        Args:
            prefix: Current token sequence.
            next_token: Next token to append.
            state: Cached state from previous scoring.
            language: Language code.
            
        Returns:
            Tuple of (log-probability for next_token, updated state).
        """
        pass


class CharacterLM(nn.Module, LanguageModelScorer):
    """
    Neural character-level language model using LSTM.
    
    This model learns character-level patterns in text and can
    provide probability estimates for character sequences. It's
    trained separately on text data (not handwriting images) and
    then used during decoding to rescore hypotheses.
    
    Architecture:
        Character Embedding -> LSTM -> Linear -> Softmax
    
    Example:
        >>> lm = CharacterLM(vocab_size=100, hidden_size=256)
        >>> sequence = [5, 10, 15, 20]  # Token indices
        >>> log_prob = lm.score(sequence)
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 128,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        """
        Initialize character language model.
        
        Args:
            vocab_size: Size of the character vocabulary.
            embedding_dim: Dimension of character embeddings.
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            tie_weights: If True, tie embedding and output weights.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output projection
        if tie_weights and embedding_dim == hidden_size:
            self.fc = nn.Linear(hidden_size, vocab_size)
            self.fc.weight = self.embedding.weight
        else:
            self.proj = nn.Linear(hidden_size, embedding_dim) if hidden_size != embedding_dim else nn.Identity()
            self.fc = nn.Linear(embedding_dim, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass returning logits for next character prediction.
        
        Args:
            x: Input token indices of shape (B, Seq).
            hidden: Optional initial hidden state.
            
        Returns:
            Tuple of (logits (B, Seq, vocab_size), final hidden state).
        """
        embed = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embed, hidden)
        
        if hasattr(self, 'proj'):
            output = self.proj(output)
        output = self.fc(output)
        
        return output, hidden
    
    @torch.no_grad()
    def score(self, token_sequence: List[int], language: str = "en") -> float:
        """
        Compute log-probability of a character sequence.
        
        Uses the chain rule: P(w) = prod P(c_i | c_{<i})
        In log space: log P(w) = sum log P(c_i | c_{<i})
        
        Args:
            token_sequence: List of token indices.
            language: Language code (not used in base implementation).
            
        Returns:
            Log-probability of the sequence.
        """
        if len(token_sequence) < 2:
            return 0.0
        
        # Prepare input (all tokens except last)
        input_tokens = torch.tensor([token_sequence[:-1]], dtype=torch.long)
        target_tokens = token_sequence[1:]
        
        # Get predictions
        logits, _ = self.forward(input_tokens)
        log_probs = F.log_softmax(logits[0], dim=-1)
        
        # Sum log probabilities for each predicted token
        total_log_prob = 0.0
        for i, target in enumerate(target_tokens):
            total_log_prob += log_probs[i, target].item()
        
        return total_log_prob
    
    @torch.no_grad()
    def score_partial(
        self,
        prefix: List[int],
        next_token: int,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        language: str = "en"
    ) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Score a partial sequence incrementally.
        
        This is more efficient for beam search as it reuses the
        hidden state from previous computations.
        
        Args:
            prefix: Current token sequence.
            next_token: Next token to score.
            state: Previous LSTM hidden state.
            language: Language code.
            
        Returns:
            Tuple of (log-probability for next_token, new hidden state).
        """
        if len(prefix) == 0:
            # No context, return uniform log probability
            return -torch.log(torch.tensor(float(self.vocab_size))).item(), None
        
        # Get prediction for next token given last token of prefix
        last_token = torch.tensor([[prefix[-1]]], dtype=torch.long)
        logits, new_state = self.forward(last_token, state)
        log_probs = F.log_softmax(logits[0, 0], dim=-1)
        
        return log_probs[next_token].item(), new_state
    
    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state to zeros."""
        weight = next(self.parameters())
        h0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        return h0, c0


class TransformerCharLM(nn.Module, LanguageModelScorer):
    """
    Transformer-based character language model.
    
    Uses self-attention instead of recurrence, which can model
    longer-range dependencies but requires quadratic memory in
    sequence length.
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(max_seq_len))
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        B, T = x.shape
        
        tok_emb = self.embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        x = self.transformer(x, mask=self.causal_mask[:T, :T])
        logits = self.fc(x)
        
        return logits
    
    @torch.no_grad()
    def score(self, token_sequence: List[int], language: str = "en") -> float:
        """Compute log-probability of a character sequence."""
        if len(token_sequence) < 2:
            return 0.0
        
        input_tokens = torch.tensor([token_sequence[:-1]], dtype=torch.long)
        target_tokens = token_sequence[1:]
        
        logits = self.forward(input_tokens)
        log_probs = F.log_softmax(logits[0], dim=-1)
        
        total_log_prob = 0.0
        for i, target in enumerate(target_tokens):
            total_log_prob += log_probs[i, target].item()
        
        return total_log_prob
    
    @torch.no_grad()
    def score_partial(
        self,
        prefix: List[int],
        next_token: int,
        state: Optional[object] = None,
        language: str = "en"
    ) -> Tuple[float, object]:
        """Score incrementally (recomputes full sequence each time)."""
        if len(prefix) == 0:
            return -torch.log(torch.tensor(float(self.vocab_size))).item(), None
        
        input_tokens = torch.tensor([prefix], dtype=torch.long)
        logits = self.forward(input_tokens)
        log_probs = F.log_softmax(logits[0, -1], dim=-1)
        
        return log_probs[next_token].item(), None

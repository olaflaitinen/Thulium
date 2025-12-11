"""
Attention-based Decoder for Sequence-to-Sequence HTR.

This module provides an autoregressive decoder using cross-attention
for handwriting text recognition. Unlike CTC-based approaches that
require monotonic alignment assumptions, the attention decoder can
learn flexible alignments between input features and output tokens.

Architecture Overview
---------------------
The decoder follows the Transformer decoder architecture:

1. Token Embedding: Maps input token indices to dense vectors
2. Positional Encoding: Adds position information to embeddings
3. Masked Self-Attention: Attends to previously generated tokens
4. Cross-Attention: Attends to encoder features (image features)
5. Feed-Forward Network: Non-linear transformation
6. Linear Projection: Maps to vocabulary logits

Mathematical Formulation
------------------------
At each decoding step t, given encoder outputs H and previous tokens y_{<t}:

    h_t = Decoder(y_{<t}, H)
    P(y_t | y_{<t}, H) = softmax(W_o * h_t)

The decoder uses causal masking to ensure each position can only
attend to earlier positions during training (teacher forcing).
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    
    Adds position-dependent signals to input embeddings using sine and
    cosine functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionDecoder(nn.Module):
    """
    Transformer-based autoregressive decoder for sequence-to-sequence HTR.
    
    This decoder implements the standard Transformer decoder architecture
    with cross-attention to encoder features and causal self-attention.
    
    Example:
        >>> decoder = AttentionDecoder(vocab_size=100, d_model=256, encoder_dim=512)
        >>> encoder_out = torch.randn(4, 50, 512)
        >>> target_tokens = torch.randint(0, 100, (4, 20))
        >>> logits = decoder(target_tokens, encoder_out)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        encoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embedding_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.encoder_proj = (
            nn.Linear(encoder_dim, d_model)
            if encoder_dim != d_model
            else nn.Identity()
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()
    
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            tgt: Target token indices of shape (B, T_tgt).
            encoder_out: Encoder outputs of shape (B, T_enc, encoder_dim).
            tgt_padding_mask: Padding mask for targets.
            encoder_padding_mask: Padding mask for encoder.
            
        Returns:
            Logits tensor of shape (B, T_tgt, vocab_size).
        """
        B, T = tgt.shape
        
        x = self.embedding(tgt) * self.embedding_scale
        x = self.pos_encoding(x)
        memory = self.encoder_proj(encoder_out)
        tgt_mask = self._generate_causal_mask(T, tgt.device)
        
        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        return self.output_proj(output)
    
    @torch.no_grad()
    def decode_greedy(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None
    ) -> List[List[int]]:
        """Greedy autoregressive decoding."""
        if max_len is None:
            max_len = self.max_seq_len
        
        B = encoder_out.size(0)
        device = encoder_out.device
        
        generated = torch.full(
            (B, 1), self.sos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            logits = self.forward(
                generated, encoder_out,
                encoder_padding_mask=encoder_padding_mask
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
            
            finished = finished | (next_tokens.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break
        
        results = []
        for i in range(B):
            tokens = generated[i].tolist()
            if tokens[0] == self.sos_token_id:
                tokens = tokens[1:]
            if self.eos_token_id in tokens:
                tokens = tokens[:tokens.index(self.eos_token_id)]
            results.append(tokens)
        
        return results
    
    @torch.no_grad()
    def decode_beam_search(
        self,
        encoder_out: torch.Tensor,
        beam_width: int = 5,
        max_len: Optional[int] = None,
        length_penalty: float = 1.0,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Beam search decoding for improved accuracy."""
        if max_len is None:
            max_len = self.max_seq_len
        
        B = encoder_out.size(0)
        results = []
        
        for b in range(B):
            enc_out_b = encoder_out[b:b+1]
            enc_mask_b = (
                encoder_padding_mask[b:b+1]
                if encoder_padding_mask is not None
                else None
            )
            
            result = self._beam_search_single(
                enc_out_b, beam_width, max_len, length_penalty, enc_mask_b
            )
            results.append(result)
        
        return results
    
    def _beam_search_single(
        self,
        encoder_out: torch.Tensor,
        beam_width: int,
        max_len: int,
        length_penalty: float,
        encoder_padding_mask: Optional[torch.Tensor]
    ) -> List[int]:
        """Beam search for a single sequence."""
        device = encoder_out.device
        
        enc_expanded = encoder_out.expand(beam_width, -1, -1)
        mask_expanded = (
            encoder_padding_mask.expand(beam_width, -1)
            if encoder_padding_mask is not None
            else None
        )
        
        beams = torch.full(
            (beam_width, 1), self.sos_token_id, dtype=torch.long, device=device
        )
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = -float('inf')
        
        finished_beams = []
        finished_scores = []
        
        for step in range(max_len - 1):
            logits = self.forward(
                beams, enc_expanded,
                encoder_padding_mask=mask_expanded
            )
            next_logits = F.log_softmax(logits[:, -1, :], dim=-1)
            
            vocab_size = next_logits.size(-1)
            scores = beam_scores.unsqueeze(1) + next_logits
            scores = scores.view(-1)
            
            top_scores, top_indices = scores.topk(beam_width * 2)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            new_beams = []
            new_scores = []
            
            for score, beam_idx, token_idx in zip(
                top_scores, beam_indices, token_indices
            ):
                if len(new_beams) >= beam_width:
                    break
                
                beam_idx = beam_idx.item()
                token_idx = token_idx.item()
                score = score.item()
                
                new_seq = torch.cat([
                    beams[beam_idx],
                    torch.tensor([token_idx], device=device)
                ])
                
                if token_idx == self.eos_token_id:
                    final_score = score / (len(new_seq) ** length_penalty)
                    finished_beams.append(new_seq[1:-1].tolist())
                    finished_scores.append(final_score)
                else:
                    new_beams.append(new_seq)
                    new_scores.append(score)
            
            if not new_beams:
                break
            
            max_new_len = max(len(b) for b in new_beams)
            padded_beams = []
            for b in new_beams:
                if len(b) < max_new_len:
                    padding = torch.full(
                        (max_new_len - len(b),),
                        self.pad_token_id,
                        device=device
                    )
                    b = torch.cat([b, padding])
                padded_beams.append(b)
            
            beams = torch.stack(padded_beams[:beam_width])
            beam_scores = torch.tensor(new_scores[:beam_width], device=device)
            
            actual_beams = beams.size(0)
            enc_expanded = encoder_out.expand(actual_beams, -1, -1)
            if encoder_padding_mask is not None:
                mask_expanded = encoder_padding_mask.expand(actual_beams, -1)
        
        if finished_beams:
            best_idx = max(range(len(finished_scores)), key=lambda i: finished_scores[i])
            return finished_beams[best_idx]
        else:
            best_beam = beams[0, 1:].tolist()
            if self.pad_token_id in best_beam:
                best_beam = best_beam[:best_beam.index(self.pad_token_id)]
            return best_beam

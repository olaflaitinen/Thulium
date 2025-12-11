"""
Transformer sequence heads for HTR models.

This module provides Transformer-based sequence modeling components
that can be used as alternatives to LSTM heads for processing
features from CNN or ViT backbones.

Transformer heads offer advantages for HTR:
- Better parallelization during training
- Explicit modeling of long-range dependencies
- No sequential bottleneck like RNNs

Architecture
------------
Input features (B, Seq, D) are processed through:
1. Optional input projection
2. Positional encoding
3. Stack of Transformer encoder layers
4. Layer normalization
5. Optional output projection

Each encoder layer consists of:
- Multi-head self-attention
- Feed-forward network
- Layer normalization and residual connections
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer sequence heads.
    
    Uses sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
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
        """Add positional encoding and apply dropout."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Transformers.
    
    Uses learned position embeddings instead of fixed sinusoidal patterns.
    This can be more flexible but requires seeing all positions during training.
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding."""
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        return self.dropout(x)


class TransformerHead(nn.Module):
    """
    Transformer Encoder as a sequence head for HTR.
    
    Replaces BiLSTMs with self-attention for capturing long-range
    dependencies in the feature sequence. Particularly effective
    for high-resolution line images with many timesteps.
    
    Attributes:
        output_dim: Dimension of output features.
    
    Example:
        >>> head = TransformerHead(input_size=256, hidden_size=256, num_layers=4)
        >>> x = torch.randn(4, 50, 256)  # (B, Seq, D)
        >>> output = head(x)  # (B, Seq, 256)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pos_encoding: str = 'sinusoidal',
        output_dim: Optional[int] = None,
        pre_norm: bool = True
    ):
        """
        Initialize Transformer head.
        
        Args:
            input_size: Dimension of input features.
            hidden_size: Transformer model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: FFN intermediate dimension. Defaults to 4 * hidden_size.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for positional encoding.
            pos_encoding: Type of positional encoding ('sinusoidal' or 'learnable').
            output_dim: Output dimension. Uses hidden_size if None.
            pre_norm: If True, use pre-layer normalization (more stable).
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_dim = output_dim or hidden_size
        
        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4
        
        # Input projection if needed
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        
        # Positional encoding
        if pos_encoding == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(
                hidden_size, max_seq_len, dropout
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                hidden_size, max_seq_len, dropout
            )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=pre_norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.output_proj = (
            nn.Linear(hidden_size, self.output_dim)
            if hidden_size != self.output_dim
            else nn.Identity()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Process sequence through Transformer encoder.
        
        Args:
            x: Input tensor of shape (B, Seq, input_size).
            mask: Optional attention mask.
            return_attention: If True, also return attention weights.
            
        Returns:
            Output tensor of shape (B, Seq, output_dim).
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.output_dim


class ConformerBlock(nn.Module):
    """
    Conformer block combining convolution and self-attention.
    
    The Conformer architecture combines the local modeling of
    convolutions with the global modeling of self-attention,
    which is particularly effective for sequence recognition.
    
    Block structure:
    1. Feed-forward module (first half)
    2. Multi-head self-attention
    3. Convolution module
    4. Feed-forward module (second half)
    5. Layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        depthwise: bool = True
    ):
        super().__init__()
        
        # First FFN half
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),  # Swish activation
            nn.Dropout(p=dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=dropout)
        )
        
        # Self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        padding = (conv_kernel_size - 1) // 2
        
        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model * 2, kernel_size=1),  # Pointwise expansion
                nn.GLU(dim=1),  # Gated activation
                nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                         padding=padding, groups=d_model),  # Depthwise
                nn.BatchNorm1d(d_model),
                nn.SiLU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),  # Pointwise
                nn.Dropout(p=dropout)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding),
                nn.BatchNorm1d(d_model),
                nn.SiLU(),
                nn.Dropout(p=dropout)
            )
        
        # Second FFN half
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=dropout)
        )
        
        # Final norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through Conformer block."""
        # First FFN (scaled by 0.5)
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.attn_dropout(attn_out)
        
        # Convolution
        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)  # (B, D, Seq) for Conv1d
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, Seq, D)
        x = residual + x
        
        # Second FFN (scaled by 0.5)
        x = x + 0.5 * self.ffn2(x)
        
        # Final norm
        x = self.norm(x)
        
        return x


class ConformerHead(nn.Module):
    """
    Conformer sequence head combining convolution and self-attention.
    
    Stacks multiple Conformer blocks for sequence modeling.
    Particularly effective for speech and handwriting recognition.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size, dropout=dropout)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=hidden_size,
                nhead=nhead,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through Conformer stack."""
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        for block in self.blocks:
            x = block(x)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim

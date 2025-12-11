"""
Sequence modeling heads for HTR models.

This module provides sequence modeling components that process
features from CNN or ViT backbones into sequence representations
suitable for CTC or attention-based decoding.

LSTM heads are effective for modeling sequential dependencies
in handwriting, especially when the backbone produces 2D feature
maps that need to be converted to 1D sequences.

Architecture Flow
-----------------
For CNN backbone output (B, C, H, W):
1. Collapse height dimension (pooling or projection)
2. Transpose to sequence format (B, W, C*H or B, W, C)
3. Apply Bi-directional LSTM layers
4. Output: (B, W, hidden*2) for bidirectional

For ViT backbone output (B, Seq, D):
1. Already in sequence format
2. Apply LSTM layers directly
3. Output: (B, Seq, hidden*2)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class HeightPoolProjection(nn.Module):
    """
    Project 2D CNN features to 1D sequence by pooling height dimension.
    
    This module converts (B, C, H, W) feature maps to (B, W, D) sequences
    suitable for recurrent or attention-based sequence modeling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        pool_type: str = 'mean'
    ):
        """
        Initialize height pooling projection.
        
        Args:
            in_channels: Number of input feature channels.
            out_dim: Output sequence dimension.
            pool_type: Pooling type ('mean', 'max', or 'adaptive').
        """
        super().__init__()
        
        self.pool_type = pool_type
        if pool_type == 'adaptive':
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.proj = nn.Linear(in_channels, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert 2D features to 1D sequence.
        
        Args:
            x: Feature map of shape (B, C, H, W).
            
        Returns:
            Sequence tensor of shape (B, W, out_dim).
        """
        B, C, H, W = x.shape
        
        if self.pool_type == 'mean':
            x = x.mean(dim=2)  # (B, C, W)
        elif self.pool_type == 'max':
            x = x.max(dim=2)[0]  # (B, C, W)
        else:  # adaptive
            x = self.pool(x).squeeze(2)  # (B, C, W)
        
        x = x.transpose(1, 2)  # (B, W, C)
        x = self.proj(x)  # (B, W, out_dim)
        
        return x


class BiLSTMHead(nn.Module):
    """
    Bidirectional LSTM sequence head for HTR.
    
    Processes feature sequences using bidirectional LSTMs to capture
    both left-to-right and right-to-left context. This is particularly
    effective for handwriting where context from both directions
    helps resolve ambiguous characters.
    
    The output dimension is hidden_size * 2 due to bidirectionality.
    
    Attributes:
        output_dim: Dimension of output features (hidden_size * 2).
    
    Example:
        >>> head = BiLSTMHead(input_size=256, hidden_size=256, num_layers=2)
        >>> x = torch.randn(4, 50, 256)  # (B, Seq, InputSize)
        >>> output, (h_n, c_n) = head(x)  # output: (B, Seq, 512)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        """
        Initialize BiLSTM head.
        
        Args:
            input_size: Dimension of input features.
            hidden_size: LSTM hidden state dimension (per direction).
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between LSTM layers (applied when num_layers > 1).
            batch_first: If True, input shape is (B, Seq, D).
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = hidden_size * 2  # Bidirectional
        
        # Optional input projection if dimensions don't match
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence through bidirectional LSTM.
        
        Args:
            x: Input tensor of shape (B, Seq, input_size).
            hidden: Optional initial hidden state (h_0, c_0).
            
        Returns:
            Tuple of:
            - Output tensor of shape (B, Seq, hidden_size * 2)
            - Final hidden state tuple (h_n, c_n)
        """
        x = self.input_proj(x)
        output, hidden = self.lstm(x, hidden)
        output = self.layer_norm(output)
        
        return output, hidden
    
    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.output_dim


class StackedLSTMHead(nn.Module):
    """
    Stacked LSTM head with residual connections.
    
    Uses multiple LSTM layers with optional skip connections
    for improved gradient flow in deeper networks.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
        residual: bool = True
    ):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.residual = residual
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Stacked LSTMs
        self.lstms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            lstm_input = hidden_size if i == 0 or not residual else self.output_dim
            if i > 0 and residual:
                lstm_input = hidden_size  # After residual projection
            
            self.lstms.append(nn.LSTM(
                input_size=hidden_size if not bidirectional else (
                    hidden_size if i == 0 else self.output_dim
                ),
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            ))
            self.dropouts.append(nn.Dropout(p=dropout))
            self.layer_norms.append(nn.LayerNorm(self.output_dim))
        
        # Residual projections if needed
        if residual and num_layers > 1:
            self.residual_projs = nn.ModuleList([
                nn.Linear(hidden_size, self.output_dim)
                if i == 0 else nn.Identity()
                for i in range(num_layers)
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence through stacked LSTMs with residual connections."""
        x = self.input_proj(x)
        
        for i, (lstm, dropout, ln) in enumerate(
            zip(self.lstms, self.dropouts, self.layer_norms)
        ):
            residual = x
            if i == 0:
                # First layer: no residual input yet
                x, _ = lstm(x)
            else:
                x, _ = lstm(x)
                if self.residual:
                    x = x + residual
            
            x = ln(x)
            x = dropout(x)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim


class LSTMWithAttention(nn.Module):
    """
    LSTM with self-attention for enhanced sequence modeling.
    
    Combines the sequential modeling of LSTMs with the global
    context modeling of self-attention.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = hidden_size * 2
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.output_dim)
        self.norm2 = nn.LayerNorm(self.output_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.output_dim * 4, self.output_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process with LSTM followed by self-attention."""
        x = self.input_proj(x)
        
        # BiLSTM
        x, _ = self.lstm(x)
        x = self.norm1(x)
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        # FFN with residual
        x = x + self.ffn(x)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim

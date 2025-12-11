"""
Vision Transformer (ViT) backbone for feature extraction in HTR models.

This module provides Vision Transformer-based backbones optimized for
handwriting text recognition. Unlike standard ViT for classification,
these variants handle variable-width line images and produce sequence
features suitable for CTC or attention-based decoding.

Architecture Overview
---------------------
The ViT backbone for HTR consists of:

1. Patch Embedding: Splits image into fixed-size patches, projects to d_model
2. Positional Encoding: Learnable or sinusoidal position embeddings
3. Transformer Encoder: Self-attention layers for global context modeling
4. Output Projection: Optional projection to desired feature dimension

For variable-width inputs, the patch sequence length varies with image width,
making this naturally suitable for sequence modeling in HTR.

Mathematical Formulation
------------------------
Given input image x of shape (B, C, H, W):

1. Extract patches: x_p of shape (B, N_patches, patch_dim)
   where N_patches = (H / patch_h) * (W / patch_w)

2. Project to d_model: z_0 = x_p * E + pos_embed
   where E is the projection matrix

3. Apply L transformer layers:
   z_l = TransformerBlock(z_{l-1})

4. Output features: z_L of shape (B, N_patches, d_model)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    Splits the input image into non-overlapping patches and projects
    each patch to a embedding vector of dimension d_model.
    
    For HTR with variable-width images, the number of patches in the
    width dimension varies, producing variable-length sequences.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: Tuple[int, int] = (16, 16),
        d_model: int = 768,
        flatten: bool = True
    ):
        """
        Initialize patch embedding.
        
        Args:
            in_channels: Number of input image channels.
            patch_size: Size of each patch (height, width).
            d_model: Dimension of output embeddings.
            flatten: If True, flatten spatial dims to sequence.
        """
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.flatten = flatten
        
        # Use Conv2d for efficient patch extraction and projection
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract and project patches.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Tuple of:
            - Patch embeddings of shape (B, N, d_model) if flatten else (B, d_model, H', W')
            - Grid size (H', W') for positional encoding
        """
        B, C, H, W = x.shape
        
        # Compute grid size
        grid_h = H // self.patch_size[0]
        grid_w = W // self.patch_size[1]
        
        # Project patches
        x = self.proj(x)  # (B, d_model, grid_h, grid_w)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, d_model)
        
        return x, (grid_h, grid_w)


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for variable-length sequences.
    
    Uses interpolation to handle sequences longer than training length.
    """
    
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input."""
        B, N, D = x.shape
        
        if N <= self.max_len:
            return x + self.pos_embed[:, :N, :]
        else:
            # Interpolate for longer sequences
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            return x + pos


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformers.
    
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for handwriting text recognition.
    
    This backbone uses the ViT architecture adapted for HTR:
    - Variable-length patch sequences for variable-width images
    - No class token (sequence output for CTC/attention decoding)
    - Optional feature projection layer
    
    Attributes:
        output_dim: Dimension of output features.
        patch_size: Size of image patches.
    
    Example:
        >>> backbone = ViTBackbone(in_channels=3, d_model=256)
        >>> x = torch.randn(4, 3, 64, 256)
        >>> features = backbone(x)  # (B, Seq, 256)
    """
    
    # Predefined configurations
    CONFIGS = {
        'vit_tiny': {'d_model': 192, 'num_heads': 3, 'num_layers': 12, 'patch_size': (8, 8)},
        'vit_small': {'d_model': 384, 'num_heads': 6, 'num_layers': 12, 'patch_size': (16, 16)},
        'vit_base': {'d_model': 768, 'num_heads': 12, 'num_layers': 12, 'patch_size': (16, 16)},
        'vit_htr_small': {'d_model': 256, 'num_heads': 8, 'num_layers': 6, 'patch_size': (8, 4)},
        'vit_htr_base': {'d_model': 512, 'num_heads': 8, 'num_layers': 8, 'patch_size': (16, 8)},
    }
    
    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = None,
        patch_size: Tuple[int, int] = (16, 16),
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        output_dim: Optional[int] = None,
        pos_encoding: str = 'learnable',
        config_name: Optional[str] = None
    ):
        """
        Initialize ViT backbone.
        
        Args:
            in_channels: Number of input image channels.
            d_model: Transformer model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Feed-forward dimension. Defaults to 4 * d_model.
            patch_size: Size of each patch (height, width).
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for positional encoding.
            output_dim: Output feature dimension. None keeps d_model.
            pos_encoding: Type of positional encoding ('learnable' or 'sinusoidal').
            config_name: Use predefined config ('vit_tiny', 'vit_small', etc.).
        """
        super().__init__()
        
        # Load config if specified
        if config_name is not None and config_name in self.CONFIGS:
            config = self.CONFIGS[config_name]
            d_model = config.get('d_model', d_model)
            num_heads = config.get('num_heads', num_heads)
            num_layers = config.get('num_layers', num_layers)
            patch_size = config.get('patch_size', patch_size)
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.output_dim = output_dim or d_model
        
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model
        )
        
        # Positional encoding
        if pos_encoding == 'learnable':
            self.pos_encoding = LearnablePositionalEmbedding(d_model, max_seq_len)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Optional output projection
        self.output_proj = (
            nn.Linear(d_model, self.output_dim)
            if d_model != self.output_dim
            else nn.Identity()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using standard ViT initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input image of shape (B, C, H, W).
            return_all_layers: If True, return features from all layers.
            
        Returns:
            Feature tensor of shape (B, Seq, output_dim).
            If return_all_layers, returns list of tensors from each layer.
        """
        # Patch embedding
        x, grid_size = self.patch_embed(x)  # (B, N, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        if return_all_layers:
            outputs = []
            for layer in self.transformer.layers:
                x = layer(x)
                outputs.append(x)
            x = self.norm(x)
            outputs[-1] = x
            return [self.output_proj(o) for o in outputs]
        else:
            x = self.transformer(x)
            x = self.norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim


class HybridViTBackbone(nn.Module):
    """
    Hybrid CNN-ViT backbone combining convolutional and attention features.
    
    Uses a small CNN stem to extract initial features, then applies
    ViT for global context modeling. This combines the efficiency of
    CNNs for local feature extraction with the modeling power of
    self-attention for long-range dependencies.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        cnn_channels: int = 64,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = d_model
        
        # CNN stem for initial feature extraction
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels * 2, d_model, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        
        # Pool height dimension
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using hybrid CNN-ViT architecture.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Feature tensor of shape (B, Seq, d_model).
        """
        # CNN feature extraction
        x = self.cnn_stem(x)  # (B, d_model, H', W')
        
        # Pool height and create sequence
        x = self.height_pool(x)  # (B, d_model, 1, W')
        x = x.squeeze(2).transpose(1, 2)  # (B, W', d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x
    
    def get_output_dim(self) -> int:
        return self.output_dim

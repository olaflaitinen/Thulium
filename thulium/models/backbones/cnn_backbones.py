"""
CNN Backbone architectures for feature extraction in HTR models.

This module provides convolutional neural network backbones optimized
for handwriting text recognition. These backbones convert input images
into feature sequences suitable for sequence modeling with LSTMs or
Transformers.

Architecture Design Considerations
----------------------------------
HTR-specific CNNs differ from standard image classification CNNs in that:

1. They must preserve horizontal resolution for sequence modeling
2. They typically use more aggressive vertical pooling
3. They need to handle variable-width inputs efficiently

The output features have shape (B, C, H', W') where W' roughly corresponds
to the number of timesteps for the sequence model. Vertical dimension H'
is typically collapsed via pooling or flattening before sequence modeling.
"""

from typing import Optional, List
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper networks.
    
    Implements the standard residual learning formulation:
    
        y = F(x) + x
    
    where F is the residual mapping (conv -> bn -> relu -> conv -> bn).
    A 1x1 convolution is used for the skip connection if dimensions change.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection projection if dimensions change
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetBackbone(nn.Module):
    """
    ResNet-style backbone optimized for handwriting recognition.
    
    This backbone uses asymmetric pooling to preserve horizontal resolution
    while reducing vertical dimension, which is critical for sequence-based
    HTR where width corresponds to timesteps.
    
    Architecture:
        Stem (7x7 conv) -> Stage1 -> Stage2 -> Stage3 -> Stage4
        
    Each stage contains multiple residual blocks. Downsampling occurs
    primarily in the vertical direction.
    
    Attributes:
        output_channels: Number of output feature channels.
        output_stride: Total downsampling factor in width dimension.
    
    Example:
        >>> backbone = ResNetBackbone(in_channels=3, layers=[2, 2, 2, 2])
        >>> x = torch.randn(4, 3, 64, 256)  # (B, C, H, W)
        >>> features = backbone(x)  # (B, 512, H', W')
    """
    
    # Predefined configurations matching standard ResNet variants
    CONFIGS = {
        'resnet18': {'layers': [2, 2, 2, 2], 'channels': [64, 128, 256, 512]},
        'resnet34': {'layers': [3, 4, 6, 3], 'channels': [64, 128, 256, 512]},
        'resnet_small': {'layers': [1, 1, 2, 2], 'channels': [32, 64, 128, 256]},
        'resnet_tiny': {'layers': [1, 1, 1, 1], 'channels': [32, 64, 128, 256]},
    }
    
    def __init__(
        self,
        in_channels: int = 3,
        layers: Optional[List[int]] = None,
        channels: Optional[List[int]] = None,
        config_name: Optional[str] = None,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0
    ):
        """
        Initialize ResNet backbone.
        
        Args:
            in_channels: Number of input image channels.
            layers: Number of blocks per stage (e.g., [2, 2, 2, 2] for ResNet18).
            channels: Number of channels per stage.
            config_name: Use predefined config ('resnet18', 'resnet34', etc.).
            dropout: Dropout probability within residual blocks.
            stochastic_depth: Drop path probability (not fully implemented).
        """
        super().__init__()
        
        if config_name is not None and config_name in self.CONFIGS:
            config = self.CONFIGS[config_name]
            layers = config['layers']
            channels = config['channels']
        else:
            layers = layers or [2, 2, 2, 2]
            channels = channels or [64, 128, 256, 512]
        
        self.output_channels = channels[-1]
        
        # Stem: Initial conv with pooling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7,
                     stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stages
        self.stage1 = self._make_stage(
            channels[0], channels[0], layers[0], stride=1, dropout=dropout
        )
        self.stage2 = self._make_stage(
            channels[0], channels[1], layers[1], stride=2, dropout=dropout
        )
        self.stage3 = self._make_stage(
            channels[1], channels[2], layers[2], stride=(2, 1), dropout=dropout
        )
        self.stage4 = self._make_stage(
            channels[2], channels[3], layers[3], stride=(2, 1), dropout=dropout
        )
        
        self._init_weights()
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride,
        dropout: float
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        blocks = []
        
        # First block may have stride
        if isinstance(stride, tuple):
            # Asymmetric stride for HTR (more vertical than horizontal)
            blocks.append(self._make_asymmetric_block(
                in_channels, out_channels, stride, dropout
            ))
        else:
            blocks.append(ResidualBlock(
                in_channels, out_channels, stride=stride, dropout=dropout
            ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(
                out_channels, out_channels, stride=1, dropout=dropout
            ))
        
        return nn.Sequential(*blocks)
    
    def _make_asymmetric_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple,
        dropout: float
    ) -> nn.Module:
        """Create block with asymmetric stride (h_stride, w_stride)."""
        h_stride, w_stride = stride
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=(h_stride, w_stride), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Feature tensor of shape (B, output_channels, H', W').
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
    
    def get_output_dim(self) -> int:
        """Return the number of output channels."""
        return self.output_channels


class LightweightCNNBackbone(nn.Module):
    """
    Lightweight CNN for resource-constrained environments.
    
    This backbone uses depthwise separable convolutions and fewer
    parameters, suitable for CPU deployment or mobile applications.
    Designed for knowledge distillation from larger models.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        output_channels: int = 256
    ):
        super().__init__()
        
        self.output_channels = output_channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise separable blocks
        self.blocks = nn.Sequential(
            self._depthwise_block(base_channels, base_channels * 2, stride=2),
            self._depthwise_block(base_channels * 2, base_channels * 4, stride=2),
            self._depthwise_block(base_channels * 4, base_channels * 4, stride=(2, 1)),
            self._depthwise_block(base_channels * 4, output_channels, stride=(2, 1)),
        )
    
    def _depthwise_block(
        self,
        in_channels: int,
        out_channels: int,
        stride
    ) -> nn.Sequential:
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                     stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x
    
    def get_output_dim(self) -> int:
        return self.output_channels

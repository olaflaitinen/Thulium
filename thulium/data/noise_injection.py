"""
Noise injection utilities for robustness testing.

This module provides tools for synthetically degrading images
to test HTR model robustness under various noise conditions.
Supported degradations include:

- Gaussian noise
- Salt-and-pepper noise
- Compression artifacts (JPEG)
- Blur effects
- Random occlusions
- Resolution variations
- Perspective transforms

These utilities are useful for:
- Robustness evaluation
- Data augmentation during training
- Domain adaptation studies
"""

from typing import Tuple, Optional, List, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """
    Configuration for noise injection.
    
    Attributes:
        gaussian_std: Standard deviation for Gaussian noise.
        salt_pepper_prob: Probability of salt-and-pepper noise.
        jpeg_quality: JPEG quality for compression artifacts.
        blur_kernel_size: Kernel size for Gaussian blur.
        occlusion_prob: Probability of adding occlusion.
        occlusion_size: Size of occlusion rectangles.
    """
    gaussian_std: float = 0.01
    salt_pepper_prob: float = 0.01
    jpeg_quality: int = 75
    blur_kernel_size: int = 3
    occlusion_prob: float = 0.1
    occlusion_size: Tuple[int, int] = (10, 50)


def add_gaussian_noise(
    image: np.ndarray,
    std: float = 0.01,
    mean: float = 0.0
) -> np.ndarray:
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W).
        std: Standard deviation of noise (relative to [0, 1] range).
        mean: Mean of noise (typically 0).
        
    Returns:
        Noisy image with same dtype as input.
    """
    original_dtype = image.dtype
    
    # Convert to float [0, 1]
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)
    
    # Add noise
    noise = np.random.normal(mean, std, image_float.shape).astype(np.float32)
    noisy = np.clip(image_float + noise, 0.0, 1.0)
    
    # Convert back
    if original_dtype == np.uint8:
        return (noisy * 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(
    image: np.ndarray,
    prob: float = 0.01,
    salt_value: float = 1.0,
    pepper_value: float = 0.0
) -> np.ndarray:
    """
    Add salt-and-pepper noise to an image.
    
    Args:
        image: Input image as numpy array.
        prob: Probability of each pixel being affected.
        salt_value: Value for salt (white) pixels.
        pepper_value: Value for pepper (black) pixels.
        
    Returns:
        Noisy image.
    """
    original_dtype = image.dtype
    
    # Convert to float [0, 1]
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
        salt_value = 1.0
        pepper_value = 0.0
    else:
        image_float = image.astype(np.float32)
    
    # Create noise mask
    noise_mask = np.random.random(image_float.shape[:2])
    
    noisy = image_float.copy()
    
    # Salt (white)
    if len(image_float.shape) == 3:
        noisy[noise_mask < prob / 2] = salt_value
    else:
        noisy[noise_mask < prob / 2] = salt_value
    
    # Pepper (black)
    if len(image_float.shape) == 3:
        noisy[(noise_mask >= prob / 2) & (noise_mask < prob)] = pepper_value
    else:
        noisy[(noise_mask >= prob / 2) & (noise_mask < prob)] = pepper_value
    
    if original_dtype == np.uint8:
        return (noisy * 255).astype(np.uint8)
    return noisy


def add_blur(
    image: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Add Gaussian blur to an image.
    
    Args:
        image: Input image as numpy array.
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian.
        
    Returns:
        Blurred image.
    """
    try:
        import cv2
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    except ImportError:
        # Fallback: simple box blur using numpy
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = np.convolve(
                    image[:, :, c].flatten(),
                    kernel.flatten(),
                    mode='same'
                ).reshape(image.shape[:2])
            return result
        return np.convolve(image.flatten(), kernel.flatten(), mode='same').reshape(image.shape)


def add_jpeg_artifacts(
    image: np.ndarray,
    quality: int = 50
) -> np.ndarray:
    """
    Add JPEG compression artifacts.
    
    Args:
        image: Input image as numpy array.
        quality: JPEG quality (1-100, lower = more artifacts).
        
    Returns:
        Image with compression artifacts.
    """
    try:
        from PIL import Image
        import io
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        pil_image = Image.fromarray(image_uint8)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        result = np.array(compressed)
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            return result.astype(np.float32) / 255.0
        return result
    except ImportError:
        # Return original if PIL not available
        return image


def add_random_occlusion(
    image: np.ndarray,
    num_occlusions: int = 1,
    size_range: Tuple[int, int] = (10, 50),
    color: Optional[Union[int, Tuple[int, ...]]] = None
) -> np.ndarray:
    """
    Add random rectangular occlusions to an image.
    
    Args:
        image: Input image as numpy array.
        num_occlusions: Number of occlusion rectangles.
        size_range: (min_size, max_size) for rectangle dimensions.
        color: Fill color. None uses random values.
        
    Returns:
        Image with occlusions.
    """
    occluded = image.copy()
    h, w = image.shape[:2]
    
    for _ in range(num_occlusions):
        # Random size
        size_h = np.random.randint(size_range[0], min(size_range[1], h))
        size_w = np.random.randint(size_range[0], min(size_range[1], w))
        
        # Random position
        y = np.random.randint(0, max(1, h - size_h))
        x = np.random.randint(0, max(1, w - size_w))
        
        # Fill color
        if color is None:
            if len(image.shape) == 3:
                fill = np.random.randint(0, 256, size=(size_h, size_w, image.shape[2]))
            else:
                fill = np.random.randint(0, 256, size=(size_h, size_w))
            
            if image.dtype == np.float32 or image.dtype == np.float64:
                fill = fill.astype(np.float32) / 255.0
        else:
            fill = np.full((size_h, size_w) + image.shape[2:], color, dtype=image.dtype)
        
        occluded[y:y+size_h, x:x+size_w] = fill
    
    return occluded


def resize_and_restore(
    image: np.ndarray,
    scale: float = 0.5
) -> np.ndarray:
    """
    Downsample and upsample image to simulate resolution loss.
    
    Args:
        image: Input image as numpy array.
        scale: Scale factor for downsampling (0 < scale < 1).
        
    Returns:
        Image with resolution degradation.
    """
    try:
        from PIL import Image
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        pil_image = Image.fromarray(image_uint8)
        orig_size = pil_image.size
        
        # Downscale
        small_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        small = pil_image.resize(small_size, Image.BILINEAR)
        
        # Upscale back
        restored = small.resize(orig_size, Image.BILINEAR)
        
        result = np.array(restored)
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            return result.astype(np.float32) / 255.0
        return result
    except ImportError:
        return image


def apply_random_noise(
    image: np.ndarray,
    config: NoiseConfig,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply random noise based on configuration.
    
    Applies each noise type with some probability based on config.
    
    Args:
        image: Input image as numpy array.
        config: NoiseConfig specifying noise parameters.
        seed: Random seed for reproducibility.
        
    Returns:
        Noisy image.
    """
    if seed is not None:
        np.random.seed(seed)
    
    result = image.copy()
    
    # Apply Gaussian noise
    if config.gaussian_std > 0:
        result = add_gaussian_noise(result, std=config.gaussian_std)
    
    # Apply salt-and-pepper noise
    if config.salt_pepper_prob > 0:
        result = add_salt_pepper_noise(result, prob=config.salt_pepper_prob)
    
    # Apply blur
    if config.blur_kernel_size > 1:
        result = add_blur(result, kernel_size=config.blur_kernel_size)
    
    # Apply JPEG artifacts
    if config.jpeg_quality < 100:
        result = add_jpeg_artifacts(result, quality=config.jpeg_quality)
    
    # Apply occlusion
    if config.occlusion_prob > 0 and np.random.random() < config.occlusion_prob:
        result = add_random_occlusion(
            result,
            size_range=config.occlusion_size
        )
    
    return result


class RobustnessTester:
    """
    Utility class for systematic robustness testing.
    
    Provides methods to apply controlled degradations and
    measure impact on recognition accuracy.
    """
    
    def __init__(self, pipeline=None):
        """
        Initialize robustness tester.
        
        Args:
            pipeline: HTR pipeline to test. Can be set later.
        """
        self.pipeline = pipeline
        self.results: List[dict] = []
    
    def set_pipeline(self, pipeline) -> None:
        """Set the pipeline to test."""
        self.pipeline = pipeline
    
    def test_noise_level(
        self,
        images: List[np.ndarray],
        references: List[str],
        noise_type: str,
        levels: List[float],
        language: str = "en"
    ) -> List[dict]:
        """
        Test recognition accuracy at different noise levels.
        
        Args:
            images: List of input images.
            references: List of ground truth strings.
            noise_type: Type of noise ('gaussian', 'blur', 'jpeg', etc.).
            levels: List of noise levels to test.
            language: Language code.
            
        Returns:
            List of result dictionaries for each level.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not set")
        
        results = []
        
        for level in levels:
            # Apply noise at this level
            noisy_images = []
            for img in images:
                if noise_type == 'gaussian':
                    noisy = add_gaussian_noise(img, std=level)
                elif noise_type == 'blur':
                    noisy = add_blur(img, kernel_size=int(level))
                elif noise_type == 'jpeg':
                    noisy = add_jpeg_artifacts(img, quality=int(level))
                elif noise_type == 'resolution':
                    noisy = resize_and_restore(img, scale=level)
                else:
                    noisy = img
                noisy_images.append(noisy)
            
            # Evaluate (placeholder - would call actual pipeline)
            # In production: hypotheses = [self.pipeline.process(img, language).full_text for img in noisy_images]
            
            result = {
                'noise_type': noise_type,
                'level': level,
                'num_samples': len(images),
                # 'cer': computed_cer,
                # 'wer': computed_wer,
            }
            results.append(result)
        
        return results

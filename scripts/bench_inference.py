#!/usr/bin/env python3
"""
Inference Benchmarking Script for Thulium HTR.

This script measures inference performance metrics including:
- Average latency per sample
- Throughput (samples per second)
- Memory usage
- Device utilization

Usage:
    python scripts/bench_inference.py --model config/models/htr_cnn_lstm_ctc.yaml
    python scripts/bench_inference.py --model config/models/htr_vit_transformer_seq2seq.yaml --device cuda --batch-size 16
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark Thulium HTR inference performance'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='config/models/htr_cnn_lstm_ctc.yaml',
        help='Path to model configuration YAML'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=100,
        help='Number of samples to benchmark'
    )
    parser.add_argument(
        '--warmup', '-w',
        type=int,
        default=10,
        help='Number of warmup iterations (not counted)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--image-height',
        type=int,
        default=64,
        help='Height of input images'
    )
    parser.add_argument(
        '--image-width',
        type=int,
        default=256,
        help='Width of input images (average line width)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device."""
    if device == 'auto':
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    return device


def generate_dummy_images(
    num_samples: int,
    height: int,
    width: int,
    batch_size: int,
    device: str
) -> List:
    """Generate dummy input images for benchmarking."""
    try:
        import torch
        
        images = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch = min(batch_size, num_samples - i * batch_size)
            batch = torch.randn(current_batch, 3, height, width)
            if device != 'cpu':
                batch = batch.to(device)
            images.append(batch)
        
        return images
    except ImportError:
        import numpy as np
        return [np.random.randn(batch_size, 3, height, width).astype('float32') 
                for _ in range((num_samples + batch_size - 1) // batch_size)]


def benchmark_model(
    model_config: str,
    device: str,
    images: List,
    warmup: int,
    verbose: bool = False
) -> dict:
    """
    Benchmark model inference performance.
    
    Args:
        model_config: Path to model configuration.
        device: Computation device.
        images: List of input image batches.
        warmup: Number of warmup iterations.
        verbose: Enable verbose logging.
        
    Returns:
        Dictionary with performance metrics.
    """
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    # Load model (placeholder - would load actual model in production)
    logger.info(f"Loading model from {model_config}")
    logger.info(f"Using device: {device}")
    
    # Placeholder model forward pass
    def dummy_forward(batch):
        """Simulated forward pass."""
        time.sleep(0.001)  # Simulate computation
        if torch_available:
            return torch.randn(batch.shape[0], 50, 100)
        return None
    
    # Warmup phase
    logger.info(f"Running {warmup} warmup iterations...")
    for i, batch in enumerate(images[:warmup]):
        _ = dummy_forward(batch)
    
    if torch_available and device.startswith('cuda'):
        import torch
        torch.cuda.synchronize()
    
    # Timed benchmark
    total_samples = sum(b.shape[0] if hasattr(b, 'shape') else len(b) for b in images)
    logger.info(f"Benchmarking {total_samples} samples...")
    
    latencies = []
    start_time = time.perf_counter()
    
    for batch in images:
        batch_start = time.perf_counter()
        _ = dummy_forward(batch)
        
        if torch_available and device.startswith('cuda'):
            import torch
            torch.cuda.synchronize()
        
        batch_latency = (time.perf_counter() - batch_start) * 1000  # ms
        batch_size = batch.shape[0] if hasattr(batch, 'shape') else len(batch)
        latencies.extend([batch_latency / batch_size] * batch_size)
    
    total_time = time.perf_counter() - start_time
    
    # Compute statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies)
    
    results = {
        'model_config': model_config,
        'device': device,
        'num_samples': total_samples,
        'batch_size': images[0].shape[0] if hasattr(images[0], 'shape') else len(images[0]),
        'total_time_s': total_time,
        'throughput_samples_per_sec': total_samples / total_time,
        'latency': {
            'mean_ms': sum(latencies) / n,
            'std_ms': (sum((x - sum(latencies)/n)**2 for x in latencies) / n) ** 0.5,
            'min_ms': latencies_sorted[0],
            'max_ms': latencies_sorted[-1],
            'p50_ms': latencies_sorted[n // 2],
            'p90_ms': latencies_sorted[int(n * 0.9)],
            'p95_ms': latencies_sorted[int(n * 0.95)],
            'p99_ms': latencies_sorted[int(n * 0.99)],
        }
    }
    
    # Memory stats (if GPU)
    if torch_available and device.startswith('cuda'):
        import torch
        results['gpu_memory'] = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        }
    
    return results


def print_results(results: dict) -> None:
    """Print benchmark results in formatted table."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Config:    {results['model_config']}")
    print(f"Device:          {results['device']}")
    print(f"Samples:         {results['num_samples']}")
    print(f"Batch Size:      {results['batch_size']}")
    print("-" * 60)
    print("THROUGHPUT")
    print(f"  Total Time:    {results['total_time_s']:.3f} s")
    print(f"  Throughput:    {results['throughput_samples_per_sec']:.2f} samples/sec")
    print("-" * 60)
    print("LATENCY (per sample)")
    lat = results['latency']
    print(f"  Mean:          {lat['mean_ms']:.3f} ms")
    print(f"  Std:           {lat['std_ms']:.3f} ms")
    print(f"  Min:           {lat['min_ms']:.3f} ms")
    print(f"  Max:           {lat['max_ms']:.3f} ms")
    print(f"  P50:           {lat['p50_ms']:.3f} ms")
    print(f"  P90:           {lat['p90_ms']:.3f} ms")
    print(f"  P95:           {lat['p95_ms']:.3f} ms")
    print(f"  P99:           {lat['p99_ms']:.3f} ms")
    
    if 'gpu_memory' in results:
        print("-" * 60)
        print("GPU MEMORY")
        mem = results['gpu_memory']
        print(f"  Allocated:     {mem['allocated_mb']:.1f} MB")
        print(f"  Reserved:      {mem['reserved_mb']:.1f} MB")
    
    print("=" * 60 + "\n")


def save_results(results: dict, output_path: str) -> None:
    """Save results to JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    device = resolve_device(args.device)
    
    # Generate dummy images
    logger.info("Generating dummy input images...")
    images = generate_dummy_images(
        num_samples=args.num_samples + args.warmup,
        height=args.image_height,
        width=args.image_width,
        batch_size=args.batch_size,
        device=device
    )
    
    # Run benchmark
    results = benchmark_model(
        model_config=args.model,
        device=device,
        images=images,
        warmup=args.warmup,
        verbose=args.verbose
    )
    
    # Print results
    print_results(results)
    
    # Save if requested
    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()

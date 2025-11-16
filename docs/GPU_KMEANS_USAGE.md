# GPU-Accelerated K-means Usage Guide

## Overview

The GPU-accelerated K-means implementation provides significant speedup for clustering operations used in the quantization process. The implementation automatically detects GPU availability and falls back to CPU if needed.

## Features

- **GPU Acceleration**: Uses PyTorch for GPU-accelerated K-means clustering (10-50x faster than CPU)
- **Automatic Fallback**: Automatically uses CPU if GPU is not available
- **Compatible API**: Drop-in replacement for the existing K-means implementation
- **K-means++ Initialization**: Better clustering quality with smart centroid initialization
- **Configurable Parameters**: Control iterations, tolerance, and number of initializations

## Installation

### Install PyTorch with GPU Support

For CUDA 11.8:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install Other Dependencies
```bash
pip3 install -r requirements.txt
```

## Usage

### Basic Usage in cluster_process.py

The `cluster_process.py` script now supports GPU acceleration by default:

```bash
# Use GPU acceleration (default)
python cluster_process.py

# Or explicitly enable GPU
python cluster_process.py --use_gpu

# Disable GPU acceleration (use CPU only)
python cluster_process.py --no_gpu
```

### Using KMeansGPU in Custom Code

```python
from utils.kmeans_gpu import KMeansGPU
import numpy as np

# Generate sample data
X = np.random.randn(1000, 128)

# Create GPU K-means instance
kmeans = KMeansGPU(
    n_clusters=256,        # Number of clusters
    max_iter=1000,         # Maximum iterations
    tol=1e-5,              # Convergence tolerance
    n_init=10,             # Number of initializations
    device='cuda',         # 'cuda' or 'cpu', or None for auto-detect
    verbose=True           # Print progress
)

# Fit the model
kmeans.fit(X)

# Access results
print(f"Cluster centers: {kmeans.cluster_centers_.shape}")
print(f"Labels: {kmeans.labels_.shape}")
print(f"Inertia: {kmeans.inertia_}")

# Predict new data
new_X = np.random.randn(100, 128)
labels = kmeans.predict(new_X)
```

### API Compatibility

The `KMeansGPU` class provides the same interface as the original `KMeans` class:

```python
# Both work the same way
from utils.kmeans import KMeans
from utils.kmeans_gpu import KMeansGPU

# Original CPU K-means
kmeans_cpu = KMeans(n_clusters=16)
kmeans_cpu.fit(X)
centers_cpu = kmeans_cpu.centers  # Access centers

# GPU K-means (compatible API)
kmeans_gpu = KMeansGPU(n_clusters=16)
kmeans_gpu.fit(X)
centers_gpu = kmeans_gpu.centers  # Access centers (same as cluster_centers_)
```

## Performance Comparison

### Expected Speedups

| Data Size | Features | Clusters | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|----------|----------|---------|
| 1,000     | 128      | 16       | 0.05s    | 0.01s    | 5x      |
| 10,000    | 128      | 256      | 5.0s     | 0.2s     | 25x     |
| 100,000   | 256      | 256      | 120s     | 3s       | 40x     |

*Note: Speedups vary based on GPU model, data size, and configuration*

### Testing Your Setup

Run this test to verify GPU acceleration:

```python
import time
import numpy as np
from utils.kmeans import KMeans
from utils.kmeans_gpu import KMeansGPU

# Generate test data
X = np.random.randn(10000, 128)

# Test CPU K-means
start = time.time()
kmeans_cpu = KMeans(n_clusters=256)
kmeans_cpu.fit(X)
cpu_time = time.time() - start

# Test GPU K-means
start = time.time()
kmeans_gpu = KMeansGPU(n_clusters=256, device='cuda')
kmeans_gpu.fit(X)
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.3f}s")
print(f"GPU time: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

## GPU Memory Considerations

For large datasets, GPU memory usage is important:

- **Memory Usage**: Approximately `n_samples × n_features × 4 bytes` (for float32)
- **Example**: 100,000 samples × 256 features = ~100 MB

If you encounter out-of-memory errors:

1. **Reduce batch size**: Process channels sequentially instead of in parallel
2. **Use CPU for large data**: `KMeansGPU(device='cpu', ...)`
3. **Reduce n_init**: Use fewer random initializations (e.g., `n_init=1`)

## Troubleshooting

### GPU Not Detected

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
```

If CUDA is not available:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA version matches your CUDA installation
- Reinstall PyTorch with correct CUDA version

### Import Errors

If you get `ModuleNotFoundError: No module named 'torch'`:
```bash
pip3 install torch torchvision
```

### Performance Issues

If GPU K-means is slow:
- Check GPU utilization: `nvidia-smi` (should show high GPU usage)
- Ensure data is on GPU: Check `kmeans.device` is `cuda`
- Increase data size (GPUs are faster for larger datasets)
- Close other GPU applications

## Configuration Recommendations

### For cluster_process.py

Recommended settings for PQ compression:

```python
# For smaller subspace dimensions (e.g., < 128)
kmeans = KMeansGPU(
    n_clusters=K,
    max_iter=1000,
    tol=1e-5,
    n_init=5,       # Fewer inits, GPU is fast enough
    device='cuda',
    verbose=False
)

# For larger subspace dimensions (e.g., >= 128)
kmeans = KMeansGPU(
    n_clusters=K,
    max_iter=500,   # Fewer iterations, usually converges quickly
    tol=1e-4,       # Slightly relaxed tolerance
    n_init=3,       # Even fewer inits
    device='cuda',
    verbose=False
)
```

## References

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [K-means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [Product Quantization for Nearest Neighbor Search](https://hal.inria.fr/inria-00514462v2/document)

## Support

If you encounter issues:
1. Check that PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
2. Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try CPU mode first: `python cluster_process.py --no_gpu`
4. Check the error message and refer to this documentation

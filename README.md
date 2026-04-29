# CUDA Softmax & Flash Attention

A ground-up CUDA implementation of softmax and scaled dot-product attention, optimized step by step and benchmarked at each stage. Built to understand how modern GPU kernels work and why Flash Attention exists.

## What's in here

| File | What it does |
|------|-------------|
| `softmax_naive.cu` | One thread per row — the obvious approach |
| `softmax_optimized.cu` | One block per row, shared memory + warp shuffles |
| `attention_baseline.cu` | Full attention: QK^T → softmax → V, step-by-step profiled |
| `attention_flash.cu` | Flash Attention-style tiled kernel — coming soon |

## Benchmark Results

### Softmax (1024 rows)

| Version | SEQ_LEN | Bandwidth | Speedup |
|---------|---------|-----------|---------|
| Naive | 512 | 4.80 GB/s | 1× |
| Optimized | 512 | 109.85 GB/s | **22.9×** |
| Optimized | 1024 | 125.95 GB/s | **26.2×** |

GPU: NVIDIA T4 (peak ~300 GB/s)

### Attention Baseline (d_k=64, d_v=64)

| SEQ_LEN | Total Time | S Matrix | QK^T | Softmax | PV |
|---------|-----------|----------|------|---------|-----|
| 64 | 0.025 ms | 0.02 MB | 0.008 ms | 0.005 ms | 0.008 ms |
| 128 | 0.031 ms | 0.06 MB | 0.012 ms | 0.005 ms | 0.011 ms |
| 256 | 0.060 ms | 0.25 MB | 0.024 ms | 0.011 ms | 0.022 ms |
| 512 | 0.175 ms | 1.00 MB | 0.078 ms | 0.020 ms | 0.070 ms |

At SEQ_LEN=4096 the S matrix alone would be **64 MB per head**. That's the problem Flash Attention solves.

## The optimization story

### Softmax: 4.8 → 110 GB/s

The naive kernel assigns one thread per row. With 1024 rows and 256 threads/block, only 4 blocks run — leaving 36 of the T4's 40 SMs completely idle. Every memory access hits global DRAM.

The optimized kernel flips the model: one block per row, 512 threads collaborate using shared memory and warp-level shuffle reductions. The row loads into on-chip SRAM once. All reductions happen there. Result: 22.9× faster.

Key techniques:
- `__shared__` memory for on-chip storage (~5 cycle latency vs ~600 for DRAM)
- `__syncthreads()` barriers to coordinate threads safely
- `__shfl_down_sync()` for warp-level parallel reduction in O(log N) steps
- Numerically stable softmax: subtract row max before exp to prevent overflow

### Attention: why it gets expensive

Attention is O(N²) in both compute and memory. The score matrix S = QK^T is N×N — at SEQ_LEN=512 that's 1MB. At SEQ_LEN=4096 it's 64MB per head. Writing and reading this matrix from DRAM on every forward pass is the bottleneck.

Per-step profiling at SEQ_LEN=512:
- QK^T GEMM: 45% of runtime
- PV GEMM: 40% of runtime  
- Softmax: 12% of runtime
- Scale: 3% of runtime

The GEMMs dominate because they're both reading and writing the full N×N matrix through global memory.

### Flash Attention: never materialize S

Flash Attention (Dao et al., 2022) tiles the computation so Q, K, V are processed in blocks that fit in shared memory. S is never written to DRAM. This reduces memory complexity from O(N²) to O(N) and cuts memory bandwidth requirements dramatically — especially important for long sequences.

## How to run

**On Google Colab (recommended — free T4 GPU):**

```python
# Cell 1
%%writefile softmax_naive.cu
# paste file contents

# Cell 2
!nvcc -O2 -o softmax_naive softmax_naive.cu && ./softmax_naive
```

**Locally (requires NVIDIA GPU + CUDA Toolkit):**

```bash
nvcc -O2 -o softmax_naive softmax_naive.cu && ./softmax_naive
nvcc -O2 -o softmax_optimized softmax_optimized.cu && ./softmax_optimized
nvcc -O2 -o attention_baseline attention_baseline.cu && ./attention_baseline
```

## References

- [Flash Attention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Online softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)

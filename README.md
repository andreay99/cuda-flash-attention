# CUDA Softmax & Flash Attention

A ground-up CUDA implementation of softmax and scaled dot-product attention, optimized step by step and benchmarked at each stage. Culminates in a raw CUDA implementation of the Flash Attention backward pass — including gradient recomputation from saved statistics, without ever materializing the full N×N score matrix.

## What's in here

| File | What it does |
|------|-------------|
| `softmax_naive.cu` | One thread per row — the obvious approach |
| `softmax_optimized.cu` | One block per row, shared memory + warp shuffles |
| `attention_baseline.cu` | Full attention: QK^T → softmax → V, step-by-step profiled |
| `attention_flash.cu` | Flash Attention-style tiled kernel with online softmax |
| `flash_attention_backward.cu` | Flash Attention forward + full backward pass in raw CUDA |
| `verify_gradients.py` | Verifies CUDA gradients against PyTorch autograd |

## Benchmark Results

### Softmax (1024 rows, NVIDIA T4)

| Version | SEQ_LEN | Bandwidth | Speedup |
|---------|---------|-----------|---------|
| Naive (1 thread/row) | 512 | 4.80 GB/s | 1× |
| Optimized (shared mem + warp shuffle) | 512 | 109.85 GB/s | **22.9×** |
| Optimized | 1024 | 125.95 GB/s | **26.2×** |

Peak bandwidth on T4: ~300 GB/s

### Attention Baseline — Per-Step Profiling (d_k=64, d_v=64)

| SEQ_LEN | Total | S matrix | QK^T | Softmax | PV |
|---------|-------|----------|------|---------|-----|
| 64 | 0.025 ms | 0.02 MB | 0.008 ms | 0.005 ms | 0.008 ms |
| 128 | 0.031 ms | 0.06 MB | 0.012 ms | 0.005 ms | 0.011 ms |
| 256 | 0.060 ms | 0.25 MB | 0.024 ms | 0.011 ms | 0.022 ms |
| 512 | 0.175 ms | 1.00 MB | 0.078 ms | 0.020 ms | 0.070 ms |

At SEQ_LEN=512, the two GEMMs (QK^T + PV) account for 85% of runtime.
At SEQ_LEN=4096, the S matrix alone would be **64 MB per head**.

### Flash Attention vs Baseline — Scaling Results (d_k=64, NVIDIA T4)

| SEQ_LEN | Baseline Time | Flash Time | Baseline HBM | Flash HBM | HBM Reduction |
|---------|--------------|------------|--------------|-----------|---------------|
| 1024 | 0.577 ms | 2.317 ms | 17.00 MB | 0.75 MB | **22.67×** |
| 2048 | 1.692 ms | 6.578 ms | 66.00 MB | 1.50 MB | **44.00×** |
| 4096 | 6.667 ms | 21.745 ms | 260.00 MB | 3.00 MB | **86.67×** |

Flash Attention's HBM traffic reduction scales as O(N) while the baseline scales as O(N²).
At SEQ_LEN=4096, the baseline requires 260MB of HBM traffic per forward pass vs 3MB for Flash — an **86.67× reduction**.

The Flash kernel is slower in wall-clock time on a T4 at these sequence lengths because the implementation is not yet tiled in the inner loops — each thread does O(N·d) sequential work with no shared memory reuse across iterations. A production implementation (FlashAttention-2) adds tiling to the backward kernels to recover the wall-clock speedup. The memory traffic reduction is real and independent of this optimization: Flash genuinely never writes the S or P matrices to DRAM at any sequence length.

At SEQ_LEN=4096, the baseline S matrix alone is **64MB per head**. For a 32-head model that is 2GB just for attention scores — before weights, activations, or gradients. Flash Attention reduces this to O(N·d), which is what makes long-context models practical.

### Flash Attention Backward — Gradient Verification (SEQ_LEN=256, d=64)

All gradients verified against CPU reference (float32):

| Gradient | Max error vs CPU | Status |
|----------|-----------------|--------|
| Forward O | 9.69e-08 | PASSED |
| dV | 3.58e-07 | PASSED |
| dQ | 4.47e-08 | PASSED |
| dK | 1.19e-07 | PASSED |

Errors are at the level of float32 rounding noise (~1e-7). The backward pass
never stores the N×N attention matrix — P is recomputed from saved m and l.

## Why This Matters

Modern large language models use attention across hundreds or thousands of tokens. At SEQ_LEN=4096, the standard attention score matrix is **64MB per head**. A model with 32 attention heads running on 8 GPUs needs **16GB just for the attention matrices** — before weights, activations, or gradients. Flash Attention reduces attention memory from O(N²) to O(N·d), which is what makes long-context models like GPT-4 (128k tokens) and Claude (200k tokens) practical at all. The backward pass is the harder half: most open-source student implementations stop at the forward pass because storing and differentiating through the recomputation requires careful math. This project implements the full training loop — forward and backward — entirely in raw CUDA, with no PyTorch autograd, no CuPy, no high-level wrappers.

## The Optimization Story

### Softmax: 4.8 → 110 GB/s (22.9×)

The naive kernel assigns one thread per row. With 1024 rows and 256 threads/block,
only 4 blocks run — leaving 36 of the T4's 40 SMs completely idle.
Every memory access hits global DRAM at ~600 cycle latency.

The optimized kernel flips the model: one block per row, 512 threads collaborate
using shared memory and warp-level shuffle reductions. The row loads into on-chip
SRAM once. All reductions happen there.

Key techniques:
- `__shared__` memory: ~5 cycle latency vs ~600 for DRAM
- `__syncthreads()`: barrier to coordinate shared memory safely
- `__shfl_down_sync()`: warp-level parallel reduction in O(log N) steps
- Numerically stable softmax: subtract row max before exp to prevent overflow

### Attention Baseline: O(N²) memory growth

Per-step profiling at SEQ_LEN=512 shows the GEMMs dominate (85% of runtime)
because both QK^T and PV read/write the full N×N score matrix through DRAM.
Doubling SEQ_LEN roughly triples runtime — the O(N²) cost showing up in practice.

### Flash Attention: online softmax + tiled computation

Flash Attention (Dao et al., 2022) processes Q, K, V in tiles that fit in
shared memory, using the online softmax algorithm to accumulate correct output
without ever materializing the full S matrix.

This implementation uses a 2D thread block (Br=16, Bc=32, 512 threads/block)
where thread (ty, tx) computes score S[ty][tx] = dot(Q[ty], K[tx]) in parallel
with all other threads, then collaborates on the online softmax update.

The key identity that makes online softmax work: given running max m and
normalizer l, when new scores arrive with tile max m_new:
```
alpha   = exp(m - m_new)       // correction for old accumulator
l_new   = alpha * l + sum(exp(scores - m_new))
O_new   = alpha * O + exp(scores - m_new) * V_tile
m       = m_new
```
At the end, O / l gives the correct softmax-weighted output — with S never
written to memory at any sequence length.

### Flash Attention Backward: recomputation instead of storage

Standard attention backward requires storing the full N×N softmax matrix P
from the forward pass, then reading it back to compute gradients — O(N²) memory.

The Flash Attention backward pass avoids this entirely. The forward pass saves
only two vectors: m[N] (per-row softmax max) and l[N] (per-row normalizer).
During backward, each P tile is recomputed on-the-fly from Q, K, m, and l:

```
P[i][j] = exp(S[i][j] - m[i]) / l[i]
```

Gradients follow from the softmax backward identity. Given D[i] = dot(dO[i], O[i]):

```
dS[i][j] = P[i][j] * (dP[i][j] - D[i])    where dP[i][j] = dot(dO[i], V[j])
dV[j]    = sum_i P[i][j] * dO[i]
dQ[i]    = sum_j dS[i][j] * K[j] / sqrt(d)
dK[j]    = sum_i dS[i][j] * Q[i] / sqrt(d)
```

The N×N score matrix is never written to memory at any point in forward or backward.
Memory stays O(N·d) throughout — the key property that makes Flash Attention
practical at large sequence lengths.

## How to Run

**Google Colab (recommended — free T4 GPU):**

```python
%%writefile flash_attention_backward.cu
# paste file contents here

!nvcc -O2 -o flash_backward_bin flash_attention_backward.cu && ./flash_backward_bin
```

**Locally (requires NVIDIA GPU + CUDA Toolkit):**

```bash
nvcc -O2 -o softmax_naive      softmax_naive.cu      && ./softmax_naive
nvcc -O2 -o softmax_optimized  softmax_optimized.cu  && ./softmax_optimized
nvcc -O2 -o attention_baseline attention_baseline.cu  && ./attention_baseline
nvcc -O2 -o attention_flash    attention_flash.cu     && ./attention_flash
nvcc -O2 -o flash_backward_bin flash_attention_backward.cu && ./flash_backward_bin
```

## References

- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

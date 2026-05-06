# Implementing Flash Attention's Backward Pass in Raw CUDA: What I Learned

Most implementations of Flash Attention stop at the forward pass. The backward pass is harder to find, harder to understand, and much harder to get right. This is what I learned building it from scratch in raw CUDA C++.

---

## Why I built this

I wanted to understand why Flash Attention exists — not just use it, but actually know what's happening at the hardware level. So I built the whole thing from the ground up: naive softmax, optimized softmax, standard attention baseline, Flash Attention forward, and finally the backward pass.

The backward pass was the part I couldn't find good raw CUDA implementations of. Most student repos use PyTorch autograd or CuPy. The original paper's reference implementation is in Triton. I wanted raw CUDA C++ — the thing closest to the metal.

---

## The memory problem Flash Attention solves

Standard scaled dot-product attention computes:

```
S = QK^T / sqrt(d)     [N × N]
P = softmax(S)          [N × N]
O = PV                  [N × d]
```

At SEQ_LEN=4096 with d=64, the S and P matrices are each 64MB per head. A 32-head model needs 4GB just for attention matrices — before weights, optimizer state, or gradients. That's why long-context models were impractical before Flash Attention.

Flash Attention never writes S or P to DRAM. It processes Q, K, V in tiles that fit in shared memory and uses the online softmax algorithm to accumulate the output correctly without materializing the full matrix.

The key identity: given a running max m and normalizer l, when a new tile of scores arrives with tile max m_new:

```
alpha = exp(m - m_new)          // correction factor for old accumulator
l_new = alpha * l + sum(exp(scores - m_new))
O_new = alpha * O + exp(scores - m_new) * V_tile
```

At the end, `O / l` gives the correct softmax-weighted output. Memory stays O(N·d) regardless of sequence length.

---

## The forward pass: what to save for backward

The standard Flash Attention forward pass outputs only O. But to compute gradients during backward, we need to be able to reconstruct the attention weights P for any (query, key) pair.

The trick: instead of storing P — which would be O(N²) and defeat the purpose — we save two small vectors per row:

- `m[i]`: the running max of scores for query row i (used for numerical stability)
- `l[i]`: the running sum of exp weights for query row i (the softmax normalizer)

With these two vectors (both O(N)), we can recompute any P[i][j] on demand during backward:

```
P[i][j] = exp(S[i][j] - m[i]) / l[i]
```

This recomputation is the core idea of the Flash Attention backward pass.

---

## Deriving the gradients

Given the loss gradient dO, we need dQ, dK, and dV.

**dV** is the easiest. Since O = P·V, by the chain rule:

```
dV = P^T · dO
```

For each key/value row j, we sum contributions from all query rows:

```
dV[j] = sum_i P[i][j] * dO[i]
```

**dQ and dK** require the softmax backward. The gradient through softmax is:

```
dS[i][j] = P[i][j] * (dP[i][j] - D[i])
```

where D[i] is a scalar correction per query row:

```
D[i] = dot(dO[i], O[i])
```

and dP[i][j] (gradient of loss w.r.t. pre-softmax attention weights) is:

```
dP[i][j] = dot(dO[i], V[j])
```

Then:

```
dQ[i] = sum_j dS[i][j] * K[j] / sqrt(d)
dK[j] = sum_i dS[i][j] * Q[i] / sqrt(d)
```

All of this happens without ever storing the full N×N dS matrix.

---

## The CUDA implementation

Each gradient has its own kernel, all launched as `<<<N, D_MODEL>>>` — one block per output row, one thread per output dimension.

**compute_D**: One block per query row. Threads split the d dimension, do a parallel reduction in shared memory to compute `D[i] = dot(dO[i], O[i])`.

**flash_attn_backward_dV**: Block i handles KV row i. Each thread computes one output dimension of dV[i] by looping over all N query rows, recomputing P[q][i] from saved m and l, and accumulating `p * dO[q][tx]`.

**flash_attn_backward_dQ**: Block i handles query row i. Loop over all KV rows, recompute P[i][kv], compute dP and dS, accumulate into dQ[i].

**flash_attn_backward_dK**: Block j handles KV row j. Loop over all query rows — the transpose structure makes this the most asymmetric kernel. For each query row, recompute P[q][j], compute dS[q][j], accumulate `dS * Q[q][tx]` into dK[j].

### The bug I hit

My first version used `dim3(D_MODEL, Bc)` thread blocks — 64×32 = 2048 threads per block. CUDA's hard limit is 1024. The kernel silently launched zero blocks. Every output was zero. The fix was switching to `<<<N, D_MODEL>>>` (one block per row, 64 threads per block) and looping inside the kernel instead of parallelizing across both dimensions.

---

## Results

All four gradients pass against CPU reference:

| Gradient | Max error | Status |
|----------|-----------|--------|
| Forward O | 9.69e-08 | PASSED |
| dV | 3.58e-07 | PASSED |
| dQ | 4.47e-08 | PASSED |
| dK | 1.19e-07 | PASSED |

Errors are float32 rounding noise — the math is exact.

---

## What I'd do differently

The backward kernels are correct but not optimally tiled. The inner loops over N are fully sequential — each thread does O(N·d) work with no shared memory reuse across the loop. A production-quality implementation (like the one in FlashAttention-2) uses tiled backward kernels where K and V tiles are loaded into shared memory and reused across multiple query rows.

The current implementation is optimized for correctness and clarity. The next step would be adding tiling to the backward kernels to reduce global memory traffic, then benchmarking against PyTorch's `scaled_dot_product_attention` backward at SEQ_LEN=4096.

---

## Code

Full implementation: [github.com/andreay99/cuda-flash-attention](https://github.com/andreay99/cuda-flash-attention)

Files:
- `flash_attention_backward.cu` — forward + backward kernels
- `verify_gradients.py` — PyTorch autograd verification script
- `attention_flash.cu` — forward-only Flash Attention (earlier iteration)
- `attention_baseline.cu` — standard 4-kernel attention pipeline for comparison

---

## References

- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

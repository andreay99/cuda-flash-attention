// =============================================================================
// attention_baseline.cu — Week 3: Scaled Dot-Product Attention
// =============================================================================
//
// What this file builds:
//   Full attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//
// Step by step:
//   1. Naive GEMM kernel  — compute S = Q * K^T  (seq_len x seq_len matrix)
//   2. Scale S by 1/sqrt(d_k)
//   3. Apply your softmax from Week 2 (row-wise, one block per row)
//   4. Output = P * V   (another GEMM)
//   5. Benchmark + profile: where does time go?
//
// Dimensions (single head, single batch item for clarity):
//   Q: [seq_len x d_k]
//   K: [seq_len x d_k]
//   V: [seq_len x d_v]   (usually d_v == d_k)
//   S: [seq_len x seq_len]  ← this is the expensive matrix
//   P: [seq_len x seq_len]  ← softmax(S)
//   O: [seq_len x d_v]     ← final output
//
// The BIG insight you'll observe:
//   S and P are seq_len x seq_len. For seq_len=1024, that's 4MB per head.
//   For seq_len=4096, that's 64MB per head — this is why Flash Attention exists.
//
// How to compile:
//   nvcc -O2 -o attention_baseline attention_baseline.cu
//
// How to run:
//   ./attention_baseline
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define SEQ_LEN   64     // sequence length (N) — number of tokens
#define D_K       64     // key/query dimension per head
#define D_V       64     // value dimension per head
                         // memory for S matrix = SEQ_LEN^2 * 4 bytes
                         // SEQ_LEN=64:  16KB  (tiny, fits in L1)
                         // SEQ_LEN=512: 1MB   (fits in L2)
                         // SEQ_LEN=1024: 4MB  (spills to DRAM)
                         // SEQ_LEN=4096: 64MB (Week 4 problem — Flash Attention)

#define BLOCK_SIZE  16   // tile size for GEMM (16x16 thread blocks)
#define NUM_RUNS    100
#define EPSILON     1e-4f  // relaxed slightly — GEMM accumulation has more float error

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ===========================================================================
// BACKGROUND: Matrix Multiplication on the GPU
// ===========================================================================
//
// C = A * B  where A is [M x K], B is [K x N], C is [M x N]
//
// Naive approach: one thread computes one element of C.
//   C[row][col] = sum over k of A[row][k] * B[k][col]
//
// Problem: B is accessed column-by-column, which is terrible for coalescing.
//   (Adjacent threads read adjacent rows of B, but B is stored row-major,
//    so they're reading elements that are N floats apart — a stride access.)
//
// Tiled approach (what we use here):
//   Divide A and B into BLOCK_SIZE x BLOCK_SIZE tiles.
//   Load each tile into shared memory, compute partial dot products,
//   accumulate, repeat for the next tile.
//   This way each global memory load is reused BLOCK_SIZE times.
//
// For attention specifically:
//   S = Q * K^T   → M=SEQ_LEN, K=D_K, N=SEQ_LEN
//   O = P * V     → M=SEQ_LEN, K=SEQ_LEN, N=D_V

// ===========================================================================
// Kernel 1a: Tiled GEMM — C = A * B   (both row-major, standard)
// ===========================================================================
//
// A is [M x K], B is [K x N], C is [M x N]
// Used for: O = P * V
//
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float acc = 0.0f;

    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                           ? A[row * K + a_col] : 0.0f;

        int b_row = t * BLOCK_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
                                           ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ===========================================================================
// Kernel 1b: Tiled GEMM — C = A * B^T  (B is transposed)
// ===========================================================================
//
// A is [M x K], B is [N x K] (we read it transposed), C is [M x N]
// Used for: S = Q * K^T
//   Q is [seq_len x d_k], K is [seq_len x d_k], S is [seq_len x seq_len]
//   S[i][j] = dot(Q[i], K[j])  — both rows of length d_k
//
// The key difference from gemm_kernel:
//   tileB loads B[col_in_B][k] instead of B[k][col_in_B]
//   i.e., we swap the row/col indices when reading B to simulate B^T
//
__global__ void gemm_nt_kernel(const float* A, const float* B, float* C,
                                int M, int K, int N) {
    // A: [M x K], B: [N x K] (accessed transposed), C: [M x N]
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;  // row in C, row in A
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // col in C, row in B

    float acc = 0.0f;
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A: A[row][t*BLOCK_SIZE + threadIdx.x]
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
                                           ? A[row * K + a_col] : 0.0f;

        // Load tile of B TRANSPOSED:
        // We want B^T[k][col] = B[col][k]
        // k index = t*BLOCK_SIZE + threadIdx.y  (the shared K dimension)
        // col index = blockIdx.x*BLOCK_SIZE + threadIdx.x  (which col of C)
        // But each thread loads tileB[threadIdx.y][threadIdx.x], which
        // corresponds to B[col_base + threadIdx.x][t*BLOCK_SIZE + threadIdx.y]
        int b_col = t * BLOCK_SIZE + threadIdx.y;  // position along K
        int b_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // which row of B
        tileB[threadIdx.y][threadIdx.x] = (b_row < N && b_col < K)
                                           ? B[b_row * K + b_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ===========================================================================
// Kernel 2: Scale matrix by scalar in-place
// ===========================================================================
//
// S = S / sqrt(d_k)
// One thread per element, simple and fast.
//
__global__ void scale_kernel(float* S, float scale, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        S[idx] *= scale;
    }
}

// ===========================================================================
// Kernel 3: Row-wise Softmax (your Week 2 optimized version)
// ===========================================================================
//
// Applied to S after scaling: P = softmax(S)
// One block per row, shared memory + warp shuffles.
//
#define SOFTMAX_BLOCK 64   // must be >= SEQ_LEN for this simple version

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void softmax_kernel(const float* input, float* output,
                                int rows, int cols) {
    extern __shared__ float sdata[];
    const int num_warps = SOFTMAX_BLOCK / 32;
    float* row_data  = sdata;
    float* warp_vals = sdata + SOFTMAX_BLOCK;

    int row  = blockIdx.x;
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    float val = (tid < cols) ? in_row[tid] : -INFINITY;
    row_data[tid] = val;
    __syncthreads();

    // Max reduction
    float local_max = row_data[tid];
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();
    float row_max = (tid < num_warps) ? warp_vals[tid] : -INFINITY;
    row_max = warp_reduce_max(row_max);
    if (tid == 0) warp_vals[0] = row_max;
    __syncthreads();
    row_max = warp_vals[0];

    // Exp
    float exp_val = (tid < cols) ? expf(row_data[tid] - row_max) : 0.0f;
    row_data[tid] = exp_val;
    __syncthreads();

    // Sum reduction
    float local_sum = exp_val;
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_vals[wid] = local_sum;
    __syncthreads();
    float row_sum = (tid < num_warps) ? warp_vals[tid] : 0.0f;
    row_sum = warp_reduce_sum(row_sum);
    if (tid == 0) warp_vals[0] = row_sum;
    __syncthreads();
    row_sum = warp_vals[0];

    if (tid < cols) out_row[tid] = row_data[tid] / row_sum;
}

// ===========================================================================
// CPU Reference Attention
// ===========================================================================
//
// Same formula, implemented naively in C for correctness checking.
//
void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, int seq_len, int d_k, int d_v) {
    float scale = 1.0f / sqrtf((float)d_k);

    // Allocate intermediate matrices
    float* S = (float*)malloc(seq_len * seq_len * sizeof(float));  // QK^T
    float* P = (float*)malloc(seq_len * seq_len * sizeof(float));  // softmax(S)

    // Step 1: S = Q * K^T
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d_k; k++) {
                dot += Q[i * d_k + k] * K[j * d_k + k];  // K^T: swap j,k indices
            }
            S[i * seq_len + j] = dot * scale;
        }
    }

    // Step 2: P = softmax(S) row-wise
    for (int i = 0; i < seq_len; i++) {
        float* row = S + i * seq_len;
        float* out = P + i * seq_len;
        float row_max = row[0];
        for (int j = 1; j < seq_len; j++)
            if (row[j] > row_max) row_max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            out[j] = expf(row[j] - row_max);
            sum += out[j];
        }
        for (int j = 0; j < seq_len; j++) out[j] /= sum;
    }

    // Step 3: O = P * V
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_v; j++) {
            float dot = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                dot += P[i * seq_len + k] * V[k * d_v + j];
            }
            O[i * d_v + j] = dot;
        }
    }

    free(S);
    free(P);
}

// ===========================================================================
// Correctness Check
// ===========================================================================
bool check_correctness(const float* ref, const float* gpu, int n,
                       const char* label) {
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > EPSILON) {
            if (errors < 3)
                printf("  [%s] MISMATCH at %d: cpu=%.6f gpu=%.6f diff=%.2e\n",
                       label, i, ref[i], gpu[i], diff);
            errors++;
        }
    }
    if (errors == 0) {
        printf("  [%s] Correctness PASSED (max diff: %.2e)\n", label, max_diff);
        return true;
    } else {
        printf("  [%s] Correctness FAILED: %d / %d mismatches (max diff: %.2e)\n",
               label, errors, n, max_diff);
        return false;
    }
}

// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    const int N  = SEQ_LEN;
    const int dk = D_K;
    const int dv = D_V;

    printf("=== Baseline Attention Benchmark ===\n");
    printf("seq_len=%d, d_k=%d, d_v=%d\n", N, dk, dv);
    printf("S matrix size: %d x %d = %.2f MB\n\n",
           N, N, (float)(N * N * sizeof(float)) / (1024 * 1024));

    // -------------------------------------------------------------------------
    // Host allocations
    // -------------------------------------------------------------------------
    size_t qkv_bytes = (size_t)N * dk * sizeof(float);
    size_t s_bytes   = (size_t)N * N  * sizeof(float);
    size_t o_bytes   = (size_t)N * dv * sizeof(float);

    float* h_Q      = (float*)malloc(qkv_bytes);
    float* h_K      = (float*)malloc(qkv_bytes);
    float* h_V      = (float*)malloc(N * dv * sizeof(float));
    float* h_O_cpu  = (float*)malloc(o_bytes);
    float* h_O_gpu  = (float*)malloc(o_bytes);

    // Random init in [-1, 1]
    srand(42);
    for (int i = 0; i < N * dk; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < N * dv; i++)
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // -------------------------------------------------------------------------
    // CPU reference
    // -------------------------------------------------------------------------
    printf("[CPU] Computing reference attention...\n");
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, dk, dv);
    printf("[CPU] Done.\n\n");

    // -------------------------------------------------------------------------
    // Device allocations
    // -------------------------------------------------------------------------
    float *d_Q, *d_K, *d_V;
    float *d_S, *d_P, *d_O;

    CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, N * dv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, s_bytes));   // QK^T
    CUDA_CHECK(cudaMalloc(&d_P, s_bytes));   // softmax(S)
    CUDA_CHECK(cudaMalloc(&d_O, o_bytes));   // output

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N * dv * sizeof(float), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // GPU Attention — Step by Step
    // -------------------------------------------------------------------------

    // Note on K^T:
    // Q * K^T means we dot each row of Q with each row of K.
    // In memory, K is [N x dk] row-major.
    // K^T would be [dk x N]. Instead of actually transposing,
    // we pass K directly and adjust our GEMM to treat it as transposed:
    //   S[i][j] = sum_k Q[i][k] * K[j][k]
    // Our gemm_kernel computes C = A * B where B is [K_dim x N].
    // So we need K stored as [N x dk] and tell gemm that B has shape [dk x N]
    // by passing K with stride dk (which is just K as-is, treating rows as columns).
    // Simpler: just pass K as B with dimensions swapped — gemm_kernel handles it.

    // GEMM grid: ceil(N/BLOCK_SIZE) x ceil(N/BLOCK_SIZE) blocks
    dim3 gemm_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gemm_grid_S((N  + BLOCK_SIZE-1)/BLOCK_SIZE,
                     (N  + BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 gemm_grid_O((dv + BLOCK_SIZE-1)/BLOCK_SIZE,
                     (N  + BLOCK_SIZE-1)/BLOCK_SIZE);

    // Step 1: S = Q * K^T
    // Q is [N x dk], K is [N x dk], S is [N x N]
    // S[i][j] = dot(Q[i,:], K[j,:]) — use gemm_nt_kernel which reads K transposed
    gemm_nt_kernel<<<gemm_grid_S, gemm_block>>>(d_Q, d_K, d_S, N, dk, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Scale S by 1/sqrt(d_k)
    float scale = 1.0f / sqrtf((float)dk);
    int total_s = N * N;
    scale_kernel<<<(total_s + 255)/256, 256>>>(d_S, scale, total_s);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: P = softmax(S) — row-wise
    int smem = (SOFTMAX_BLOCK + SOFTMAX_BLOCK/32) * sizeof(float);
    softmax_kernel<<<N, SOFTMAX_BLOCK, smem>>>(d_S, d_P, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: O = P * V
    gemm_kernel<<<gemm_grid_O, gemm_block>>>(d_P, d_V, d_O, N, N, dv);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back and check
    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, o_bytes, cudaMemcpyDeviceToHost));
    check_correctness(h_O_cpu, h_O_gpu, N * dv, "attention");
    printf("\n");

    // -------------------------------------------------------------------------
    // Benchmark: time each step separately
    // -------------------------------------------------------------------------
    printf("--- Per-Step Timing (avg over %d runs) ---\n", NUM_RUNS);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float ms;

    // Step 1: QK^T GEMM
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NUM_RUNS; i++)
        gemm_nt_kernel<<<gemm_grid_S, gemm_block>>>(d_Q, d_K, d_S, N, dk, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float t_gemm1 = ms / NUM_RUNS;
    printf("  Step 1 QK^T GEMM:  %.4f ms\n", t_gemm1);

    // Step 2: Scale
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NUM_RUNS; i++)
        scale_kernel<<<(total_s+255)/256, 256>>>(d_S, scale, total_s);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float t_scale = ms / NUM_RUNS;
    printf("  Step 2 Scale:      %.4f ms\n", t_scale);

    // Step 3: Softmax
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NUM_RUNS; i++)
        softmax_kernel<<<N, SOFTMAX_BLOCK, smem>>>(d_S, d_P, N, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float t_softmax = ms / NUM_RUNS;
    printf("  Step 3 Softmax:    %.4f ms\n", t_softmax);

    // Step 4: PV GEMM
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NUM_RUNS; i++)
        gemm_kernel<<<gemm_grid_O, gemm_block>>>(d_P, d_V, d_O, N, N, dv);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float t_gemm2 = ms / NUM_RUNS;
    printf("  Step 4 PV GEMM:    %.4f ms\n", t_gemm2);

    float t_total = t_gemm1 + t_scale + t_softmax + t_gemm2;
    printf("  ─────────────────────────────\n");
    printf("  Total:             %.4f ms\n\n", t_total);

    // -------------------------------------------------------------------------
    // Memory usage analysis
    // -------------------------------------------------------------------------
    printf("--- Memory Footprint ---\n");
    float q_mb  = (float)qkv_bytes / (1024*1024);
    float k_mb  = (float)qkv_bytes / (1024*1024);
    float v_mb  = (float)(N * dv * sizeof(float)) / (1024*1024);
    float s_mb  = (float)s_bytes / (1024*1024);
    float o_mb  = (float)o_bytes / (1024*1024);
    printf("  Q:          %.3f MB\n", q_mb);
    printf("  K:          %.3f MB\n", k_mb);
    printf("  V:          %.3f MB\n", v_mb);
    printf("  S (QK^T):   %.3f MB  ← grows as seq_len^2 !\n", s_mb);
    printf("  P (softmax):%.3f MB  ← grows as seq_len^2 !\n", s_mb);
    printf("  O:          %.3f MB\n", o_mb);
    printf("  Total:      %.3f MB\n\n",
           q_mb + k_mb + v_mb + 2*s_mb + o_mb);

    printf("--- What Happens as seq_len Grows ---\n");
    printf("  seq_len=64:   S = %.2f MB\n",
           (float)(64*64*4)/(1024*1024));
    printf("  seq_len=512:  S = %.2f MB\n",
           (float)(512*512*4)/(1024*1024));
    printf("  seq_len=1024: S = %.2f MB\n",
           (float)(1024*1024*4)/(1024*1024));
    printf("  seq_len=4096: S = %.2f MB  ← Flash Attention's target\n",
           (float)(4096LL*4096*4)/(1024*1024));
    printf("\nThe S and P matrices are the problem Flash Attention solves.\n");
    printf("It never materializes them fully — that's Week 4.\n\n");

    printf("Try increasing SEQ_LEN to 128, 256, 512 and watch S grow.\n");
    printf("At SEQ_LEN=512 with d_k=64 you'll start seeing the memory cost.\n");

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_O));
    free(h_Q); free(h_K); free(h_V);
    free(h_O_cpu); free(h_O_gpu);

    return 0;
}

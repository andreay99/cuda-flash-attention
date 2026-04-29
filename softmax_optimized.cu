// =============================================================================
// softmax_optimized.cu — Week 2: Optimized CUDA Softmax
// =============================================================================
//
// What's new vs softmax_naive.cu:
//   1. One BLOCK per row (not one thread) — full parallelism within each row
//   2. Shared memory — threads load data once into fast on-chip memory
//   3. Parallel tree reduction — find max and sum in O(log N) steps
//   4. Warp shuffle (__shfl_down_sync) — fastest possible warp-level reduction
//   5. Side-by-side benchmark vs naive kernel
//
// How to compile:
//   nvcc -O2 -o softmax_optimized softmax_optimized.cu
//
// How to run:
//   ./softmax_optimized
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define BATCH_SIZE   1024    // number of rows
#define SEQ_LEN      512     // number of columns — MUST be <= BLOCK_SIZE * elements_per_thread
#define BLOCK_SIZE   512     // threads per block — one block owns one row
                             // rule of thumb: BLOCK_SIZE >= SEQ_LEN for this kernel
#define EPSILON      1e-5f
#define NUM_RUNS     100

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
// BACKGROUND: What is Shared Memory?
// ===========================================================================
//
// GPU memory hierarchy (fastest → slowest):
//
//   Registers     ~1 cycle    per-thread, tiny (~255 per thread)
//   Shared Memory ~5 cycles   per-block, 48–96 KB, programmer-controlled
//   L1 Cache      ~30 cycles  automatic
//   L2 Cache      ~200 cycles automatic
//   Global Memory ~600 cycles DRAM, what cudaMalloc gives you
//
// Shared memory is declared with __shared__ inside a kernel.
// ALL threads in a block can read/write it.
// It vanishes when the block finishes.
//
// The pattern:
//   1. Each thread loads its element(s) from global → shared memory
//   2. __syncthreads() — wait for ALL threads to finish loading
//   3. Threads collaborate using the now-fast shared memory
//   4. Write results back to global memory

// ===========================================================================
// BACKGROUND: What is a Warp Shuffle?
// ===========================================================================
//
// A "warp" is a group of 32 threads that execute in lockstep on the GPU.
// __shfl_down_sync lets threads share register values WITHOUT going through
// shared memory — the data moves through special warp-level interconnects.
//
// Example: parallel sum reduction across 32 threads using shuffles
//
//   Step 1 (offset=16): thread 0 gets thread 16's value, thread 1 gets thread 17's, etc.
//   Step 2 (offset=8):  thread 0 gets thread 8's value, etc.
//   Step 3 (offset=4):  thread 0 gets thread 4's value, etc.
//   Step 4 (offset=2):  thread 0 gets thread 2's value, etc.
//   Step 5 (offset=1):  thread 0 gets thread 1's value, etc.
//   → thread 0 now holds the sum of all 32 values
//
// This takes 5 steps (log2(32)) instead of 31 sequential additions.

// ===========================================================================
// Helper: Warp-level reduction (max or sum)
// ===========================================================================
//
// __shfl_down_sync(mask, val, offset):
//   - mask: which threads participate (0xFFFFFFFF = all 32)
//   - val:  the value THIS thread contributes
//   - offset: receive from the thread that is `offset` lanes ahead
//
__device__ float warp_reduce_max(float val) {
    // After each step, the "winning" value propagates down
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;  // lane 0 of each warp holds the max
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // lane 0 of each warp holds the sum
}

// ===========================================================================
// Optimized Softmax Kernel
// ===========================================================================
//
// Launch config: <<<BATCH_SIZE, BLOCK_SIZE>>>
//   - One block per row
//   - BLOCK_SIZE threads collaborate on their row
//
// Shared memory layout:
//   - sdata[BLOCK_SIZE]: holds the row's values (or intermediate results)
//   - warp_max[num_warps]: each warp's max, for cross-warp reduction
//   - warp_sum[num_warps]: each warp's sum, for cross-warp reduction
//
// num_warps = BLOCK_SIZE / 32
// For BLOCK_SIZE=512: num_warps = 16
//
__global__ void softmax_kernel_optimized(const float* input, float* output,
                                          int rows, int cols) {
    // -------------------------------------------------------------------------
    // Shared memory declarations
    // Each block gets its own private copy of these arrays.
    // -------------------------------------------------------------------------
    extern __shared__ float sdata[];  // size = BLOCK_SIZE floats (set at launch)
    // We'll also carve out space for warp reductions at the end of sdata
    // Layout: [0 .. cols-1] = row data, [cols .. cols+num_warps-1] = warp results
    // For simplicity here we use a fixed layout with BLOCK_SIZE total floats.

    const int num_warps = BLOCK_SIZE / 32;

    // Pointers into shared memory
    float* row_data  = sdata;                    // [0 .. BLOCK_SIZE-1]: row values
    float* warp_vals = sdata + BLOCK_SIZE;       // [BLOCK_SIZE .. BLOCK_SIZE+num_warps-1]
    // Note: total shared mem needed = (BLOCK_SIZE + num_warps) * sizeof(float)

    // -------------------------------------------------------------------------
    // Who am I?
    // -------------------------------------------------------------------------
    int row     = blockIdx.x;           // which row this block owns
    int tid     = threadIdx.x;          // my thread index within the block
    int lane    = tid % 32;             // my position within my warp (0–31)
    int warp_id = tid / 32;            // which warp I belong to (0–num_warps-1)

    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    // =========================================================================
    // PHASE 1: Load row into shared memory
    // =========================================================================
    //
    // Each thread loads one element. If cols < BLOCK_SIZE, out-of-range
    // threads load -infinity (so they don't affect max/sum).
    //
    float val = (tid < cols) ? in_row[tid] : -INFINITY;
    row_data[tid] = val;

    // __syncthreads(): BARRIER — every thread in the block must reach this
    // line before any thread can pass it. Ensures shared memory is fully
    // loaded before we start reading from it.
    __syncthreads();

    // =========================================================================
    // PHASE 2: Find the row maximum (for numerical stability)
    // =========================================================================
    //
    // Step A: Each warp finds its local max using shuffle reduction.
    //
    float local_max = row_data[tid];
    local_max = warp_reduce_max(local_max);
    // Now lane 0 of each warp holds that warp's max.

    // Step B: Lane 0 of each warp writes its result to shared memory.
    if (lane == 0) {
        warp_vals[warp_id] = local_max;
    }
    __syncthreads();  // wait for all warps to write their results

    // Step C: Thread 0 reads all warp maxes and finds the global max.
    // (Only one warp's worth of data to reduce — num_warps <= 16 for BLOCK_SIZE=512)
    float row_max = -INFINITY;
    if (tid < num_warps) {
        row_max = warp_vals[tid];
    }
    // One more warp reduction — thread 0 now holds the true row max.
    row_max = warp_reduce_max(row_max);

    // Step D: Broadcast row_max to all threads via shared memory.
    if (tid == 0) warp_vals[0] = row_max;
    __syncthreads();
    row_max = warp_vals[0];

    // =========================================================================
    // PHASE 3: Compute exp(x - max) and store back to shared memory
    // =========================================================================
    float exp_val = (tid < cols) ? expf(row_data[tid] - row_max) : 0.0f;
    row_data[tid] = exp_val;
    __syncthreads();

    // =========================================================================
    // PHASE 4: Sum all exp values (same pattern as max reduction)
    // =========================================================================

    // Step A: Each warp sums its local values.
    float local_sum = exp_val;
    local_sum = warp_reduce_sum(local_sum);

    // Step B: Lane 0 writes warp sum to shared memory.
    if (lane == 0) {
        warp_vals[warp_id] = local_sum;
    }
    __syncthreads();

    // Step C: Thread 0 sums all warp sums → global sum.
    float row_sum = 0.0f;
    if (tid < num_warps) {
        row_sum = warp_vals[tid];
    }
    row_sum = warp_reduce_sum(row_sum);

    // Step D: Broadcast row_sum.
    if (tid == 0) warp_vals[0] = row_sum;
    __syncthreads();
    row_sum = warp_vals[0];

    // =========================================================================
    // PHASE 5: Normalize and write to global memory
    // =========================================================================
    if (tid < cols) {
        out_row[tid] = row_data[tid] / row_sum;
    }
}

// ===========================================================================
// Naive kernel (copied from Week 1 for side-by-side comparison)
// ===========================================================================
__global__ void softmax_kernel_naive(const float* input, float* output,
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    float row_max = in_row[0];
    for (int j = 1; j < cols; j++) {
        if (in_row[j] > row_max) row_max = in_row[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        out_row[j] = expf(in_row[j] - row_max);
        sum += out_row[j];
    }
    for (int j = 0; j < cols; j++) {
        out_row[j] /= sum;
    }
}

// ===========================================================================
// CPU Reference
// ===========================================================================
void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        const float* in_row  = input  + row * cols;
        float*       out_row = output + row * cols;
        float row_max = in_row[0];
        for (int j = 1; j < cols; j++) {
            if (in_row[j] > row_max) row_max = in_row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            out_row[j] = expf(in_row[j] - row_max);
            sum += out_row[j];
        }
        for (int j = 0; j < cols; j++) {
            out_row[j] /= sum;
        }
    }
}

// ===========================================================================
// Correctness Check
// ===========================================================================
bool check_correctness(const float* cpu_out, const float* gpu_out,
                       int rows, int cols, const char* label) {
    int errors = 0;
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(cpu_out[i] - gpu_out[i]);
        if (diff > EPSILON) {
            if (errors < 5) {
                printf("  [%s] MISMATCH at %d: cpu=%.6f gpu=%.6f diff=%.2e\n",
                       label, i, cpu_out[i], gpu_out[i], diff);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  [%s] Correctness PASSED (%d elements)\n", label, rows * cols);
        return true;
    } else {
        printf("  [%s] Correctness FAILED: %d mismatches\n", label, errors);
        return false;
    }
}

// ===========================================================================
// Timing Helpers
// ===========================================================================
float time_naive(const float* d_in, float* d_out, int rows, int cols,
                 int block_size) {
    int grid = (rows + block_size - 1) / block_size;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // warm-up
    softmax_kernel_naive<<<grid, block_size>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++)
        softmax_kernel_naive<<<grid, block_size>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / NUM_RUNS;
}

float time_optimized(const float* d_in, float* d_out, int rows, int cols) {
    // One block per row, BLOCK_SIZE threads per block
    int grid = rows;
    // Shared memory: row data + warp reduction space
    int num_warps = BLOCK_SIZE / 32;
    size_t smem = (BLOCK_SIZE + num_warps) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // warm-up
    softmax_kernel_optimized<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++)
        softmax_kernel_optimized<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / NUM_RUNS;
}

// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    const int rows = BATCH_SIZE;
    const int cols = SEQ_LEN;
    const size_t bytes = (size_t)rows * cols * sizeof(float);

    printf("=== Softmax Optimization Benchmark ===\n");
    printf("Matrix: %d rows x %d cols (%.2f MB)\n\n", rows, cols,
           (float)bytes / (1024 * 1024));

    // Host memory
    float* h_input   = (float*)malloc(bytes);
    float* h_cpu_out = (float*)malloc(bytes);
    float* h_gpu_out = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;

    // CPU reference
    softmax_cpu(h_input, h_cpu_out, rows, cols);

    // Device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Naive kernel
    // -------------------------------------------------------------------------
    printf("--- Naive Kernel (1 thread per row) ---\n");
    {
        int block_size = 256;
        int grid = (rows + block_size - 1) / block_size;
        softmax_kernel_naive<<<grid, block_size>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, bytes, cudaMemcpyDeviceToHost));
        check_correctness(h_cpu_out, h_gpu_out, rows, cols, "naive");

        float ms = time_naive(d_input, d_output, rows, cols, block_size);
        float bw = (2.0f * bytes) / (ms * 1e-3f) / 1e9f;
        printf("  Grid: %d blocks x %d threads\n", grid, block_size);
        printf("  Avg time:  %.4f ms\n", ms);
        printf("  Bandwidth: %.2f GB/s\n\n", bw);
    }

    // -------------------------------------------------------------------------
    // Optimized kernel
    // -------------------------------------------------------------------------
    printf("--- Optimized Kernel (1 block per row, shared mem + warp shuffle) ---\n");
    {
        int num_warps = BLOCK_SIZE / 32;
        size_t smem = (BLOCK_SIZE + num_warps) * sizeof(float);

        softmax_kernel_optimized<<<rows, BLOCK_SIZE, smem>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, bytes, cudaMemcpyDeviceToHost));
        check_correctness(h_cpu_out, h_gpu_out, rows, cols, "optimized");

        float ms = time_optimized(d_input, d_output, rows, cols);
        float bw = (2.0f * bytes) / (ms * 1e-3f) / 1e9f;
        printf("  Grid: %d blocks x %d threads\n", rows, BLOCK_SIZE);
        printf("  Shared mem per block: %zu bytes\n", smem);
        printf("  Avg time:  %.4f ms\n", ms);
        printf("  Bandwidth: %.2f GB/s\n\n", bw);
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    printf("=== Experiment: vary SEQ_LEN ===\n");
    printf("Try changing SEQ_LEN and BLOCK_SIZE to these pairs and re-run:\n");
    printf("  SEQ_LEN=128,  BLOCK_SIZE=128\n");
    printf("  SEQ_LEN=512,  BLOCK_SIZE=512   (current)\n");
    printf("  SEQ_LEN=1024, BLOCK_SIZE=1024\n");
    printf("  SEQ_LEN=2048, BLOCK_SIZE=1024  (threads each handle 2 elements)\n");
    printf("\nNext up: attention_baseline.cu (Week 3)\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_cpu_out);
    free(h_gpu_out);

    return 0;
}

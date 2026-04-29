// =============================================================================
// softmax_naive.cu — Week 1 Starter: Naive CUDA Softmax
// =============================================================================
//
// What this file contains:
//   1. CPU softmax (reference / correctness check)
//   2. Naive CUDA softmax kernel (one thread per row — intentionally simple)
//   3. CUDA timing harness using cudaEvent
//   4. Correctness check: compare GPU output vs CPU output
//
// How to compile:
//   nvcc -O2 -o softmax_naive softmax_naive.cu
//
// How to run:
//   ./softmax_naive
//
// If you're on Google Colab (free T4 GPU), put this in a cell:
//   !nvcc -O2 -o softmax_naive softmax_naive.cu && ./softmax_naive
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Config — feel free to change these
// ---------------------------------------------------------------------------
#define BATCH_SIZE   1024   // number of rows (e.g., sequence length in attention)
#define SEQ_LEN      512    // number of columns (e.g., vocab size or key dimension)
#define EPSILON      1e-5f  // tolerance for correctness check

// ---------------------------------------------------------------------------
// Helper macro: check CUDA errors and bail out with a useful message
// ---------------------------------------------------------------------------
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
// PART 1: CPU Softmax (Reference Implementation)
// ===========================================================================
//
// For each row i in the matrix:
//   1. Find the max value (for numerical stability)
//   2. Subtract max, compute exp of each element
//   3. Sum all exp values
//   4. Divide each exp by the sum
//
// The "subtract max" trick prevents overflow: exp(large number) → inf
// Subtracting max doesn't change the output mathematically:
//   softmax(x) = softmax(x - max(x))  ← try proving this yourself!
//
void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        const float* in_row  = input  + row * cols;
        float*       out_row = output + row * cols;

        // Step 1: find max in this row
        float row_max = in_row[0];
        for (int j = 1; j < cols; j++) {
            if (in_row[j] > row_max) row_max = in_row[j];
        }

        // Step 2: compute exp(x - max) and accumulate sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            out_row[j] = expf(in_row[j] - row_max);
            sum += out_row[j];
        }

        // Step 3: normalize
        for (int j = 0; j < cols; j++) {
            out_row[j] /= sum;
        }
    }
}

// ===========================================================================
// PART 2: Naive CUDA Softmax Kernel
// ===========================================================================
//
// Strategy: one thread handles one entire row.
//
// This is naive because:
//   - Most threads in a warp might be doing redundant work
//   - We're not using shared memory (all reads/writes go to slow global memory)
//   - No parallelism WITHIN a row (one thread does all the work)
//
// But it's a great starting point! In Week 2 you'll replace this with
// a parallel version that uses shared memory and warp reductions.
//
// Grid/block setup:
//   - Each thread handles 1 row
//   - blockDim.x threads per block (e.g., 256)
//   - gridDim.x = ceil(BATCH_SIZE / blockDim.x) blocks
//
__global__ void softmax_kernel_naive(const float* input, float* output,
                                     int rows, int cols) {
    // Which row does this thread own?
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: don't go out of bounds if BATCH_SIZE isn't a multiple of blockDim.x
    if (row >= rows) return;

    // Pointers to the start of this thread's row
    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    // --- Step 1: Find row max (for numerical stability) ---
    float row_max = in_row[0];
    for (int j = 1; j < cols; j++) {
        if (in_row[j] > row_max) row_max = in_row[j];
    }

    // --- Step 2: Compute exp(x - max) and sum ---
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        out_row[j] = expf(in_row[j] - row_max);
        sum += out_row[j];
    }

    // --- Step 3: Normalize ---
    for (int j = 0; j < cols; j++) {
        out_row[j] /= sum;
    }
}

// ===========================================================================
// PART 3: Correctness Check
// ===========================================================================
//
// Compare GPU output vs CPU output element-by-element.
// Returns true if all values are within EPSILON of each other.
//
bool check_correctness(const float* cpu_out, const float* gpu_out,
                       int rows, int cols) {
    int errors = 0;
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(cpu_out[i] - gpu_out[i]);
        if (diff > EPSILON) {
            if (errors < 5) {  // only print first 5 mismatches
                printf("  MISMATCH at index %d: cpu=%.6f, gpu=%.6f, diff=%.2e\n",
                       i, cpu_out[i], gpu_out[i], diff);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  Correctness check PASSED (%d elements checked)\n",
               rows * cols);
        return true;
    } else {
        printf("  Correctness check FAILED: %d mismatches out of %d elements\n",
               errors, rows * cols);
        return false;
    }
}

// ===========================================================================
// PART 4: Timing Harness
// ===========================================================================
//
// CUDA events are the right way to time GPU kernels.
// cudaEventRecord() inserts a timestamp into the CUDA stream.
// cudaEventSynchronize() blocks the CPU until the GPU reaches that event.
// cudaEventElapsedTime() gives milliseconds between two events.
//
float time_kernel(const float* d_input, float* d_output,
                  int rows, int cols,
                  int block_size, int num_runs) {
    int grid_size = (rows + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up run (GPU has startup costs the first time)
    softmax_kernel_naive<<<grid_size, block_size>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        softmax_kernel_naive<<<grid_size, block_size>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / num_runs;  // average ms per run
}

// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    const int rows = BATCH_SIZE;
    const int cols = SEQ_LEN;
    const size_t matrix_bytes = rows * cols * sizeof(float);
    const int NUM_RUNS = 100;      // average over this many kernel runs
    const int BLOCK_SIZE = 256;    // threads per block (tweak this in Week 2!)

    printf("=== Naive CUDA Softmax ===\n");
    printf("Matrix: %d rows x %d cols (%.2f MB)\n\n",
           rows, cols, (float)matrix_bytes / (1024 * 1024));

    // -------------------------------------------------------------------------
    // Allocate host memory
    // -------------------------------------------------------------------------
    float* h_input   = (float*)malloc(matrix_bytes);
    float* h_cpu_out = (float*)malloc(matrix_bytes);
    float* h_gpu_out = (float*)malloc(matrix_bytes);

    if (!h_input || !h_cpu_out || !h_gpu_out) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Fill input with random values in [-3, 3]
    srand(42);
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    }

    // -------------------------------------------------------------------------
    // CPU softmax (reference)
    // -------------------------------------------------------------------------
    printf("[CPU] Running softmax...\n");
    softmax_cpu(h_input, h_cpu_out, rows, cols);
    printf("[CPU] Done.\n\n");

    // -------------------------------------------------------------------------
    // Allocate device memory and copy input
    // -------------------------------------------------------------------------
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, matrix_bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Run GPU kernel and verify correctness
    // -------------------------------------------------------------------------
    printf("[GPU] Running naive softmax kernel...\n");
    int grid_size = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_kernel_naive<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back and check
    CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output, matrix_bytes, cudaMemcpyDeviceToHost));
    check_correctness(h_cpu_out, h_gpu_out, rows, cols);
    printf("\n");

    // -------------------------------------------------------------------------
    // Benchmark
    // -------------------------------------------------------------------------
    printf("[Benchmark] Averaging over %d runs...\n", NUM_RUNS);
    float avg_ms = time_kernel(d_input, d_output, rows, cols, BLOCK_SIZE, NUM_RUNS);

    // Compute memory bandwidth utilization
    // We read input once, write output once = 2 * matrix_bytes
    float bandwidth_gb_s = (2.0f * matrix_bytes) / (avg_ms * 1e-3f) / 1e9f;

    printf("\n--- Results ---\n");
    printf("  Block size:        %d threads\n", BLOCK_SIZE);
    printf("  Grid size:         %d blocks\n", grid_size);
    printf("  Avg kernel time:   %.4f ms\n", avg_ms);
    printf("  Memory bandwidth:  %.2f GB/s\n", bandwidth_gb_s);
    printf("\nNote: A modern GPU (e.g., T4) has ~300 GB/s peak bandwidth.\n");
    printf("How close are you? That gap is what Week 2 is about.\n");

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_cpu_out);
    free(h_gpu_out);

    printf("\nDone!\n");
    return 0;
}

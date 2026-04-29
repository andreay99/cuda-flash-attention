// =============================================================================
// attention_flash.cu — Flash Attention V3: Block-Level Score Tile GEMM
// =============================================================================
//
// V1 problem: 1 thread per query row → sequential dot products
// V2 problem: 1 warp per query row → 32 serial warp reductions per tile
//
// V3 insight: treat the score tile (Br x Bc) as a mini-GEMM
//   Thread layout: dim3(Bc, Br) — a 2D block of Br*Bc threads
//   Thread (ty, tx) computes S[ty][tx] = dot(Q[ty], K[tx])
//   All Br*Bc scores computed in PARALLEL — no sequential loop over j
//   The dot product over d_k uses a shared-memory reduction within each row
//
// Then for the online softmax + PV:
//   Each row ty has Bc threads that collaborate on:
//   - finding row max (parallel reduction across tx dimension)
//   - computing exp weights
//   - accumulating P*V (each thread tx handles a subset of d_v dimensions)
//
// Launch: <<<N/Br, dim3(Bc, Br)>>>
//   blockDim = (Bc=32, Br=16) → 512 threads/block — good occupancy
//
// How to compile:
//   nvcc -O2 -o attention_flash attention_flash.cu
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define SEQ_LEN  1024
#define D_MODEL  64
#define Br       16     // query rows per block
#define Bc       32     // key cols per tile — also threadDim.x
                        // block = dim3(Bc, Br) = 512 threads
#define NUM_RUNS 100
#define EPSILON  1e-3f

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
// Flash Attention V3 Kernel
// ===========================================================================
//
// Thread identity:
//   tx = threadIdx.x ∈ [0, Bc)  — which key in the tile (column of score tile)
//   ty = threadIdx.y ∈ [0, Br)  — which query row in the tile
//
// Shared memory:
//   Qi[Br][D_MODEL]  — query tile
//   Kj[Bc][D_MODEL]  — key tile
//   Vj[Bc][D_MODEL]  — value tile
//   Sij[Br][Bc]      — score tile, computed as mini-GEMM
//   row_stats[Br][2] — per-row max and sum for online softmax
//
__global__ void flash_attention_v3(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int N, int d)
{
    __shared__ float Qi[Br][D_MODEL];
    __shared__ float Kj[Bc][D_MODEL];
    __shared__ float Vj[Bc][D_MODEL];
    __shared__ float Sij[Br][Bc];
    __shared__ float row_max[Br];   // per-row running max
    __shared__ float row_sum[Br];   // per-row running normalizer
    __shared__ float row_alpha[Br]; // per-row correction factor for O_acc rescaling

    int tx = threadIdx.x;   // key index within tile [0, Bc)
    int ty = threadIdx.y;   // query row within tile [0, Br)
    int q_row = blockIdx.x * Br + ty;  // absolute query row

    // -------------------------------------------------------------------------
    // Per-thread output accumulator
    // Thread (ty, tx) owns output dimensions: tx, tx+Bc, tx+2*Bc, ...
    // i.e., each thread handles ceil(d/Bc) output dimensions
    // For D_MODEL=64, Bc=32: each thread owns exactly 2 dimensions
    // -------------------------------------------------------------------------
    const int dims_per_thread = (D_MODEL + Bc - 1) / Bc;
    float O_acc[2];  // = dims_per_thread, hardcoded for D_MODEL=64, Bc=32
    O_acc[0] = 0.f; O_acc[1] = 0.f;

    // Running statistics — only meaningful for tx==0, but all threads track
    // their own copy and we use shared mem to broadcast
    float m_i = -INFINITY;
    float l_i = 0.f;

    // -------------------------------------------------------------------------
    // Load query tile: each thread loads its query row's dimensions
    // Thread (ty, tx) loads Qi[ty][tx] and Qi[ty][tx+Bc]
    // -------------------------------------------------------------------------
    if (q_row < N) {
        for (int k = tx; k < d; k += Bc) {
            Qi[ty][k] = Q[q_row * d + k];
        }
    } else {
        for (int k = tx; k < d; k += Bc) Qi[ty][k] = 0.f;
    }

    // Initialize row statistics
    if (tx == 0) { row_max[ty] = -INFINITY; row_sum[ty] = 0.f; row_alpha[ty] = 1.f; }
    __syncthreads();

    float scale = 1.f / sqrtf((float)d);
    int num_tiles = (N + Bc - 1) / Bc;

    // =========================================================================
    // Main loop over K/V tiles
    // =========================================================================
    for (int tile_j = 0; tile_j < num_tiles; tile_j++) {

        // --- Load K and V tiles ---
        // Thread (ty, tx) loads Kj[tx][...] and Vj[tx][...]
        // (ty threads cooperate on loading the tx-th row of K/V)
        int abs_kv = tile_j * Bc + tx;
        for (int k = ty; k < d; k += Br) {
            Kj[tx][k] = (abs_kv < N) ? K[abs_kv * d + k] : 0.f;
            Vj[tx][k] = (abs_kv < N) ? V[abs_kv * d + k] : 0.f;
        }
        __syncthreads();

        // --- Compute score tile as mini-GEMM ---
        // Thread (ty, tx) computes S[ty][tx] = dot(Qi[ty], Kj[tx])
        // Each thread does d sequential multiply-adds — no reduction needed!
        // All Br*Bc scores computed fully in parallel.
        float s = 0.f;
        if (q_row < N && abs_kv < N) {
            for (int k = 0; k < d; k++) {
                s += Qi[ty][k] * Kj[tx][k];
            }
            s *= scale;
        } else {
            s = -INFINITY;
        }
        Sij[ty][tx] = s;
        __syncthreads();

        // --- Online softmax: find row max and compute alpha ---
        // tx==0 computes tile max and the correction factor alpha,
        // then broadcasts both via shared memory so ALL threads can
        // rescale their own O_acc registers correctly.
        if (tx == 0) {
            float tile_max = -INFINITY;
            for (int j = 0; j < Bc; j++)
                tile_max = fmaxf(tile_max, Sij[ty][j]);

            float m_new   = fmaxf(m_i, tile_max);
            float alpha   = expf(m_i - m_new);  // correction for old accumulator

            row_alpha[ty] = alpha;   // broadcast alpha to all tx threads
            row_sum[ty]   = alpha * row_sum[ty];
            m_i           = m_new;
            row_max[ty]   = m_new;
        }
        __syncthreads();

        // All threads read the updated max and alpha
        m_i          = row_max[ty];
        float alpha  = row_alpha[ty];

        // Rescale O_acc by alpha — every thread does this for its own dims
        for (int local_d = 0; local_d < dims_per_thread; local_d++) {
            O_acc[local_d] *= alpha;
        }

        // --- Compute exp weights ---
        float exp_s = expf(Sij[ty][tx] - m_i);
        Sij[ty][tx] = exp_s;
        __syncthreads();

        // Thread tx==0 accumulates row sum
        if (tx == 0) {
            float tile_sum = 0.f;
            for (int j = 0; j < Bc; j++) tile_sum += Sij[ty][j];
            row_sum[ty] += tile_sum;
        }
        __syncthreads();
        l_i = row_sum[ty];  // broadcast

        // --- Update output accumulator: O_acc += P_tile * V_tile ---
        // O_acc is already rescaled by alpha above, so just add new contribution
        for (int local_d = 0; local_d < dims_per_thread; local_d++) {
            int out_dim = local_d * Bc + tx;
            if (out_dim < d) {
                float pv = 0.f;
                for (int j = 0; j < Bc; j++) {
                    pv += Sij[ty][j] * Vj[j][out_dim];
                }
                O_acc[local_d] += pv;
            }
        }
        __syncthreads();
    }

    // =========================================================================
    // Normalize and write output
    // =========================================================================
    if (q_row < N) {
        for (int local_d = 0; local_d < dims_per_thread; local_d++) {
            int out_dim = local_d * Bc + tx;
            if (out_dim < d) {
                O[q_row * d + out_dim] = O_acc[local_d] / l_i;
            }
        }
    }
}

// ===========================================================================
// Baseline kernels
// ===========================================================================
__global__ void gemm_nt_kernel(const float* A,const float* B,float* C,int M,int K,int N){
    #define BS 16
    __shared__ float tA[BS][BS],tB[BS][BS];
    int row=blockIdx.y*BS+threadIdx.y,col=blockIdx.x*BS+threadIdx.x; float acc=0.f;
    for(int t=0;t<(K+BS-1)/BS;t++){
        int ac=t*BS+threadIdx.x; tA[threadIdx.y][threadIdx.x]=(row<M&&ac<K)?A[row*K+ac]:0.f;
        int bc=t*BS+threadIdx.y,br=blockIdx.x*BS+threadIdx.x;
        tB[threadIdx.y][threadIdx.x]=(br<N&&bc<K)?B[br*K+bc]:0.f;
        __syncthreads(); for(int k=0;k<BS;k++) acc+=tA[threadIdx.y][k]*tB[k][threadIdx.x]; __syncthreads();
    } if(row<M&&col<N) C[row*N+col]=acc;
    #undef BS
}
__global__ void gemm_kernel(const float* A,const float* B,float* C,int M,int K,int N){
    #define BS 16
    __shared__ float tA[BS][BS],tB[BS][BS];
    int row=blockIdx.y*BS+threadIdx.y,col=blockIdx.x*BS+threadIdx.x; float acc=0.f;
    for(int t=0;t<(K+BS-1)/BS;t++){
        int ac=t*BS+threadIdx.x; tA[threadIdx.y][threadIdx.x]=(row<M&&ac<K)?A[row*K+ac]:0.f;
        int br=t*BS+threadIdx.y; tB[threadIdx.y][threadIdx.x]=(br<K&&col<N)?B[br*N+col]:0.f;
        __syncthreads(); for(int k=0;k<BS;k++) acc+=tA[threadIdx.y][k]*tB[k][threadIdx.x]; __syncthreads();
    } if(row<M&&col<N) C[row*N+col]=acc;
    #undef BS
}
__device__ float dmax(float v){for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_down_sync(0xFFFFFFFF,v,o));return v;}
__device__ float dsum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xFFFFFFFF,v,o);return v;}
__global__ void softmax_kernel(const float* in,float* out,int rows,int cols){
    #define SB 512
    extern __shared__ float sd[]; float *rd=sd,*wv=sd+SB;
    int row=blockIdx.x,tid=threadIdx.x,lane=tid%32,wid=tid/32;
    if(row>=rows) return;
    float v=(tid<cols)?in[row*cols+tid]:-INFINITY; rd[tid]=v; __syncthreads();
    float lm=rd[tid]; lm=dmax(lm); if(lane==0)wv[wid]=lm; __syncthreads();
    float rm=(tid<SB/32)?wv[tid]:-INFINITY; rm=dmax(rm); if(tid==0)wv[0]=rm; __syncthreads(); rm=wv[0];
    float ev=(tid<cols)?expf(rd[tid]-rm):0.f; rd[tid]=ev; __syncthreads();
    float ls=ev; ls=dsum(ls); if(lane==0)wv[wid]=ls; __syncthreads();
    float rs=(tid<SB/32)?wv[tid]:0.f; rs=dsum(rs); if(tid==0)wv[0]=rs; __syncthreads(); rs=wv[0];
    if(tid<cols) out[row*cols+tid]=rd[tid]/rs;
    #undef SB
}
__global__ void scale_kernel(float* S,float sc,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)S[i]*=sc;}

// ===========================================================================
// CPU Reference
// ===========================================================================
void attention_cpu(const float* Q,const float* K,const float* V,float* O,int N,int d){
    float sc=1.f/sqrtf((float)d);
    float *S=(float*)malloc(N*N*4),*P=(float*)malloc(N*N*4);
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){float dot=0.f;for(int k=0;k<d;k++)dot+=Q[i*d+k]*K[j*d+k];S[i*N+j]=dot*sc;}
    for(int i=0;i<N;i++){float mx=S[i*N];for(int j=1;j<N;j++)if(S[i*N+j]>mx)mx=S[i*N+j];float sm=0.f;for(int j=0;j<N;j++){P[i*N+j]=expf(S[i*N+j]-mx);sm+=P[i*N+j];}for(int j=0;j<N;j++)P[i*N+j]/=sm;}
    for(int i=0;i<N;i++) for(int j=0;j<d;j++){float dot=0.f;for(int k=0;k<N;k++)dot+=P[i*N+k]*V[k*d+j];O[i*d+j]=dot;}
    free(S);free(P);
}

bool check(const float* ref,const float* gpu,int n,const char* label){
    int err=0;float mx=0.f;
    for(int i=0;i<n;i++){float d=fabsf(ref[i]-gpu[i]);if(d>mx)mx=d;if(d>EPSILON){if(err<3)printf("  [%s] MISMATCH at %d: cpu=%.6f gpu=%.6f diff=%.2e\n",label,i,ref[i],gpu[i],d);err++;}}
    if(!err){printf("  [%s] Correctness PASSED (max diff: %.2e)\n",label,mx);return true;}
    printf("  [%s] FAILED: %d/%d (max diff: %.2e)\n",label,err,n,mx);return false;
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(){
    const int N=SEQ_LEN,d=D_MODEL;
    dim3 block(Bc,Br);
    int grid=N/Br;

    printf("=== Flash Attention V3 (Block-Level Score GEMM) vs Baseline ===\n");
    printf("seq_len=%d, d_k=%d, Br=%d, Bc=%d, %d threads/block\n\n",N,d,Br,Bc,Br*Bc);

    size_t qb=(size_t)N*d*4,sb=(size_t)N*N*4;
    float *hQ=(float*)malloc(qb),*hK=(float*)malloc(qb),*hV=(float*)malloc(qb);
    float *hCpu=(float*)malloc(qb),*hGpu=(float*)malloc(qb);
    srand(42);
    for(int i=0;i<N*d;i++){hQ[i]=((float)rand()/RAND_MAX)*2.f-1.f;hK[i]=((float)rand()/RAND_MAX)*2.f-1.f;hV[i]=((float)rand()/RAND_MAX)*2.f-1.f;}

    printf("[CPU] Computing reference...\n");
    attention_cpu(hQ,hK,hV,hCpu,N,d);
    printf("[CPU] Done.\n\n");

    float *dQ,*dK,*dV,*dO,*dS,*dP;
    CUDA_CHECK(cudaMalloc(&dQ,qb));CUDA_CHECK(cudaMalloc(&dK,qb));CUDA_CHECK(cudaMalloc(&dV,qb));
    CUDA_CHECK(cudaMalloc(&dO,qb));CUDA_CHECK(cudaMalloc(&dS,sb));CUDA_CHECK(cudaMalloc(&dP,sb));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,qb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,hK,qb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,qb,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1;
    CUDA_CHECK(cudaEventCreate(&t0));CUDA_CHECK(cudaEventCreate(&t1));

    // Flash V3
    printf("--- Flash Attention V3 ---\n");
    flash_attention_v3<<<grid,block>>>(dQ,dK,dV,dO,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hGpu,dO,qb,cudaMemcpyDeviceToHost));
    check(hCpu,hGpu,N*d,"flash_v3");

    CUDA_CHECK(cudaEventRecord(t0));
    for(int i=0;i<NUM_RUNS;i++) flash_attention_v3<<<grid,block>>>(dQ,dK,dV,dO,N,d);
    CUDA_CHECK(cudaEventRecord(t1));CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
    float tf=ms/NUM_RUNS;
    printf("  Avg time:    %.4f ms\n",tf);
    printf("  HBM traffic: %.2f MB (Q+K+V only)\n",3.f*qb/(1024*1024));
    printf("  Bandwidth:   %.2f GB/s\n\n",(3.f*qb)/(tf*1e-3f)/1e9f);

    // Baseline
    printf("--- Baseline (4-kernel pipeline) ---\n");
    {
        dim3 bg(16,16),ggs((N+15)/16,(N+15)/16),ggo((d+15)/16,(N+15)/16);
        float sc=1.f/sqrtf((float)d);
        int smem=(512+512/32)*sizeof(float);
        CUDA_CHECK(cudaEventRecord(t0));
        for(int i=0;i<NUM_RUNS;i++){
            gemm_nt_kernel<<<ggs,bg>>>(dQ,dK,dS,N,d,N);
            scale_kernel<<<(N*N+255)/256,256>>>(dS,sc,N*N);
            softmax_kernel<<<N,512,smem>>>(dS,dP,N,N);
            gemm_kernel<<<ggo,bg>>>(dP,dV,dO,N,N,d);
        }
        CUDA_CHECK(cudaEventRecord(t1));CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
        float tb=ms/NUM_RUNS;
        float bb=3.f*qb+4.f*sb+qb;
        printf("  Avg time:    %.4f ms\n",tb);
        printf("  HBM traffic: %.2f MB\n",bb/(1024*1024));
        printf("  Bandwidth:   %.2f GB/s\n\n",bb/(tb*1e-3f)/1e9f);

        printf("=== Summary ===\n");
        printf("  Flash V3:  %.4f ms\n",tf);
        printf("  Baseline:  %.4f ms\n",tb);
        printf("  Speedup:   %.2fx\n\n",tb/tf);
        printf("  Flash HBM:    %.2f MB\n",3.f*qb/(1024*1024));
        printf("  Baseline HBM: %.2f MB\n",bb/(1024*1024));
        printf("  HBM reduction: %.2fx\n",bb/(3.f*qb));
    }

    CUDA_CHECK(cudaEventDestroy(t0));CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(dQ));CUDA_CHECK(cudaFree(dK));CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));CUDA_CHECK(cudaFree(dS));CUDA_CHECK(cudaFree(dP));
    free(hQ);free(hK);free(hV);free(hCpu);free(hGpu);
    return 0;
}

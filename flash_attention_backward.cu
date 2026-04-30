// =============================================================================
// flash_attention_backward.cu — Flash Attention Forward + Backward in Raw CUDA
// =============================================================================
//
// This is the project most student CUDA repos don't have.
// Standard attention stores the full N×N softmax matrix P during forward,
// then reads it back during backward — O(N²) memory.
//
// Flash Attention backward (Dao et al. 2022, Algorithm 4):
//   - Forward saves only m[N] and l[N] (row max and normalizer)
//   - Backward RECOMPUTES P tile by tile from Q, K, m, l
//   - Never materializes the full N×N matrix at any point
//   - Memory: O(N·d) throughout forward AND backward
//
// Gradients computed:
//   dV = P^T · dO                          [N × d]
//   D  = rowsum(dO ⊙ O)                   [N]   (softmax backward scalar)
//   dS = P ⊙ (dP - D)  where dP = dO·V^T [N × N] (never stored)
//   dQ = dS · K / sqrt(d)                  [N × d]
//   dK = dS^T · Q / sqrt(d)               [N × d]
//
// How to compile:
//   nvcc -O2 -o flash_backward_bin flash_attention_backward.cu
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
#define SEQ_LEN   256
#define D_MODEL   64
#define Br        16
#define Bc        32
#define EPSILON   1e-3f

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ===========================================================================
// FORWARD PASS — saves m and l for backward
// ===========================================================================
__global__ void flash_attn_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ m_out,
    float*       __restrict__ l_out,
    int N, int d)
{
    __shared__ float Qi[Br][D_MODEL];
    __shared__ float Kj[Bc][D_MODEL];
    __shared__ float Vj[Bc][D_MODEL];
    __shared__ float Sij[Br][Bc];
    __shared__ float row_max[Br];
    __shared__ float row_sum[Br];
    __shared__ float row_alpha[Br];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int q_row = blockIdx.x * Br + ty;

    const int dims_per_thread = (D_MODEL + Bc - 1) / Bc;
    float O_acc[2]; O_acc[0] = 0.f; O_acc[1] = 0.f;
    float m_i = -INFINITY, l_i = 0.f;

    if (q_row < N) {
        for (int k = tx; k < d; k += Bc) Qi[ty][k] = Q[q_row * d + k];
    } else {
        for (int k = tx; k < d; k += Bc) Qi[ty][k] = 0.f;
    }
    if (tx == 0) { row_max[ty] = -INFINITY; row_sum[ty] = 0.f; row_alpha[ty] = 1.f; }
    __syncthreads();

    float scale = 1.f / sqrtf((float)d);
    int num_tiles = (N + Bc - 1) / Bc;

    for (int tile_j = 0; tile_j < num_tiles; tile_j++) {
        int abs_kv = tile_j * Bc + tx;
        for (int k = ty; k < d; k += Br) {
            Kj[tx][k] = (abs_kv < N) ? K[abs_kv * d + k] : 0.f;
            Vj[tx][k] = (abs_kv < N) ? V[abs_kv * d + k] : 0.f;
        }
        __syncthreads();

        float s = 0.f;
        if (q_row < N && abs_kv < N) {
            for (int k = 0; k < d; k++) s += Qi[ty][k] * Kj[tx][k];
            s *= scale;
        } else { s = -INFINITY; }
        Sij[ty][tx] = s;
        __syncthreads();

        if (tx == 0) {
            float tile_max = -INFINITY;
            for (int j = 0; j < Bc; j++) tile_max = fmaxf(tile_max, Sij[ty][j]);
            float m_new = fmaxf(m_i, tile_max);
            row_alpha[ty] = expf(m_i - m_new);
            row_sum[ty]   = row_alpha[ty] * row_sum[ty];
            m_i = m_new; row_max[ty] = m_new;
        }
        __syncthreads();
        m_i = row_max[ty];
        float alpha = row_alpha[ty];
        for (int ld = 0; ld < dims_per_thread; ld++) O_acc[ld] *= alpha;

        float exp_s = expf(Sij[ty][tx] - m_i);
        Sij[ty][tx] = exp_s;
        __syncthreads();

        if (tx == 0) {
            float ts = 0.f;
            for (int j = 0; j < Bc; j++) ts += Sij[ty][j];
            row_sum[ty] += ts;
        }
        __syncthreads();
        l_i = row_sum[ty];

        for (int ld = 0; ld < dims_per_thread; ld++) {
            int out_dim = ld * Bc + tx;
            if (out_dim < d) {
                float pv = 0.f;
                for (int j = 0; j < Bc; j++) pv += Sij[ty][j] * Vj[j][out_dim];
                O_acc[ld] += pv;
            }
        }
        __syncthreads();
    }

    if (q_row < N) {
        for (int ld = 0; ld < dims_per_thread; ld++) {
            int out_dim = ld * Bc + tx;
            if (out_dim < d) O[q_row * d + out_dim] = O_acc[ld] / l_i;
        }
        if (tx == 0) {
            m_out[q_row] = m_i;
            l_out[q_row] = l_i;
        }
    }
}

// ===========================================================================
// BACKWARD STEP 1: D = rowsum(dO * O)
// ===========================================================================
__global__ void compute_D(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float*       __restrict__ D,
    int N, int d)
{
    __shared__ float sdata[D_MODEL];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    float partial = 0.f;
    for (int k = tid; k < d; k += blockDim.x)
        partial += dO[row * d + k] * O[row * d + k];
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) D[row] = sdata[0];
}

// ===========================================================================
// BACKWARD STEP 2: dV
// ===========================================================================
// Launch: <<<N, D_MODEL>>>
// blockIdx.x  = which KV row to compute dV for (one block per KV row)
// threadIdx.x = output dimension (0..d-1)
// Each thread accumulates dV[kv_row][tx] = sum over all query rows of P[q][kv]*dO[q][tx]
__global__ void flash_attn_backward_dV(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ dO,
    const float* __restrict__ m,
    const float* __restrict__ l,
    float*       __restrict__ dV,
    int N, int d)
{
    int kv_row = blockIdx.x;   // one block per KV row
    int tx     = threadIdx.x;  // output dimension

    if (kv_row >= N || tx >= d) return;

    float scale = 1.f / sqrtf((float)d);
    float dv_acc = 0.f;

    for (int abs_q = 0; abs_q < N; abs_q++) {
        // Recompute S[abs_q][kv_row]
        float s = 0.f;
        for (int k = 0; k < d; k++)
            s += Q[abs_q * d + k] * K[kv_row * d + k];
        s *= scale;

        float p = expf(s - m[abs_q]) / l[abs_q];
        dv_acc += p * dO[abs_q * d + tx];
    }

    dV[kv_row * d + tx] = dv_acc;
}

// ===========================================================================
// BACKWARD STEP 3: dQ
// ===========================================================================
// Launch: <<<N, D_MODEL>>>
// blockIdx.x  = which Q row
// threadIdx.x = output dimension
__global__ void flash_attn_backward_dQ(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ m,
    const float* __restrict__ l,
    const float* __restrict__ D,
    float*       __restrict__ dQ,
    int N, int d)
{
    int q_row = blockIdx.x;
    int tx    = threadIdx.x;

    if (q_row >= N || tx >= d) return;

    float scale = 1.f / sqrtf((float)d);
    float dq_acc = 0.f;
    float m_i = m[q_row];
    float l_i = l[q_row];
    float D_i = D[q_row];

    for (int abs_kv = 0; abs_kv < N; abs_kv++) {
        // Recompute S[q_row][abs_kv]
        float s = 0.f;
        for (int k = 0; k < d; k++)
            s += Q[q_row * d + k] * K[abs_kv * d + k];
        s *= scale;

        float p = expf(s - m_i) / l_i;

        // dP[q_row][abs_kv] = dot(dO[q_row], V[abs_kv])
        float dp = 0.f;
        for (int k = 0; k < d; k++)
            dp += dO[q_row * d + k] * V[abs_kv * d + k];

        float ds = p * (dp - D_i);
        dq_acc += ds * K[abs_kv * d + tx] * scale;
    }

    dQ[q_row * d + tx] = dq_acc;
}

// ===========================================================================
// BACKWARD STEP 4: dK
// ===========================================================================
// Launch: <<<N, D_MODEL>>>
// blockIdx.x  = which K row
// threadIdx.x = output dimension
__global__ void flash_attn_backward_dK(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ m,
    const float* __restrict__ l,
    const float* __restrict__ D,
    float*       __restrict__ dK,
    int N, int d)
{
    int kv_row = blockIdx.x;
    int tx     = threadIdx.x;

    if (kv_row >= N || tx >= d) return;

    float scale = 1.f / sqrtf((float)d);
    float dk_acc = 0.f;

    for (int abs_q = 0; abs_q < N; abs_q++) {
        float m_i = m[abs_q];
        float l_i = l[abs_q];
        float D_i = D[abs_q];

        // Recompute S[abs_q][kv_row]
        float s = 0.f;
        for (int k = 0; k < d; k++)
            s += Q[abs_q * d + k] * K[kv_row * d + k];
        s *= scale;

        float p = expf(s - m_i) / l_i;

        // dP[abs_q][kv_row] = dot(dO[abs_q], V[kv_row])
        float dp = 0.f;
        for (int k = 0; k < d; k++)
            dp += dO[abs_q * d + k] * V[kv_row * d + k];

        float ds = p * (dp - D_i);
        dk_acc += ds * Q[abs_q * d + tx] * scale;
    }

    dK[kv_row * d + tx] = dk_acc;
}

// ===========================================================================
// CPU Reference
// ===========================================================================
void attention_cpu_forward(const float* Q, const float* K, const float* V,
                            float* O, float* m_cpu, float* l_cpu, int N, int d) {
    float scale = 1.f / sqrtf((float)d);
    float* S = (float*)malloc(N*N*4);
    float* P = (float*)malloc(N*N*4);
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) {
        float dot=0.f; for (int k=0;k<d;k++) dot+=Q[i*d+k]*K[j*d+k];
        S[i*N+j]=dot*scale;
    }
    for (int i=0;i<N;i++) {
        float mx=S[i*N]; for(int j=1;j<N;j++) if(S[i*N+j]>mx) mx=S[i*N+j];
        float sm=0.f; for(int j=0;j<N;j++){P[i*N+j]=expf(S[i*N+j]-mx);sm+=P[i*N+j];}
        for(int j=0;j<N;j++) P[i*N+j]/=sm;
        if(m_cpu) m_cpu[i]=mx;
        if(l_cpu) l_cpu[i]=sm;
    }
    for(int i=0;i<N;i++) for(int j=0;j<d;j++){
        float dot=0.f; for(int k=0;k<N;k++) dot+=P[i*N+k]*V[k*d+j];
        O[i*d+j]=dot;
    }
    free(S); free(P);
}

void attention_cpu_backward(const float* Q, const float* K, const float* V,
                             const float* dO, float* dQ, float* dK, float* dV,
                             int N, int d) {
    float scale = 1.f / sqrtf((float)d);
    float* S = (float*)malloc(N*N*4);
    float* P = (float*)malloc(N*N*4);
    float* O = (float*)malloc(N*d*4);
    attention_cpu_forward(Q,K,V,O,NULL,NULL,N,d);
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        float dot=0.f; for(int k=0;k<d;k++) dot+=Q[i*d+k]*K[j*d+k];
        S[i*N+j]=dot*scale;
    }
    for(int i=0;i<N;i++){
        float mx=S[i*N]; for(int j=1;j<N;j++) if(S[i*N+j]>mx) mx=S[i*N+j];
        float sm=0.f; for(int j=0;j<N;j++){P[i*N+j]=expf(S[i*N+j]-mx);sm+=P[i*N+j];}
        for(int j=0;j<N;j++) P[i*N+j]/=sm;
    }
    for(int j=0;j<N;j++) for(int k=0;k<d;k++){
        float acc=0.f; for(int i=0;i<N;i++) acc+=P[i*N+j]*dO[i*d+k];
        dV[j*d+k]=acc;
    }
    float* D=(float*)malloc(N*4);
    for(int i=0;i<N;i++){
        float dot=0.f; for(int k=0;k<d;k++) dot+=dO[i*d+k]*O[i*d+k];
        D[i]=dot;
    }
    float* dP=(float*)malloc(N*N*4);
    float* dS=(float*)malloc(N*N*4);
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        float dot=0.f; for(int k=0;k<d;k++) dot+=dO[i*d+k]*V[j*d+k];
        dP[i*N+j]=dot;
        dS[i*N+j]=P[i*N+j]*(dP[i*N+j]-D[i]);
    }
    for(int i=0;i<N;i++) for(int k=0;k<d;k++){
        float acc=0.f; for(int j=0;j<N;j++) acc+=dS[i*N+j]*K[j*d+k];
        dQ[i*d+k]=acc*scale;
    }
    for(int j=0;j<N;j++) for(int k=0;k<d;k++){
        float acc=0.f; for(int i=0;i<N;i++) acc+=dS[i*N+j]*Q[i*d+k];
        dK[j*d+k]=acc*scale;
    }
    free(S);free(P);free(O);free(D);free(dP);free(dS);
}

// ===========================================================================
// Correctness Check
// ===========================================================================
bool check(const float* ref, const float* gpu, int n, const char* label) {
    int err=0; float mx=0.f;
    for(int i=0;i<n;i++){
        float df=fabsf(ref[i]-gpu[i]);
        if(df>mx) mx=df;
        if(df>EPSILON){
            if(err<3) printf("  [%s] MISMATCH at %d: cpu=%.6f gpu=%.6f diff=%.2e\n",
                             label,i,ref[i],gpu[i],df);
            err++;
        }
    }
    if(!err){printf("  [%s] PASSED (max diff: %.2e)\n",label,mx);return true;}
    printf("  [%s] FAILED: %d/%d (max diff: %.2e)\n",label,err,n,mx);
    return false;
}

// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    const int N=SEQ_LEN, d=D_MODEL;
    printf("=== Flash Attention Forward + Backward ===\n");
    printf("seq_len=%d, d=%d, Br=%d, Bc=%d\n\n",N,d,Br,Bc);

    size_t qb=(size_t)N*d*4;
    float *hQ=(float*)malloc(qb),*hK=(float*)malloc(qb),*hV=(float*)malloc(qb);
    float *hO_cpu=(float*)malloc(qb),*hdO=(float*)malloc(qb);
    float *hdQ_cpu=(float*)malloc(qb),*hdK_cpu=(float*)malloc(qb),*hdV_cpu=(float*)malloc(qb);
    float *hdQ_gpu=(float*)malloc(qb),*hdK_gpu=(float*)malloc(qb),*hdV_gpu=(float*)malloc(qb);
    float *hO_gpu=(float*)malloc(qb);
    float *hm=(float*)malloc(N*4),*hl=(float*)malloc(N*4),*hD=(float*)malloc(N*4);

    srand(42);
    for(int i=0;i<N*d;i++){
        hQ[i]=((float)rand()/RAND_MAX)*2.f-1.f;
        hK[i]=((float)rand()/RAND_MAX)*2.f-1.f;
        hV[i]=((float)rand()/RAND_MAX)*2.f-1.f;
    }
    for(int i=0;i<N*d;i++) hdO[i]=1.f;

    printf("[CPU] Computing reference...\n");
    attention_cpu_forward(hQ,hK,hV,hO_cpu,NULL,NULL,N,d);
    attention_cpu_backward(hQ,hK,hV,hdO,hdQ_cpu,hdK_cpu,hdV_cpu,N,d);
    printf("[CPU] Done.\n\n");

    float *dQ,*dK,*dV,*dO_d,*dO_out,*dm,*dl,*dD,*ddQ,*ddK,*ddV;
    CUDA_CHECK(cudaMalloc(&dQ,qb));  CUDA_CHECK(cudaMalloc(&dK,qb));
    CUDA_CHECK(cudaMalloc(&dV,qb));  CUDA_CHECK(cudaMalloc(&dO_d,qb));
    CUDA_CHECK(cudaMalloc(&dO_out,qb));
    CUDA_CHECK(cudaMalloc(&dm,N*4)); CUDA_CHECK(cudaMalloc(&dl,N*4));
    CUDA_CHECK(cudaMalloc(&dD,N*4));
    CUDA_CHECK(cudaMalloc(&ddQ,qb)); CUDA_CHECK(cudaMalloc(&ddK,qb));
    CUDA_CHECK(cudaMalloc(&ddV,qb));

    CUDA_CHECK(cudaMemcpy(dQ,hQ,qb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,hK,qb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,qb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO_d,hdO,qb,cudaMemcpyHostToDevice));

    // Forward
    printf("--- Forward Pass ---\n");
    flash_attn_forward<<<N/Br, dim3(Bc,Br)>>>(dQ,dK,dV,dO_out,dm,dl,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO_gpu,dO_out,qb,cudaMemcpyDeviceToHost));
    check(hO_cpu,hO_gpu,N*d,"forward O");
    printf("\n");

    // D
    printf("--- Backward: D ---\n");
    compute_D<<<N,D_MODEL>>>(dO_d,dO_out,dD,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  D computed.\n\n");

    // dV — <<<N, D_MODEL>>> = 256 blocks of 64 threads
    printf("--- Backward: dV ---\n");
    flash_attn_backward_dV<<<N,D_MODEL>>>(dQ,dK,dO_d,dm,dl,ddV,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdV_gpu,ddV,qb,cudaMemcpyDeviceToHost));
    check(hdV_cpu,hdV_gpu,N*d,"dV");
    printf("\n");

    // dQ — <<<N, D_MODEL>>>
    printf("--- Backward: dQ ---\n");
    flash_attn_backward_dQ<<<N,D_MODEL>>>(dQ,dK,dV,dO_d,dm,dl,dD,ddQ,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdQ_gpu,ddQ,qb,cudaMemcpyDeviceToHost));
    check(hdQ_cpu,hdQ_gpu,N*d,"dQ");
    printf("\n");

    // dK — <<<N, D_MODEL>>>
    printf("--- Backward: dK ---\n");
    flash_attn_backward_dK<<<N,D_MODEL>>>(dQ,dK,dV,dO_d,dm,dl,dD,ddK,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdK_gpu,ddK,qb,cudaMemcpyDeviceToHost));
    check(hdK_cpu,hdK_gpu,N*d,"dK");
    printf("\n");

    printf("=== All gradient checks complete ===\n");

    CUDA_CHECK(cudaFree(dQ));CUDA_CHECK(cudaFree(dK));CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO_d));CUDA_CHECK(cudaFree(dO_out));
    CUDA_CHECK(cudaFree(dm));CUDA_CHECK(cudaFree(dl));CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(ddQ));CUDA_CHECK(cudaFree(ddK));CUDA_CHECK(cudaFree(ddV));
    free(hQ);free(hK);free(hV);free(hO_cpu);free(hdO);
    free(hdQ_cpu);free(hdK_cpu);free(hdV_cpu);
    free(hdQ_gpu);free(hdK_gpu);free(hdV_gpu);
    free(hO_gpu);free(hm);free(hl);free(hD);
    return 0;
}

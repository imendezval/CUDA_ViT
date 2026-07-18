#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include <math_constants.h>


namespace {

constexpr int kThreads = 256;
constexpr int warps_per_block = 8;
constexpr int warp_size = kThreads / warps_per_block; // 32, warp size

constexpr int Br = 16;
constexpr int Bc = 32;
constexpr int KHeadDim = 64;
constexpr int rounds = Br / warps_per_block;  // 2

__global__ void FlashAttention_kernel(
    const float* __restrict__ Q,    // [B, H, T, Dh]    
    const float* __restrict__ K,    // [B, H, T, Dh]
    const float* __restrict__ V,    // [B, H, T, Dh]
    float* __restrict__ out,        // [B, H, T, Dh]
    int num_tokens
) {

// FlashAttention: theoretically the best advantage is not matrializing attention scores [T * T]
// in reality however we already accomplish this with our fused attention kernel:
// it doesnt write + read attention scores twice for softmax + scores @ V

// So what are the actual DIFFERENCES:

// DATA REUSE, with an analogy to patch embedding
// 1 block = 1 patch + one weight dim -> no data reuse
// 1 block = 1 patch + all weight dims -> enables patch data reuse (w/ shared memory)
// 1 block = 1+ patches + all weight dims -> enables data reuse in second dim - embedding weight reuse

// So vs a fused attention kernel, FlashAttention uses Q tiles, aka multiple Q rows, so we reuse K and V data
// This is essential. 
// It differs a lot from patch embedding because it is more complex since attention is multiple operations combined.
// Therefore eg online softmax, accumulated outputs...

// SHARED MEMORY
// In a ViT, we dont necessarily save on shared memory by not materializing the T long attention scores row, 
// since we have to store Q, K and V tiles there (idally m and l for online softmax + accumulated outputs in registers).
// In an LLM, T (= num_tokens) can be MUCH larger, so there we could save on memory.

// LOOP OVERHEAD. Per Q tile, load K/V, sync, compute, softmax update, sync
// With individual K V rows, we do this T times. With 32 row tiles, we do this T / 32 times
// thus saving time in sync, address calculation, shared memory staging

// DATA LOADING into shared memory. Loading a whole tile instead of a single row
// = better memory coalescing + fuller use of block
// + many tiny load phases are less efficient than fewer larger load phases

// PARALLELISM
// Less parallelism, since less blocks (myb more parallelism inside block but less global parallelism)
// We trade less parallelism for more useful work + reuse in each block

// MATRIX-MATRIX vs MATRIX-VECTOR - tensor core matrix multiplication
// For us blockwise the bigger effect is data reuse since we design simple kernels that do matrix-matrix by iterating rows
// however, GPU tensor cores way more efficient on matrix-matrix, 
// with tensor cores operating on small matrix fragments in GEMM style (too difficult to implement for now)
// (each warp doing : [Bp, P] @ [P, Be] → [Bp, Be])
// instead of row at a time implementation on CUDA cores.

const int64_t block = static_cast<int64_t>(blockIdx.x);

const float scale = 1.0f / sqrtf(static_cast<float>(KHeadDim));

// Pointer offsets
const int tiles_per_Q =
    (num_tokens + Br - 1) / Br;
const int tiles_per_KV =
    (num_tokens + Bc - 1) / Bc;
const int64_t BH_idx = block / tiles_per_Q;
const int Q_tile_idx = block % tiles_per_Q;

const float* Q_tile_start = Q 
    + BH_idx * num_tokens * KHeadDim 
    + Q_tile_idx * Br * KHeadDim;
const float* K_start = K + BH_idx * num_tokens * KHeadDim;
const float* V_start = V + BH_idx * num_tokens * KHeadDim;
float* out_start = out + BH_idx * num_tokens * KHeadDim;

const int Q_tile_size = Br * KHeadDim;
const int KV_tile_size = Bc * KHeadDim;

// Prep Shared Memory
extern __shared__ float shared[];
float* Q_shared = shared;                   // [Br, Dh]
float* K_shared = Q_shared + Q_tile_size;   // [Bc, Dh]
float* V_shared = K_shared + KV_tile_size;  // [Bc, Dh]

// persistent in registers:
// m + l [Br] for online softmax
// output accumulators [Br, Dh] fragments
// Q tile fragments

// PER WARP:
// 1 Q row + Bc K rows -> Bc attention scores
// Dh output accumulators

// PER LANE:
// n = Bc / warp_size
// m = Dh / warp_size
// TEMP: n K rows -> n attention scores
// PERMANENT (per Q row): m output accumulators

// all * Q row rounds

const int tid = static_cast<int>(threadIdx.x);

const int group_id = tid / warp_size;
const int lane_id = tid % warp_size;

// Running max + exp sum for online softmax
float m[rounds];
float l[rounds];

// Output accumulator
float out0[rounds];
float out1[rounds];

#pragma unroll
for (int r = 0; r < rounds; ++r) {
    m[r] = -CUDART_INF_F;
    l[r] = 0.0f;
    out0[r] = 0.0f;
    out1[r] = 0.0f;
}

// ==================== Load Q tile ====================
for (int col = tid; col < Q_tile_size; col += blockDim.x) {
    Q_shared[col] = Q_tile_start[col];
}
__syncthreads();

// Warps handle 1 Q row of tile + same K V tile at a time
// Outer loop: iterate over K + V tiles: load, sync
// Inner loop: warps iterate over Q rows: compute scores, update softmax, update accumulator, sync?
for (int KV_tile_idx = 0; KV_tile_idx < tiles_per_KV; KV_tile_idx++) {
    
    // ==================== Load KV tiles ====================
    for (int idx = tid; idx < KV_tile_size; idx += blockDim.x) {
        // save K transposed for QK memory coalescing
        int row = idx / KHeadDim;
        int col = idx % KHeadDim;
        // read global memory coalesced + index shared mem transposed
        K_shared[col * Bc + row] = K_start[KV_tile_idx * KV_tile_size + idx];
        
        // save V normal for P @ V memory coalescing
        V_shared[idx] = V_start[KV_tile_idx * KV_tile_size + idx];
    }
    __syncthreads();

    #pragma unroll
    for (int round = 0; round < rounds; round++) {
    
        const int Q_tile_row_idx = group_id + round * warps_per_block;
        const int global_Q_row = Q_tile_idx * Br + Q_tile_row_idx;
    
        bool active = (Q_tile_row_idx < Br) && (global_Q_row < num_tokens);

        if (active) {
            // ==================== Q @ K rows ====================
            //for (int col = lane_id; col < KV_tile_size; col += group_size)
            
            // we are computing Q_row of tile * Bc rows of K tile
            // each K row needs its own accumulator
            // to avoid using shared memory, 2 simple options:
            // (eg for Bc = 64, warp size = 32)
            
            // option 1:
            // each lane owns n = 2 (= Bc / warp size) K rows
            // and has 2 accumulators + no reduction needed
            // K stored transposed, so all lanes read adjacent memory
            // because we iterate over Dh dim, with all lanes seeing all Dh idx simultaneously
            
            // option 2: split warp into subgroups, each subgroup one Q row
            // more data reuse but more register pressure
            
            // we can never divide K rows across lanes
            // 1 K row = 1 att score. 1 lane = 1 att score accumulator in register
            // that would require shared memory reduction / many more registers
            
            float local_score = 0.0f;

            for (int Dh_idx = 0; Dh_idx < KHeadDim; Dh_idx++) {
                const int Q_el_idx = Q_tile_row_idx * KHeadDim + Dh_idx;
                const int K_el_idx = Dh_idx * Bc + lane_id;

                // Q val broadcasted across all warps
                local_score += Q_shared[Q_el_idx] * K_shared[K_el_idx];
                // each warp has now in registers [Bc] attention scores
            }
            local_score *= scale;
            
            // ======== Current tile max ========
            // get local max score if Bc / Warp size > 1
            // float tile_max = fmaxf(local_score0, local_score1)
            float tile_max = local_score;

            // Warp shuffle max reduction
            for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
                tile_max = fmaxf(
                    tile_max, 
                    __shfl_down_sync(0xffffffffu, tile_max, offset)
                );
            }
            // broadcast lane 0 to all lanes
            tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
            
            // ======== Merge previous max ========
            const float m_new = fmaxf(m[round], tile_max);
            const float alpha = expf(m[round] - m_new);
            
            // ======== Compute exp ========
            const float p = expf(local_score - m_new);

            // ======== Update denominator ========
            float tile_sum = p;
            for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
                tile_sum += __shfl_down_sync(
                    0xffffffffu,
                    tile_sum,
                    offset
                );
            }
            tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);

            float l_new = alpha * l[round] + tile_sum;

            // ======== Rescale prev out ========
            out0[round] *= alpha;
            out1[round] *= alpha;
            
            // ======== P @ V ========
            // we can never divide Vs Dh dim across lanes
            // 1 Dh dim = 1 output element
            // Dh dims must live in same lane for output accumulator in lane register
            for (int att_score_idx = 0; att_score_idx < 32; att_score_idx++) {
                const float p_broadcast = 
                    __shfl_sync(0xffffffff, p, att_score_idx);
                out0[round] += p_broadcast * V_shared[att_score_idx * KHeadDim + lane_id];
                out1[round] += p_broadcast * V_shared[att_score_idx * KHeadDim + lane_id + 32];
            }
            
            // ======== Update state ========
            m[round] = m_new;
            l[round] = l_new;
        }

    }
    __syncthreads(); // shared KV tile overwriting
}

#pragma unroll
for (int round = 0; round < rounds; ++round) {
    const int Q_tile_row_idx =
        group_id + round * warps_per_block;

    const int global_Q_row =
        Q_tile_idx * Br + Q_tile_row_idx;

    if (global_Q_row < num_tokens) {
        const float inv_l = 1.0f / l[round];

        float* output_row =
            out_start + global_Q_row * KHeadDim;

        output_row[lane_id] =
            out0[round] * inv_l;

        output_row[lane_id + 32] =
            out1[round] * inv_l;
    }
}

}

}


torch::Tensor FlashAttention_cuda(
    torch::Tensor Q,   // [B, H, T, Dh]
    torch::Tensor K,   // [B, H, T, Dh]
    torch::Tensor V    // [B, H, T, Dh]
) {
    // ----- Device checks -----
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    TORCH_CHECK(
        Q.device() == K.device() &&
        Q.device() == V.device(),
        "Q, K, and V must be on the same CUDA device"
    );

    // ----- Data type checks -----
    TORCH_CHECK(
        Q.scalar_type() == torch::kFloat32,
        "Q must have dtype float32"
    );

    TORCH_CHECK(
        K.scalar_type() == torch::kFloat32,
        "K must have dtype float32"
    );

    TORCH_CHECK(
        V.scalar_type() == torch::kFloat32,
        "V must have dtype float32"
    );

    // ----- Shape checks -----
    TORCH_CHECK(Q.dim() == 4,
        "Q must have shape [B, H, T, Dh]"
    );
    TORCH_CHECK(K.dim() == 4,
        "K must have shape [B, H, T, Dh]"
    );
    TORCH_CHECK(V.dim() == 4,
        "V must have shape [B, H, T, Dh]"
    );

    TORCH_CHECK(
        Q.size(0) == K.size(0) &&
        Q.size(1) == K.size(1) &&
        Q.size(2) == K.size(2) &&
        Q.size(3) == K.size(3),
        "Q and K must have the same shape [B, H, T, Dh]"
    );

    TORCH_CHECK(
        Q.size(0) == V.size(0) &&
        Q.size(1) == V.size(1) &&
        Q.size(2) == V.size(2) &&
        Q.size(3) == V.size(3),
        "Q and V must have the same shape [B, H, T, Dh]"
    );

    // Pointer arithmetic assumes standard contiguous row-major storage
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    const int64_t B64  = Q.size(0);
    const int64_t H64  = Q.size(1);
    const int64_t T64  = Q.size(2);
    const int64_t Dh64 = Q.size(3);

    TORCH_CHECK(B64  <= std::numeric_limits<int>::max(), "B too large");
    TORCH_CHECK(H64  <= std::numeric_limits<int>::max(), "H too large");
    TORCH_CHECK(T64  <= std::numeric_limits<int>::max(), "num_tokens too large");
    TORCH_CHECK(Dh64 <= std::numeric_limits<int>::max(), "head_dim too large");

    const int B = static_cast<int>(B64);
    const int H = static_cast<int>(H64);
    const int num_tokens = static_cast<int>(T64);
    const int head_dim = static_cast<int>(Dh64);

    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK(H > 0, "H must be > 0");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
    TORCH_CHECK(head_dim > 0, "head_dim must be > 0");

    TORCH_CHECK(
        num_tokens % Br == 0,
        "num_tokens must currently be divisible by Br, partial tile handling not yet implemented"
    );
    TORCH_CHECK(
        num_tokens % Bc == 0,
        "num_tokens must currently be divisible by Bc, partial tile handling not yet implemented"
    );
    TORCH_CHECK(
        Dh64 == 64,
        "This FlashAttention kernel currently requires head_dim == 64"
    );

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(Q.device());

    // Allocate output tensor through PyTorch
    auto out = torch::empty_like(Q);

    const int64_t num_blocks =
        static_cast<int64_t>(B) * H * (num_tokens / Br);

    TORCH_CHECK(
        num_blocks <= std::numeric_limits<unsigned int>::max(),
        "Too many attention rows for 1D CUDA grid"
    );

    const size_t shared_bytes = (
            Br * head_dim +
            Bc * head_dim +
            Bc * head_dim
        ) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    static_assert(warp_size == 32, "Each embedding group must be one warp");

    // Launch custom CUDA kernel
    FlashAttention_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        num_tokens
    );

    // Check for launch errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "FlashAttention_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
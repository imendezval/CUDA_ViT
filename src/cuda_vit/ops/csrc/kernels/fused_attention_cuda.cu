#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

__global__ void fused_atention_kernel(
    const float* __restrict__ Q,    // [B, H, T, Dh]    
    const float* __restrict__ K,    // [B, H, T, Dh]
    const float* __restrict__ V,    // [B, H, T, Dh]
    float* __restrict__ out,        // [B, H, T, Dh]
    int num_tokens,
    int head_dim
) {

    // Fused Attention Kernel: the main difference conceptually is that,
    // since we arent going to save the attention scores [B, H, T, T] in mem,
    // we cannot just let one block calculate a single element of the attention scores.
    // Each block has to calculate a whole row of the attention scores ([B, H, idx, :] for one idx).

    // We do scaled dot product -> softmax, (using warps) we get a whole row.
    // that attention scores row then has to be multiplied against all V columns (of that B, H pair)
    // aka [B, H, :, i] for all i = 0: head_dim, producing one row of out [B, H, idx, :].
    // This way we never write the whole [B H T T] attention scores to memory, we only materialize a row in shared mem.

    // For future reference: FlashAttention doesnt even materialize the full attention scores row,
    // it does online softmax instead.

    // Most important notes: first time implementing single blocks that calculate multiple outputs of matrix multiplications
    // before both scaled qk + attention @ V was 1 block = 1 output element. this will be different now
    // also we are storing a num_tokenes size row. so we must allocate shared memory for this.
    // for softmax, we take a row, used shared mem for max and sum reductions. But that shared array is occupied by the num_tokens row
    // so we need a second shared array for reductions.

    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const float* Q_row = Q + block * head_dim;             // 1 block -> 1 Q row -> = one attention score row
    const float* K_bh = K + block * head_dim * num_tokens; // 1 block -> 1 K mat corresponding to a (B,H) pair

    const float* V_bh = V + block * head_dim * num_tokens; // w/ V columns not contigious
    const float* out_row = out + block * head_dim;

    extern __shared__ float shared[];

    float* scores = shared; // num_tokens length
    float* reduction_buffer = scores + static_cast<int64_t>(num_tokens);
}

}
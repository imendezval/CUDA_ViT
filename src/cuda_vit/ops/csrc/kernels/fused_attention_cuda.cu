#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int groups_per_block = 8;
constexpr int group_size = kThreads / groups_per_block;

__global__ void fused_attention_kernel(
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

    // Offset pointers
    const int64_t bh_idx = block / num_tokens;

    const float* Q_row = Q + block * head_dim;              // 1 block -> 1 Q row -> = one attention score row
    const float* K_bh = K + bh_idx * head_dim * num_tokens; // 1 block -> 1 K mat corresponding to a (B,H) pair

    const float* V_bh = V + bh_idx * head_dim * num_tokens; // w/ V columns not contigious
    float* out_row = out + block * head_dim;

    // Prep shared memory
    extern __shared__ float shared[];

    float* scores = shared; // num_tokens length
    float* red_buffer = scores + static_cast<int64_t>(num_tokens);
    
    // Prep groups and lanes
    int group_id = tid / group_size; // each group handles multiple output element aka row * K columns sequentially
    int lane_id = tid % group_size;  // each lane_id handles multiple element wise * sequentially

    // per group reduction buffer
    float* g_red_buffer = red_buffer 
            + static_cast<int64_t>(group_id) * group_size;


    // ==================== Q @ K rows ====================
    // Q row @ all K rows of B, H pair
    int rounds_QK = (num_tokens + groups_per_block - 1) / groups_per_block;

    for (int round = 0; round < rounds_QK; ++round) {
        int out_el = round * groups_per_block + group_id;
        bool active = out_el < num_tokens;
        // equivalent to:
        // for (int out_el = group_id; out_el < num_tokens; out_el += groups_per_block)
        // with all threads reaching all __syncthreads();

        float local_sum = 0.0f;
        
        if (active) {
            for (int col = lane_id; col < head_dim; col += group_size) {
                local_sum += Q_row[col] * K_bh[out_el * head_dim + col];
            }
        }
        
        g_red_buffer[lane_id] = local_sum;
        __syncthreads();
        
        for (int stride = group_size / 2; stride > 0; stride >>= 1) {
            if (lane_id < stride) {
                g_red_buffer[lane_id] += g_red_buffer[lane_id + stride];
            }
            __syncthreads();
        }
        
        if (active && lane_id == 0) {
            scores[out_el] = g_red_buffer[0] / sqrtf(static_cast<float>(head_dim));
        }
        __syncthreads();
    }

    // ==================== Softmax ====================
    // over attention scores row

    // Find MAX for safe softmax exp(x - max_x)
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < num_tokens; col += blockDim.x) {
        local_max = fmaxf(scores[col], local_max);
    }

    red_buffer[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_buffer[tid] = fmaxf(red_buffer[tid], red_buffer[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = red_buffer[0];
    __syncthreads();

    // Calculate SUM
    float local_sum = 0.0f;
    for (int col = tid; col < num_tokens; col += blockDim.x) {
        local_sum += expf(scores[col] - row_max);
    }
    red_buffer[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_buffer[tid] += red_buffer[tid + stride];
        }
        __syncthreads();
    }

    const float denominator = red_buffer[0];

    for (int col = tid; col < num_tokens; col += blockDim.x) {
        scores[col] = expf(scores[col] - row_max) / denominator;
    }
    __syncthreads();
    
    // ==================== Attention scores @ V columns ====================
    // Attention Scores row @ all V columns of B, H pair

    int rounds_AV = (head_dim + groups_per_block - 1) / groups_per_block;

    for (int round = 0; round < rounds_AV; round++) {
        int out_el = group_id + round * groups_per_block;
        bool active = out_el < head_dim;

        float local_sum = 0.0f; 

        if (active) {
            for (int col = lane_id; col < num_tokens; col += group_size) {
                local_sum += scores[col] * V_bh[out_el + col * head_dim];
            }
        }

        g_red_buffer[lane_id] = local_sum;
        __syncthreads();

        for (int stride = group_size / 2; stride > 0; stride >>= 1) {
            if (lane_id < stride) {
                g_red_buffer[lane_id] += g_red_buffer[lane_id + stride];
            }
            __syncthreads();
        }

        if (active && lane_id == 0) {
            out_row[out_el] = g_red_buffer[0];
        }

        __syncthreads();
    }
}

}


torch::Tensor FusedAttention_cuda(
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

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(Q.device());

    // Allocate output tensor through PyTorch
    auto out = torch::empty_like(Q);

    const int64_t num_blocks =
        static_cast<int64_t>(B) * H * num_tokens;

    TORCH_CHECK(
        num_blocks <= std::numeric_limits<unsigned int>::max(),
        "Too many attention rows for 1D CUDA grid"
    );

    const size_t shared_bytes =
        (static_cast<size_t>(num_tokens) + kThreads) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    // Launch custom CUDA kernel
    fused_attention_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        num_tokens,
        head_dim
    );

    // Check for launch errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "fused_attention_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
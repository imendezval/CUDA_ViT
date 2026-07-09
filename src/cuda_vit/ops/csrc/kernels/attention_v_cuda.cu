#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

__global__ void attention_V_kernel(
    const float* __restrict__ att_scores,   // [B, H, T, T]
    const float* __restrict__ V,            // [B, H, T, Dh]
    float* __restrict__ out,                // [B, H, T, Dh]
    int num_tokens,
    int head_dim
) {
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    // 1 block: 1 att_score query token row * one V Dh column
    // first att_score [B, H, 0, :] * all i in V [B, H, :, i = 0:head_dim], then att_score [B, H, 1, :]...

    const int64_t Dh_idx = block % head_dim;
    const int64_t att_score_row_idx = block / head_dim;
    const int64_t bh_idx = block / (num_tokens * head_dim);

    const float* att_scores_row = att_scores + att_score_row_idx * num_tokens;
    // V columns not contigious - taking for one head_dim all tokens
    // get start and let tid select specific elements
    const float* V_start = V + Dh_idx + bh_idx * num_tokens * head_dim; // could simplify to * block

    float* out_el = out + block;

    extern __shared__ float shared_sums[];
    float local_sum = 0.0f;

    // NOT MEMORY COALESCED: neighboring threads in warp not accessing nearby addresses 
    for (int col = tid; col < num_tokens; col += blockDim.x) {
        local_sum += V_start[col * head_dim] * att_scores_row[col]; // threads select for head_dim element corresponding to each token
    }
    shared_sums[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_el[0] = shared_sums[0];
    }
}

}


torch::Tensor attention_V_cuda( // assumes T_query == T_key
                torch::Tensor att_scores,   // [B, H, T, T]    
                torch::Tensor V             // [B, H, T, Dh]    
) {
    // ----- Device checks -----
    TORCH_CHECK(att_scores.is_cuda(), "Attention Scores must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    TORCH_CHECK(
        att_scores.device() == V.device(),
        "Attention Scores and V must be on the same CUDA device"
    );

    // ----- Data type checks -----

    TORCH_CHECK(
        att_scores.scalar_type() == torch::kFloat32,
        "Attention Scores must have dtype float32"
    );

    TORCH_CHECK(
        V.scalar_type() == torch::kFloat32,
        "V must have dtype float32"
    );

    // ----- Shape checks -----

    TORCH_CHECK(att_scores.dim() == 4, 
        "Attention Scores must have shape [B, H, T, T]"
    );
    TORCH_CHECK(V.dim() == 4,
        "V must have shape [B, H, T, Dh]"
    );

    TORCH_CHECK(
        att_scores.size(2) == att_scores.size(3),
        "Attention Scores' last 2 dimensions must be equal (num_tokens)"
    );

    TORCH_CHECK(
        att_scores.size(0) == V.size(0) &&
        att_scores.size(1) == V.size(1) &&
        att_scores.size(2) == V.size(2),
        "Attention Scores and V must match in dimensions 0, 1, and 2"
    );

    // Pointer arithmetic assumes standard contiguous row-major storage
    TORCH_CHECK(att_scores.is_contiguous(), "Attention Scores must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    const int64_t B64 = V.size(0);
    const int64_t H64 = V.size(1);
    const int64_t T64 = V.size(2);
    const int64_t Dh64 = V.size(3);

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
    c10::cuda::CUDAGuard device_guard(V.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty_like(V);

    // Launch config
    const int64_t num_blocks = static_cast<int64_t>(B) * H * num_tokens * head_dim;
    const size_t shared_bytes = kThreads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(V.get_device());

    TORCH_CHECK(num_blocks <= std::numeric_limits<unsigned int>::max(), "Too many outputs for 1D CUDA grid");

    // Launch custom CUDA kernel
    attention_V_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        att_scores.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        num_tokens,
        head_dim
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "attention_V_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
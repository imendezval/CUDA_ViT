#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>


namespace {

constexpr int kThreads = 256;

__global__ void scaled_QK_kernel(
    const float* __restrict__ Q,    // [B, H, T, Dh]    
    const float* __restrict__ K,    // [B, H, T, Dh]
    float* __restrict__ out,        // [B, H, T, T] (order TQ, TK)
    int num_tokens,
    int head_dim
) {
    // Scaled QK
    // Every block: one Q row with one K row
    // first Q: [B, H, 0, Dh] with all K rows, then Q(T = 1)...
    // equivalent to Q @ K.permute(0, 1, 3, 2)
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const int64_t key_idx = block % num_tokens;
    const int64_t q_row_idx = block / num_tokens; // idx all Q rows flattened
    const int64_t bh_idx = block / (num_tokens * num_tokens);

    const float* row_Q = Q + q_row_idx * head_dim;
    const float* row_K = K + (key_idx * head_dim) + (bh_idx * num_tokens * head_dim);
    float* out_el = out + block;

    extern __shared__ float shared_sums[];
    float local_sum = 0.0f;

    for (int col = tid; col < head_dim; col += blockDim.x) {
        local_sum += row_Q[col] * row_K[col];
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
        out_el[0] = shared_sums[0] / sqrtf(float(head_dim));
    }
}

}


torch::Tensor scaled_QK_cuda(
                torch::Tensor Q,   // [B, H, T, Dh]    
                torch::Tensor K    // [B, H, T, Dh]    
) {

    // ----- Device checks -----
    TORCH_CHECK(Q.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "W must be a CUDA tensor");

    TORCH_CHECK(
        Q.device() == K.device(),
        "Q and K must be on the same CUDA device"
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

    // ----- Shape checks -----

    TORCH_CHECK(Q.dim() == 4, 
        "Q must have shape [B, H, T, Dh]"
    );
    TORCH_CHECK(K.dim() == 4,
        "K must have shape [B, H, T, Dh]"
    );

    TORCH_CHECK(
        Q.sizes() == K.sizes(),
        "Q and K must have exactly the same shape for all dimensions"
    );

    // Pointer arithmetic assumes standard contiguous row-major storage
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");

    const int64_t B64 = Q.size(0);
    const int64_t H64 = Q.size(1);
    const int64_t T64 = Q.size(2);
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

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty(
        {B, H, num_tokens, num_tokens},
        Q.options()
    );

    // Launch config
    const int64_t num_blocks = B64 * H * num_tokens * num_tokens;
    const size_t shared_bytes = kThreads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    TORCH_CHECK(num_blocks <= std::numeric_limits<unsigned int>::max(), "Too many outputs for 1D CUDA grid");

    // Launch custom CUDA kernel
    scaled_QK_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        Q.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        K.data_ptr<float>(),
        out.data_ptr<float>(),
        num_tokens,
        head_dim
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "scaled_QK_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
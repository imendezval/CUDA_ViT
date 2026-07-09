#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

__global__ void LayerNorm_kernel(
    const float* __restrict__ x,      // just pointer to [0, 0, 0] of [B, T, D]
    const float* __restrict__ gamma,  // D
    const float* __restrict__ beta,   // D
    float* __restrict__ out,
    int emb_size,   // D
    int rows, // num rows, B * T
    float eps
) {
    // Define Row from blockIdx w/ one Block per Row
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    if (row >= rows) {
        return;
    }

    // Advance pointers to row start in flat memory
    const float* row_x = x + row * emb_size;
    float* row_out = out + row * emb_size;

    // Prepare shared memory
    extern __shared__ float shared_sums[];

    // MEAN
    // compute local sums per thread
    float local_sum = 0.0f;

    for (int64_t col = tid; col < emb_size; col += blockDim.x) {
        local_sum += row_x[col];
    }

    // Write local sums into shared memory
    shared_sums[tid] = local_sum;
    __syncthreads();

    // Block wise reduction into shared_sums[0]. Parallelized sum()
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    const float mean = shared_sums[0] / static_cast<float>(emb_size);
    __syncthreads(); // all threads read mean before reusing shared mem

    // STD
    float local_sq_sum = 0.0f;

    for(int64_t col = tid; col < emb_size; col += blockDim.x) {
        const float diff = row_x[col] - mean;
        local_sq_sum += diff * diff;
    }

    shared_sums[tid] = local_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    const float variance = shared_sums[0] / static_cast<float>(emb_size);
    const float inv_std = rsqrt(variance + eps);

    // Normalize + write
    for (int64_t col = tid; col < emb_size; col += blockDim.x) {
        const float layernorm = gamma[col] * (
            (row_x[col] - mean) * inv_std
        ) + beta[col];
    
        row_out[col] = layernorm;
    }
}

}  // namespace

torch::Tensor LayerNorm_cuda(
                torch::Tensor x, 
                torch::Tensor gamma,
                torch::Tensor beta,
                double eps
) {
    // Tensor checks
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32,
                "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32,
                "beta must be float32");
    TORCH_CHECK(x.dim() >= 1,
                "x must have at least one dimension");
    TORCH_CHECK(x.is_contiguous(),
                "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(),
                "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(),
                "beta must be contiguous");

    const int64_t E64 = x.size(-1);         // D
    const int64_t num_x = x.numel();

    TORCH_CHECK(num_x  <= std::numeric_limits<int>::max(), "x too large");
    TORCH_CHECK(E64  <= std::numeric_limits<int>::max(), "embedding size too large");

    const int emb_size = static_cast<int>(E64);
    const int rows = num_x / emb_size;  // B * T


    TORCH_CHECK(emb_size > 0,
                "last dimension must be non-empty");

    TORCH_CHECK(gamma.dim() == 1,
                "gamma must have shape [D]");
    TORCH_CHECK(beta.dim() == 1,
                "beta must have shape [D]");

    TORCH_CHECK(gamma.numel() == emb_size,
                "gamma must have D elements");
    TORCH_CHECK(beta.numel() == emb_size,
                "beta must have D elements");

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(x.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty_like(x);

    // Launch config
    const int64_t blocks = rows;
    const size_t shared_bytes = kThreads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    TORCH_CHECK(blocks <= std::numeric_limits<unsigned int>::max(), "Too many outputs for 1D CUDA grid");

    // Launch custom CUDA kernel
    LayerNorm_kernel<<<blocks, kThreads, shared_bytes, stream>>>(
        x.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        emb_size,
        rows,
        static_cast<float>(eps)
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "vector_add_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out; // Return output
}
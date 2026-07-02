#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

}  // namespace

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // Tensor checks
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same GPU");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(a.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty_like(a);
    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    // Launch config
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

    // Launch custom CUDA kernel
    vector_add_kernel<<<blocks, kThreads, 0, stream>>>(
        a.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
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

/*
Python wrapper
    ↓
C++ binding
    ↓
CUDA launcher function
    ↓
__global__ CUDA kernel(s)
*/
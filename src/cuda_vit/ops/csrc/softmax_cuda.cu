#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <math_constants.h>


namespace {

constexpr int kThreads = 256;

__global__ void SoftMax_kernel(
    const float* __restrict__ x,    // B, F
    float* __restrict__ out,        // B, F
    int num_features
) {
    // 1 block = softmax across 1 batch
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    // Offset pointers
    const float* row_x = x + block * num_features;
    float* row_out = out + block * num_features;

    // Prep shared memory
    extern __shared__ float shared[];

    // Find MAX for safe softmax exp(x - max_x)
    float local_max = -CUDART_INF_F;

    for (int col = tid; col < num_features; col += blockDim.x) {
        const float curr = row_x[col];
        if (curr > local_max) { // or local_max = fmaxf(local_max, row_x[col]);
            local_max = curr;
        }
    }
    shared[tid] = local_max;
    __syncthreads();

    // MAX Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared[tid + stride] > shared[tid]) { // or shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
                shared[tid] = shared[tid + stride]; 
            }
        }
        __syncthreads();
    }

    const float row_max = shared[0];
    __syncthreads();

    // Calculate SUM for denominator
    float local_sum = 0.0f;

    for (int col = tid; col < num_features; col += blockDim.x) {
        local_sum += expf(row_x[col] - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();
    
    // SUM Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    const float denominator = shared[0];
    __syncthreads();
    
    // Write output
    for (int col = tid; col < num_features; col += blockDim.x) {
        row_out[col] = expf(row_x[col] - row_max) / denominator; 
    }
}

}


torch::Tensor SoftMax_cuda(
                torch::Tensor x // B, in_features
                // For ViT attention scores [B, H, T, T] -> [B * H * T, num_features = T]
) {

    // ----- Device checks -----
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    // ----- Data type checks -----

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "x must have dtype float32"
    );

    // ----- Shape checks -----

    TORCH_CHECK(x.dim() == 2, "x must have shape [B, num_features]");


    // Pointer arithmetic assumes standard contiguous row-major storage
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    // PyTorch sizes come as int64 conventionally
    const int B = static_cast<int>(x.size(0));
    const int num_features = static_cast<int>(x.size(1));

    TORCH_CHECK(num_features > 0, "num_features must be > 0");

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(x.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty_like(x);

    // Launch config
    const int64_t num_blocks = static_cast<int64_t>(B);
    const size_t shared_bytes = kThreads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    // ideally in a library we would check here also if num_blocks is allowed by 1D grid

    // Launch custom CUDA kernel
    SoftMax_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        x.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        out.data_ptr<float>(),
        num_features
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "softmax_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
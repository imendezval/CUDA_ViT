#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>


namespace {

/*
PyTorch W convention makes each out_features weight vector a contiguous row in memory
Mathematically, x @ W.T, but CUDA takes non transposed matrix in memory
therefore works for FC, QKV projections, W_o output projection
but NOT for: 
Scaled_QK since second tensor K has a weight mat per B, H pair
Attention @ V, since mathematically no V.T, so V columns not stored consecutively in memory
*/
    
constexpr int kThreads = 256;

__global__ void linear_forward_kernel(
    const float* __restrict__ x,    // B, F (= in_features)
    const float* __restrict__ W,    // emb_size (= out_features), F (= in_features)
    float* __restrict__ out,        // B, emb_size
    int in_features,
    int out_features
) {
    // Per blockIdx.x: 1 x batch @ 1 embedding dim
    // first x [0, :] with W [i, :] for all i, then x [1, :] ...
    const int64_t block = static_cast<int64_t>(blockIdx.x);

    const float* row_w = W + (block % out_features) * in_features;
    const float* row_x = x + (block / out_features) * in_features; // batch_idx
    float* out_el = out + block;

    const int tid = static_cast<int>(threadIdx.x);

    extern __shared__ float shared_sums[];
    float local_sum = 0.0f;

    for (int col = tid; col < in_features; col += blockDim.x) {
        local_sum += row_w[col] * row_x[col];
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

} // namespace


torch::Tensor linear_forward_cuda(
                torch::Tensor x,    // B, in_features. In a ViT convert [B, T, emb_size] -> [B * T, emb_size]
                torch::Tensor W     // out_features, in_features
) {

    // ----- Device checks -----
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");

    TORCH_CHECK(
        x.device() == W.device(),
        "x and W must be on the same CUDA device"
    );

    // ----- Data type checks -----

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "x must have dtype float32"
    );

    TORCH_CHECK(
        W.scalar_type() == torch::kFloat32,
        "W must have dtype float32"
    );

    // ----- Shape checks -----

    TORCH_CHECK(x.dim() == 2, "x must have shape [B, in_features]");
    TORCH_CHECK(
        W.dim() == 2,
        "W must have shape [out_features, in_features]"
    );

    TORCH_CHECK(
        x.size(1) == W.size(1),
        "x.size(1) must equal W.size(1): expected matching in_features"
    );

    // Pointer arithmetic assumes standard contiguous row-major storage
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");

    // PyTorch sizes come as int64 conventionally
    const int B = static_cast<int>(x.size(0));
    const int in_features = static_cast<int>(x.size(1));
    const int out_features = static_cast<int>(W.size(0));

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(x.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty(
        {B, out_features},
        x.options()
    );

    if (B == 0 || out_features == 0) {
        return out;
    }

    // Launch config
    const int64_t num_blocks = static_cast<int64_t>(B) * out_features;
    const size_t shared_bytes = kThreads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    // ideally in a library we would check here also if num_blocks is allowed by 1D grid

    // Launch custom CUDA kernel
    linear_forward_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        x.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        W.data_ptr<float>(),
        out.data_ptr<float>(),
        in_features,
        out_features
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "linear_forward_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
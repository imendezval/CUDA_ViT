#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>


__device__ __forceinline__
float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.79788456080f;  // sqrt(2 / pi)
    constexpr float kCoeff = 0.044715f;

    float x3 = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x3);

    return 0.5f * x * (1.0f + tanhf(inner));
}


namespace {
    
constexpr int kThreads = 256;

__global__ void fused_mlp_linear_gelu_kernel(
    const float* __restrict__ x,    // B, F (= in_features)
    const float* __restrict__ W,    // emb_size (= out_features), F (= in_features)
    const float* __restrict__ b,    // emb_size (= out_features)
    // PyTorch W convention makes each output neurons weight vector a contiguous row :)
    float* __restrict__ out,        // B, emb_size
    int in_features,
    int out_features
) {
    // Per blockIdx.x: 1 x batch @ 1 embedding dim
    // first in_features(B = 0) with @ W_0, then same w/ @ W_1...
    // all W with batch 0 first, then with batch 1...
    const int64_t block = static_cast<int64_t>(blockIdx.x);

    // Pointer offsets safer with int64
    const float* row_w = W + (block % out_features) * in_features;
    const float* bias  = b + (block % out_features); // out_feature
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

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float z = shared_sums[0] + bias[0];
        out_el[0] = gelu_tanh(z);
    }
}

} // namespace


torch::Tensor fused_MLPlinear_GELU_cuda(
                torch::Tensor x,    // B, in_features. In a ViT convert [B, T, emb_size] -> [B * T, emb_size]
                torch::Tensor W,    // out_features, in_features
                torch::Tensor b     // out_features
) {

    // ----- Device checks -----
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");

    TORCH_CHECK(
        x.device() == W.device() && x.device() == b.device(),
        "x, W, and b must be on the same CUDA device"
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

    TORCH_CHECK(
        b.scalar_type() == torch::kFloat32,
        "b must have dtype float32"
    );

    // ----- Shape checks -----

    TORCH_CHECK(x.dim() == 2, "x must have shape [B, in_features]");
    TORCH_CHECK(
        W.dim() == 2,
        "W must have shape [out_features, in_features]"
    );
    TORCH_CHECK(
        b.dim() == 1,
        "b must have shape [out_features]"
    );

    TORCH_CHECK(
        x.size(1) == W.size(1),
        "x.size(1) must equal W.size(1): expected matching in_features"
    );

    TORCH_CHECK(
        W.size(0) == b.size(0),
        "W.size(0) must equal b.size(0): expected matching out_features"
    );

    // Pointer arithmetic assumes standard contiguous row-major storage
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    const int B64 = x.size(0);
    const int I64 = x.size(1);
    const int O64 = W.size(0);

    TORCH_CHECK(B64  <= std::numeric_limits<int>::max(), "B too large");
    TORCH_CHECK(I64  <= std::numeric_limits<int>::max(), "in_features too large");
    TORCH_CHECK(O64  <= std::numeric_limits<int>::max(), "out_features too large");
    
    const int B = static_cast<int>(B64);
    const int in_features = static_cast<int>(I64);
    const int out_features = static_cast<int>(O64);

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(x.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty(
        {B, out_features},
        x.options()
    );

    // Launch config
    const int64_t num_blocks = B64 * out_features;
    const size_t shared_bytes = kThreads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    TORCH_CHECK(num_blocks <= std::numeric_limits<unsigned int>::max(), "Too many outputs for 1D CUDA grid");

    // Launch custom CUDA kernel
    fused_mlp_linear_gelu_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        x.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        in_features,
        out_features
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "mlp_linear_gelu_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
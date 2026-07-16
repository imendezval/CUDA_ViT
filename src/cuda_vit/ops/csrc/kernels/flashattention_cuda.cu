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

__global__ void FlashAttention_kernel(
    const float* __restrict__ Q,    // [B, H, T, Dh]    
    const float* __restrict__ K,    // [B, H, T, Dh]
    const float* __restrict__ V,    // [B, H, T, Dh]
    float* __restrict__ out,        // [B, H, T, Dh]
    int num_tokens,
    int head_dim
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

    const size_t shared_bytes = // warp shuffles but still need mem for softmax max + sum reductions
        (static_cast<size_t>(num_tokens) + kThreads) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.get_device());

    static_assert(group_size == 32, "Each embedding group must be one warp");

    // Launch custom CUDA kernel
    FlashAttention_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
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
        "FlashAttention_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
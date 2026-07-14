#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include <cmath>


namespace {

constexpr int kThreads = 256;
constexpr int groups_per_block = 8;
constexpr int group_size = kThreads / groups_per_block;

__global__ void PatchEmbeddingV2_kernel(
    const float* __restrict__ x,    // B, C, H, W
    const float* __restrict__ W,    // emb_dimensions = num_kernels, num_patch_el
    float* __restrict__ out,        // B, T_num = num_patches, emb_dims = num_kernels
    int patch_size,
    int emb_size,
    int num_patches_h,
    int num_patches_w,
    int C_img,
    int H_img,
    int W_img
) {
    // Each block: 1 patch @ N emb_dims = N out instances
    // V2 big speedup, avoids different blocks loading up the same patch
    const int64_t block = static_cast<int64_t>(blockIdx.x);

    const int64_t num_patches =
        static_cast<int64_t>(num_patches_h) * num_patches_w;

    const int64_t batch_idx = block / num_patches;
    const int patch_idx = block % num_patches;

    const int num_patch_el = patch_size * patch_size * C_img;

    // Advance pointers
    float* out_row = out + block * emb_size;

    // Patch grid pos
    const int patch_x = patch_idx % num_patches_w;
    const int patch_y = patch_idx / num_patches_w;

    // Top left pixel of patch
    const int start_x = patch_x * patch_size;
    const int start_y = patch_y * patch_size;
    
    // Each thread -> multiple pixels
    const int tid = static_cast<int>(threadIdx.x);

    const int group_id = tid / group_size;
    const int lane_id = tid % group_size;

    // Prep shared memory
    extern __shared__ float shared[];

    float* patch_shared = shared;
    float* reduction_buffer = shared + num_patch_el;

    float* group_reduction = reduction_buffer + group_id * group_size;

    // Bring patch into shared memory to reuse retrieving only once
    for (int col = tid; col < num_patch_el; col += blockDim.x) {
        // mapping of pixel index as row vector -> pixel memory address in memory
        const int c = col / (patch_size * patch_size);
        const int rem = col % (patch_size * patch_size);
        
        // patch pixel to image pixel 
        const int local_x = rem % patch_size;
        const int local_y = rem / patch_size;

        const int image_x = start_x + local_x;
        const int image_y = start_y + local_y;

        // Thread -> pixel index
        const int64_t x_idx = ((batch_idx * C_img + c) * H_img + image_y) * W_img + image_x;
        
        // Main calculation
        patch_shared[col] = x[x_idx];
    }
    __syncthreads();


    // For loop to let groups cover all embedding dimensions
    const int rounds = (emb_size + groups_per_block - 1)/ groups_per_block;

    for (int round = 0; round < rounds; round++) {
        const int out_el = round * groups_per_block + group_id;
        const bool active = out_el < emb_size;
        
        float local_sum = 0.0f;

        if (active) {
            // for loop to let lanes cover all patch pixels
            for (int col = lane_id; col < num_patch_el; col += group_size) {
                // Main calculation
                local_sum += W[static_cast<int64_t>(out_el) * num_patch_el + col] * patch_shared[col];
            }
        }

        group_reduction[lane_id] = local_sum;
        __syncthreads();

        // Per group reduction
        for (int stride = group_size / 2; stride > 0; stride >>= 1) {
            if (lane_id < stride) {
                group_reduction[lane_id] += group_reduction[lane_id + stride];
            }
            __syncthreads();
        }
    
        if (active && lane_id == 0) { // only one thread has to write to output
            out_row[out_el] = group_reduction[0];
        }
        __syncthreads();
    }

}

} // namespace


torch::Tensor PatchEmbeddingV2_cuda(
                torch::Tensor x,    // B, C, H, W
                torch::Tensor W     // Emb_size, num_patch_el
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");

    TORCH_CHECK(x.device() == W.device(),
                "x and W must be on the same CUDA device");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "x must be float32");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32,
                "W must be float32");

    TORCH_CHECK(x.dim() == 4,
                "x must have shape [B, C, H, W]");
    TORCH_CHECK(W.dim() == 2,
                "W must have shape [emb_size, C * patch_size * patch_size]");

    
                TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");

    const int64_t B64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);

    const int64_t emb_size64 = W.size(0);
    const int64_t num_patch_el64 = W.size(-1);

    TORCH_CHECK(B64  <= std::numeric_limits<int>::max(), "B too large");
    TORCH_CHECK(C64  <= std::numeric_limits<int>::max(), "C too large");
    TORCH_CHECK(H64  <= std::numeric_limits<int>::max(), "H too large");
    TORCH_CHECK(W64  <= std::numeric_limits<int>::max(), "W too large");

    TORCH_CHECK(emb_size64  <= std::numeric_limits<int>::max(), "emb_size too large");
    TORCH_CHECK(num_patch_el64  <= std::numeric_limits<int>::max(), "num_patch_el too large");

    const int B_img = static_cast<int>(B64);
    const int C_img = static_cast<int>(C64);
    const int H_img = static_cast<int>(H64);
    const int W_img = static_cast<int>(W64);

    const int emb_size = static_cast<int>(emb_size64);
    const int num_patch_el = static_cast<int>(num_patch_el64);

    TORCH_CHECK(B_img > 0 && C_img > 0 && H_img > 0 && W_img > 0,
                "x dimensions must all be positive");
    TORCH_CHECK(emb_size > 0 && num_patch_el > 0,
                "W dimensions must be positive");

    // W width must equal C * P * P
    TORCH_CHECK(num_patch_el % C_img == 0,
                "W.size(1) must be divisible by image channels C");


    const int patch_area = num_patch_el / C_img;
    const int patch_size = static_cast<int>(std::sqrt(static_cast<double>(patch_area)));

    TORCH_CHECK(
        patch_size > 0 && patch_size * patch_size == patch_area,
        "W.size(1) / C must be a perfect square: expected C * P * P"
    );

    TORCH_CHECK(H_img % patch_size == 0,
                "image height must be divisible by patch_size");
    TORCH_CHECK(W_img % patch_size == 0,
                "image width must be divisible by patch_size");

    const int num_patches_w = W_img / patch_size;
    const int num_patches_h = H_img / patch_size;
    const int num_patches = num_patches_w * num_patches_h;

    TORCH_CHECK(
        B_img <= std::numeric_limits<int>::max() &&
        C_img <= std::numeric_limits<int>::max() &&
        H_img <= std::numeric_limits<int>::max() &&
        W_img <= std::numeric_limits<int>::max() &&
        emb_size <= std::numeric_limits<int>::max() &&
        num_patches_h <= std::numeric_limits<int>::max() &&
        num_patches_w <= std::numeric_limits<int>::max(),
        "tensor dimensions are too large for this kernel's int arguments"
    );

    // Makes the extension respect PyTorch's current GPU/device/stream
    c10::cuda::CUDAGuard device_guard(x.device());

    // Allocate output tensor through pytorch (cudaMalloc)
    auto out = torch::empty(
        {B_img, num_patches, emb_size},
        x.options()
    );

    // Launch config
    const int64_t num_blocks = B64 * num_patches;
    const size_t shared_bytes = (num_patch_el64 + kThreads) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    TORCH_CHECK(num_blocks <= std::numeric_limits<unsigned int>::max(), "Too many outputs for 1D CUDA grid");

    // Launch custom CUDA kernel
    PatchEmbeddingV2_kernel<<<num_blocks, kThreads, shared_bytes, stream>>>(
        x.data_ptr<float>(), // Get raw GPU pointers from tensors (d_a)
        W.data_ptr<float>(),
        out.data_ptr<float>(),
        patch_size,
        emb_size,
        num_patches_h,
        num_patches_w,
        C_img,
        H_img,
        W_img
    );

    // Check for errors
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "PatchEmbeddingV2_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
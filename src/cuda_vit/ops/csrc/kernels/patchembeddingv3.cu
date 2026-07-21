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
constexpr int group_size = kThreads / groups_per_block; // 32 (= 1 warp)

constexpr int patches_per_block = 4;

__global__ void PatchEmbeddingV3_kernel(
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
    // Each block: N patches @ N emb_dims
    // Data reuse along 2 dimensions:
    // patches across embedding dimensions
    // embedding weights across multiple patches
    const int64_t block = static_cast<int64_t>(blockIdx.x);

    const int64_t num_patches =
        static_cast<int64_t>(num_patches_h) * num_patches_w;

    const int64_t patch_blocks_per_batch =
        (num_patches + patches_per_block - 1) / patches_per_block;

    const int64_t batch_idx = block / patch_blocks_per_batch;
    const int64_t patch_block_idx = block % patch_blocks_per_batch;

    const int first_patch_idx =
        static_cast<int>(patch_block_idx) * patches_per_block;

    const int num_patch_el = patch_size * patch_size * C_img;

    // Each thread -> multiple pixels
    const int tid = static_cast<int>(threadIdx.x);

    const int group_id = tid / group_size;
    const int lane_id = tid % group_size;

    // Prep shared mem
    // [patches_per_block * num_patch_el patch vals]
    // [groups_per_block * num_patch_el weight vals]
    extern __shared__ float shared[];

    float* patch_shared = shared;

    float* weight_shared =
        patch_shared + patches_per_block * num_patch_el;

    // Bring multiple patches into shared memory
    for (int patch_round = 0;
         patch_round < patches_per_block;
         patch_round++) {

        const int patch_idx = first_patch_idx + patch_round;
        const bool active_patch = patch_idx < num_patches;

        if (active_patch) {
            // Patch grid pos
            const int patch_x = patch_idx % num_patches_w;
            const int patch_y = patch_idx / num_patches_w;

            // Top left pixel of patch
            const int start_x = patch_x * patch_size;
            const int start_y = patch_y * patch_size;

            float* patch_row =
                patch_shared + patch_round * num_patch_el;

            for (int col = tid;
                 col < num_patch_el;
                 col += blockDim.x) {

                // Map pixel idx as row vector -> pixel address in memory
                const int c = col / (patch_size * patch_size);
                const int rem = col % (patch_size * patch_size);

                // Patch pixel to image pixel
                const int local_x = rem % patch_size;
                const int local_y = rem / patch_size;

                const int image_x = start_x + local_x;
                const int image_y = start_y + local_y;

                // Thread -> pixel index
                const int64_t x_idx =
                    ((batch_idx * C_img + c) * H_img + image_y)
                    * W_img + image_x;

                patch_row[col] = x[x_idx];
            }
        }
    }
    __syncthreads();

    // For loop to let groups cover all emb dims
    const int emb_rounds =
        (emb_size + groups_per_block - 1) / groups_per_block;

    for (int emb_round = 0;
         emb_round < emb_rounds;
         emb_round++) {

        const int out_el =
            emb_round * groups_per_block + group_id;

        const bool active_embedding = out_el < emb_size;

        float* group_weight_shared =
            weight_shared + group_id * num_patch_el;

        // each warp brings one embedding weight row into shared memory
        if (active_embedding) {
            const float* row_w =
                W + static_cast<int64_t>(out_el) * num_patch_el;

            for (int col = lane_id;
                 col < num_patch_el;
                 col += group_size) {

                group_weight_shared[col] = row_w[col];
            }
        }
        __syncthreads();

        // apply the cached weight row to all patches handled by the block
        for (int patch_round = 0;
             patch_round < patches_per_block;
             patch_round++) {

            const int patch_idx = first_patch_idx + patch_round;
            const bool active_patch = patch_idx < num_patches;

            if (active_embedding && active_patch) {
                const float* patch_row =
                    patch_shared + patch_round * num_patch_el;

                float local_sum = 0.0f;

                // For loop to let lanes cover all patch pixels
                for (int col = lane_id;
                     col < num_patch_el;
                     col += group_size) {

                    // Main calculation
                    local_sum +=
                        group_weight_shared[col] * patch_row[col];
                }

                // warp shuffle instead of reduction with shared memory
                for (int offset = group_size / 2;
                     offset > 0;
                     offset >>= 1) {

                    local_sum += __shfl_down_sync(
                        0xffffffff,
                        local_sum,
                        offset
                    );
                }

                if (lane_id == 0) {
                    float* out_row =
                        out
                        + (batch_idx * num_patches + patch_idx)
                        * emb_size;

                    out_row[out_el] = local_sum;
                }
            }
        }

        // sync since shared memory overwritten during next emb round
        __syncthreads();
    }
}

} // namespace


torch::Tensor PatchEmbeddingV3_cuda(
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

    TORCH_CHECK(B64 <= std::numeric_limits<int>::max(), "B too large");
    TORCH_CHECK(C64 <= std::numeric_limits<int>::max(), "C too large");
    TORCH_CHECK(H64 <= std::numeric_limits<int>::max(), "H too large");
    TORCH_CHECK(W64 <= std::numeric_limits<int>::max(), "W too large");

    TORCH_CHECK(emb_size64 <= std::numeric_limits<int>::max(),
                "emb_size too large");
    TORCH_CHECK(num_patch_el64 <= std::numeric_limits<int>::max(),
                "num_patch_el too large");

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

    const int patch_size =
        static_cast<int>(
            std::sqrt(static_cast<double>(patch_area))
        );

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

    const int patch_blocks_per_batch =
        (num_patches + patches_per_block - 1)
        / patches_per_block;

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

    // Allocate output tensor through PyTorch
    auto out = torch::empty(
        {B_img, num_patches, emb_size},
        x.options()
    );

    // Launch config
    const int64_t num_blocks =
        B64 * patch_blocks_per_batch;

    const size_t shared_bytes =
        static_cast<size_t>(
            patches_per_block + groups_per_block
        )
        * num_patch_el64
        * sizeof(float);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(x.get_device());

    static_assert(group_size == 32,
                  "Each embedding group must be one warp");

    TORCH_CHECK(
        num_blocks <= std::numeric_limits<unsigned int>::max(),
        "Too many patch blocks for 1D CUDA grid"
    );

    // Launch custom CUDA kernel
    PatchEmbeddingV3_kernel<<<
        num_blocks,
        kThreads,
        shared_bytes,
        stream
    >>>(
        x.data_ptr<float>(),
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
        "PatchEmbeddingV3_kernel launch failed: ",
        cudaGetErrorString(error)
    );

    return out;
}
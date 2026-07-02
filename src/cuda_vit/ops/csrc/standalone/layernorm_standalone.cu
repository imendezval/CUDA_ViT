#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>


__global__ void LayerNorm(
    const float* __restrict__ x,      // just pointer to [0, 0, 0] of [B, T, D]
    const float* __restrict__ gamma,  // D
    const float* __restrict__ beta,   // D
    float* __restrict__ out,
    int emb_size,   // D
    int hidden_dim, // num rows, B * T
    float eps
) {
    // Define Row from blockIdx w/ one Block per Row
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    if (row >= hidden_dim) {
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


// Small helper so CUDA failures are visible immediately.
void check_cuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": "
                  << cudaGetErrorString(result)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


int main() {
    constexpr int B = 2;
    constexpr int T = 3;
    constexpr int D = 4;

    constexpr int rows = B * T;
    constexpr int n = rows * D;

    constexpr float eps = 1e-5f;

    const size_t tensor_bytes = n * sizeof(float);
    const size_t emb_bytes = D * sizeof(float);

    std::vector<float> h_tensor = {
        // batch 0, token 0
        1.0f, 2.0f, 3.0f, 4.0f,
        // batch 0, token 1
        5.0f, 6.0f, 7.0f, 8.0f,
        // batch 0, token 2
        2.0f, 4.0f, 6.0f, 8.0f,

        // batch 1, token 0
        10.0f, 20.0f, 30.0f, 40.0f,
        // batch 1, token 1
        -1.0f, 0.0f, 1.0f, 2.0f,
        // batch 1, token 2
        3.0f, 3.0f, 3.0f, 3.0f
    };


    std::vector<float> h_gamma = {
        1.0f, 1.5f, 0.5f, 2.0f
    };
    std::vector<float> h_beta = {
        0.0f, 0.1f, -0.2f, 0.3f
    };

    std::vector<float> h_output(n);
    std::vector<float> h_expected(n);

    // CPU reference implementation for correctness checking
    for (int row = 0; row < rows; ++row) {
        const int offset = row * D;

        float sum = 0.0f;
        for (int col = 0; col < D; ++col) {
            sum += h_tensor[offset + col];
        }

        const float mean = sum / static_cast<float>(D);

        float sq_sum = 0.0f;
        for (int col = 0; col < D; ++col) {
            const float diff = h_tensor[offset + col] - mean;
            sq_sum += diff * diff;
        }

        const float variance = sq_sum / static_cast<float>(D);
        const float inv_std = 1.0f / std::sqrt(variance + eps);

        for (int col = 0; col < D; ++col) {
            const float normalized =
                (h_tensor[offset + col] - mean) * inv_std;

            h_expected[offset + col] =
                normalized * h_gamma[col] + h_beta[col];
        }
    }

    // Init GPU side pointers
    float* d_tensor = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;

    // Allocate GPU memory to pointers
    check_cuda(
        cudaMalloc(&d_tensor, tensor_bytes),
        "Failed to allocate d_tensor"
    );
    check_cuda(
        cudaMalloc(&d_gamma, emb_bytes),
        "Failed to allocate d_gamma"
    );
    check_cuda(
        cudaMalloc(&d_beta, emb_bytes),
        "Failed to allocate d_beta"
    );
    check_cuda(
        cudaMalloc(&d_output, tensor_bytes),
        "Failed to allocate d_output"
    );

    // Copy data to GPU side pointers
    check_cuda(
        cudaMemcpy(
            d_tensor,
            h_tensor.data(),
            tensor_bytes,
            cudaMemcpyHostToDevice
        ),
        "Failed to copy h_tensor"
    );

    check_cuda(
        cudaMemcpy(
            d_gamma,
            h_gamma.data(),
            emb_bytes,
            cudaMemcpyHostToDevice
        ),
        "Failed to copy h_gamma"
    );

    check_cuda(
        cudaMemcpy(
            d_beta,
            h_beta.data(),
            emb_bytes,
            cudaMemcpyHostToDevice
        ),
        "Failed to copy h_beta"
    );


    // Launch config
    const int threads_per_block = 256;
    const int blocks_per_grid = rows;

    const size_t shared_bytes = threads_per_block * sizeof(float); 

    LayerNorm<<<blocks_per_grid, threads_per_block, shared_bytes>>> (
        d_tensor, d_gamma, d_beta, d_output, D, rows, eps
    );

    // Check whether launch setup failed.
    check_cuda(cudaGetLastError(), "Kernel launch failed");
    // Make CPU wait for GPU, so runtime errors surface here
    check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");

    check_cuda(
        cudaMemcpy(
            h_output.data(),
            d_output,
            tensor_bytes,
            cudaMemcpyDeviceToHost
        ),
        "Failed to copy d_output to h_output"
    );


    bool passed = true;
    constexpr float tolerance = 1e-4f;

    for (int i = 0; i < n; ++i) {
        const float error = std::fabs(h_output[i] - h_expected[i]);

        if (error > tolerance) {
            std::cerr
                << "Mismatch at index " << i
                << ": GPU = " << h_output[i]
                << ", CPU = " << h_expected[i]
                << ", error = " << error
                << "\n";

            passed = false;
        }
    }

    if (passed) {
        std::cout << "LayerNorm test passed.\n";
    }

    check_cuda(cudaFree(d_tensor), "Failed to free d_tensor");
    check_cuda(cudaFree(d_gamma), "Failed to free d_gamma");
    check_cuda(cudaFree(d_beta), "Failed to free d_beta");
    check_cuda(cudaFree(d_output), "Failed to free d_output");

    d_tensor = nullptr;
    d_beta = nullptr;
    d_gamma = nullptr;
    d_output = nullptr;

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
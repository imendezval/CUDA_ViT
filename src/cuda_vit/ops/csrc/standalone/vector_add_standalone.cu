#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>


__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        c[index] = a[index] + b[index];
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
    // Choose input size + calculate bytes
    const int n = 257;
    const size_t bytes = n * sizeof(float);

    // Init CPU vectors
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    // Fill h_a and h_b
    for(int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i) * 2.0f;
    }

    // Init GPU pointers
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // Allocate GPU memory to pointers
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_a), bytes),
        "Failed to allocate d_a"
    );
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_b), bytes),
        "Failed to allocate d_b"
    );
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&d_c), bytes),
        "Failed to allocate d_c"
    );

    // Copy inputs: CPU -> GPU
    check_cuda(
        cudaMemcpy(
            d_a,
            h_a.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Failed to copy h_a to d_a"
    );
    check_cuda(
        cudaMemcpy(
            d_b,
            h_b.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Failed to copy h_b to d_b"
    );

    // Launch Config
    const int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block; //integer division instead of ceil (n / tpb)


    // Launch Kernel
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_a, d_b, d_c, n
    );

    // Check whether launch setup failed.
    check_cuda(cudaGetLastError(), "Kernel launch failed");

    // Make CPU wait for GPU, so runtime errors surface here
    check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");


    // Copy result: GPU -> CPU
    check_cuda(
        cudaMemcpy(
            h_c.data(),
            d_c,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Failed to copy d_c to h_c"
    );


    // Verify on CPU
    bool passed = true;

    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];

        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_c[i]
                      << ", expected " << expected
                      << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Vector add passed for n = " << n << std::endl;
    }
    
    
    // Clean up GPU memory
    check_cuda(cudaFree(d_a), "Failed to free d_a");
    check_cuda(cudaFree(d_b), "Failed to free d_b");
    check_cuda(cudaFree(d_c), "Failed to free d_c");
    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;
    
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
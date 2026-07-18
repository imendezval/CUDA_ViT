#include <torch/extension.h>

torch::Tensor FlashAttention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "FlashAttention",
        &FlashAttention_cuda,
        "FlashAttention (CUDA)"
    );
}
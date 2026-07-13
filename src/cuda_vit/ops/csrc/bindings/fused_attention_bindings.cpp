#include <torch/extension.h>

torch::Tensor FusedAttention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_attention",
        &FusedAttention_cuda,
        "Fused Core Attention (CUDA)"
    );
}
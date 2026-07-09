#include <torch/extension.h>

torch::Tensor scaled_QK_cuda(
    torch::Tensor Q,
    torch::Tensor K
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "scaled_qk",
        &scaled_QK_cuda,
        "Scaled QK (CUDA)"
    );
}
#include <torch/extension.h>

torch::Tensor fused_MLPlinear_GELU_cuda(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor b
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_MLPlinear_GELU",
        &fused_MLPlinear_GELU_cuda,
        "Fused MLPlinear + Bias + GELU (CUDA)"
    );
}
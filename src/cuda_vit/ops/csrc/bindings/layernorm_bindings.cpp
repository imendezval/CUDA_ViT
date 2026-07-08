#include <torch/extension.h>

torch::Tensor LayerNorm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "layernorm",
        &LayerNorm_cuda,
        "LayerNorm (CUDA)"
    );
}
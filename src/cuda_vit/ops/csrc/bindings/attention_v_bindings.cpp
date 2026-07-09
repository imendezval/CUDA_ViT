#include <torch/extension.h>

torch::Tensor attention_V_cuda(
    torch::Tensor att_scores,
    torch::Tensor V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "attention_v",
        &attention_V_cuda,
        "Attention @ V (CUDA)"
    );
}
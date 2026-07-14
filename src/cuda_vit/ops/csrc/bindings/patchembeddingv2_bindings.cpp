#include <torch/extension.h>

torch::Tensor PatchEmbeddingV2_cuda(
    torch::Tensor x,
    torch::Tensor W
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "patchembeddingv2",
        &PatchEmbeddingV2_cuda,
        "PatchEmbedding V2 (CUDA)"
    );
}
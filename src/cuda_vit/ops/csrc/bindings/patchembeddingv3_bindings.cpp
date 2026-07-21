#include <torch/extension.h>

torch::Tensor PatchEmbeddingV3_cuda(
    torch::Tensor x,
    torch::Tensor W
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "patchembeddingv3",
        &PatchEmbeddingV3_cuda,
        "PatchEmbedding V3 (CUDA)"
    );
}
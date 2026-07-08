#include <torch/extension.h>

torch::Tensor PatchEmbedding_cuda(
    torch::Tensor x,
    torch::Tensor W
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "patchembedding",
        &PatchEmbedding_cuda,
        "PatchEmbedding (CUDA)"
    );
}
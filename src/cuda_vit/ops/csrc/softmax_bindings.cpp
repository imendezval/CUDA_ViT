#include <torch/extension.h>

torch::Tensor SoftMax_cuda(
    torch::Tensor x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "softmax",
        &SoftMax_cuda,
        "SoftMax (CUDA)"
    );
}
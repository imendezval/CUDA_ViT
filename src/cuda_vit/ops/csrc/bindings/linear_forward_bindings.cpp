#include <torch/extension.h>

torch::Tensor linear_forward_cuda(
    torch::Tensor x,
    torch::Tensor W
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "linear_forward",
        &linear_forward_cuda,
        "Linear Forward (CUDA)"
    );
}
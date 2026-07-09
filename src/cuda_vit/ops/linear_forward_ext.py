from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_linear_forward():
    return load(
        name="linear_forward",
        sources=[
            str(ROOT / "csrc" / "bindings" / "linear_forward_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "linear_forward_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
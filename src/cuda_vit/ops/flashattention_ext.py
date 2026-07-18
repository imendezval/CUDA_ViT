from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_flashattention():
    return load(
        name="flashattention_ext",
        sources=[
            str(ROOT / "csrc" / "bindings" / "flashattention_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "flashattention_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
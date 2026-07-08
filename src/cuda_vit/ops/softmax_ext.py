from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_softmax():
    return load(
        name="softmax_ext",
        sources=[
            str(ROOT / "csrc" / "softmax_bindings.cpp"),
            str(ROOT / "csrc" / "softmax_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
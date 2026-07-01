from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_vector_add():
    return load(
        name="vector_add_ext",
        sources=[
            str(ROOT / "csrc" / "bindings.cpp"),
            str(ROOT / "csrc" / "vector_add_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
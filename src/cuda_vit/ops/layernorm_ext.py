from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_layernorm():
    return load(
        name="layernorm_ext",
        sources=[
            str(ROOT / "csrc" / "bindings" / "layernorm_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "layernorm_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
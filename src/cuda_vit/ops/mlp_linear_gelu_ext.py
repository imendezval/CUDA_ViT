from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_fused_mlp_linear_gelu():
    return load(
        name="mlp_linear_gelu_ext",
        sources=[
            str(ROOT / "csrc" / "mlp_linear_gelu_bindings.cpp"),
            str(ROOT / "csrc" / "mlp_linear_gelu_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_scaled_qk():
    return load(
        name="scaled_qk_ext",
        sources=[
            str(ROOT / "csrc" / "bindings" / "scaled_qk_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "scaled_qk_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
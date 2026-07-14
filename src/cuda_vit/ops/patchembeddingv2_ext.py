from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_patchembeddingv2():
    return load(
        name="patchembeddingv2_ext",
        sources=[
            str(ROOT / "csrc" / "bindings" / "patchembeddingv2_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "patchembeddingv2_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
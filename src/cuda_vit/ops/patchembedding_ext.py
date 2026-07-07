from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_patchembedding():
    return load(
        name="patchembedding_ext",
        sources=[
            str(ROOT / "csrc" / "patchembedding_bindings.cpp"),
            str(ROOT / "csrc" / "patchembedding_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
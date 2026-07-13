from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_fused_attention():
    return load(
        name="fused_attention_ext",
        sources=[
            str(ROOT / "csrc" / "bindings" / "fused_attention_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "fused_attention_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def load_attention_v():
    return load(
        name="attention_v_ext",
        sources=[
            str(ROOT / "csrc" / "bindings" / "attention_v_bindings.cpp"),
            str(ROOT / "csrc" / "kernels"  / "attention_v_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
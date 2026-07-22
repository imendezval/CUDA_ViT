from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchEmbeddingProfileShape:
    batch: int
    channels: int
    image_h: int
    image_w: int
    patch: int
    embed: int

    @property
    def patch_elements(self) -> int:
        return self.channels * self.patch * self.patch

    @property
    def patches(self) -> int:
        return (self.image_h // self.patch) * (self.image_w // self.patch)

    @property
    def label(self) -> str:
        return (
            f"B{self.batch}_C{self.channels}_H{self.image_h}_W{self.image_w}_"
            f"P{self.patch}_D{self.embed}"
        )


PRESENTATION_SHAPE = PatchEmbeddingProfileShape(2, 3, 224, 224, 16, 768)
PROFILE_VARIANTS = (
    "PyTorch Conv2d",
    "PatchEmbedding v3",
)


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def run_variant(name: str, op: Callable[[], torch.Tensor], iterations: int) -> None:
    with nvtx_range(name):
        for _ in range(iterations):
            op()


def make_inputs(shape: PatchEmbeddingProfileShape) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(
        shape.batch,
        shape.channels,
        shape.image_h,
        shape.image_w,
        device="cuda",
        dtype=torch.float32,
    )
    weight = torch.randn(
        shape.embed,
        shape.patch_elements,
        device="cuda",
        dtype=torch.float32,
    )
    return x, weight


def pytorch_patchembedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    shape: PatchEmbeddingProfileShape,
) -> torch.Tensor:
    conv_weight = weight.view(shape.embed, shape.channels, shape.patch, shape.patch)
    conv_out = F.conv2d(x, conv_weight, bias=None, stride=shape.patch)
    return (
        conv_out.permute(0, 2, 3, 1)
        .reshape(shape.batch, shape.patches, shape.embed)
        .contiguous()
    )


def profile_patch_embedding(
    *,
    warmup: int,
    iterations: int,
    use_cuda_profiler: bool,
) -> None:
    from cuda_vit.ops.patchembeddingv3_ext import load_patchembeddingv2 as load_patchembeddingv3

    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    shape = PRESENTATION_SHAPE
    x, weight = make_inputs(shape)
    ext_v3 = load_patchembeddingv3()
    ops = (
        ("PyTorch Conv2d", lambda: pytorch_patchembedding(x, weight, shape)),
        ("PatchEmbedding v3", lambda: ext_v3.patchembeddingv3(x, weight)),
    )

    with torch.inference_mode():
        for _, op in ops:
            for _ in range(warmup):
                op()
        torch.cuda.synchronize()

        if use_cuda_profiler:
            torch.cuda.profiler.start()
        with nvtx_range(f"patch_embedding:{shape.label}"):
            for name, op in ops:
                run_variant(name, op, iterations)
        torch.cuda.synchronize()
        if use_cuda_profiler:
            torch.cuda.profiler.stop()

    print(f"shape={shape.label}")
    print("variants=" + ",".join(PROFILE_VARIANTS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--no-cuda-profiler",
        action="store_true",
        help="Do not emit cudaProfilerStart/Stop markers for Nsight capture ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    profile_patch_embedding(
        warmup=args.warmup,
        iterations=args.iterations,
        use_cuda_profiler=not args.no_cuda_profiler,
    )


if __name__ == "__main__":
    main()

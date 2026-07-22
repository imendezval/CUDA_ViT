from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmarks.common.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    check_close,
    effective_bandwidth_gbs,
    format_comparison,
    format_correctness,
    format_run_header,
    format_table,
    time_cuda,
)
from cuda_vit.ops.patchembedding_ext import load_patchembedding
from cuda_vit.ops.patchembeddingv2_ext import load_patchembeddingv2
from cuda_vit.ops.patchembeddingv3_ext import load_patchembeddingv2 as load_patchembeddingv3


@dataclass(frozen=True)
class PatchEmbeddingShape:
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


SHAPES = (
    PatchEmbeddingShape(1, 3, 32, 32, 16, 64),
    PatchEmbeddingShape(4, 3, 32, 32, 8, 64),
    PatchEmbeddingShape(2, 3, 224, 224, 16, 384),
    PatchEmbeddingShape(8, 3, 224, 224, 16, 768),
)


def pytorch_patchembedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    shape: PatchEmbeddingShape,
) -> torch.Tensor:
    conv_weight = weight.view(
        shape.embed,
        shape.channels,
        shape.patch,
        shape.patch,
    )
    conv_out = F.conv2d(x, conv_weight, bias=None, stride=shape.patch)
    return (
        conv_out.permute(0, 2, 3, 1)
        .reshape(shape.batch, shape.patches, shape.embed)
        .contiguous()
    )


def logical_bytes(shape: PatchEmbeddingShape) -> int:
    outputs = shape.batch * shape.patches * shape.embed
    reads_per_output = 2 * shape.patch_elements
    writes_per_output = 1
    return outputs * (reads_per_output + writes_per_output) * 4


def make_inputs(shape: PatchEmbeddingShape) -> tuple[torch.Tensor, torch.Tensor]:
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


def benchmark_shape(
    ext_v1: object,
    ext_v2: object,
    ext_v3: object,
    shape: PatchEmbeddingShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    x, weight = make_inputs(shape)

    expected = pytorch_patchembedding(x, weight, shape)
    correctness = (
        check_close(
            "patchembedding",
            ext_v1.patchembedding(x, weight),
            expected,
            rtol=1e-4,
            atol=1e-4,
        ),
        check_close(
            "patchembeddingv2",
            ext_v2.patchembeddingv2(x, weight),
            expected,
            rtol=1e-4,
            atol=1e-4,
        ),
        check_close(
            "patchembeddingv3",
            ext_v3.patchembeddingv3(x, weight),
            expected,
            rtol=1e-4,
            atol=1e-4,
        ),
    )

    timings = (
        time_cuda(
            "pytorch_conv2d",
            lambda: pytorch_patchembedding(x, weight, shape),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "patchembedding",
            lambda: ext_v1.patchembedding(x, weight),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "patchembeddingv2",
            lambda: ext_v2.patchembeddingv2(x, weight),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "patchembeddingv3",
            lambda: ext_v3.patchembeddingv3(x, weight),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    )

    bytes_per_call = logical_bytes(shape)

    print(f"\nshape={shape.label}")
    for result in correctness:
        print(format_correctness(result))
    print(format_table(timings))
    for timing in timings:
        bandwidth = effective_bandwidth_gbs(bytes_per_call, timing)
        print(f"{timing.name}: logical_bandwidth={bandwidth:.1f} GB/s")
    print(format_comparison(timings[0], timings[1]))
    print(format_comparison(timings[0], timings[2]))
    print(format_comparison(timings[0], timings[3]))
    print(format_comparison(timings[1], timings[2]))
    print(format_comparison(timings[2], timings[3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    config = BenchmarkConfig(
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(format_run_header("Patch Embedding Benchmark", BenchmarkEnv.current(), config))

    ext_v1 = load_patchembedding()
    ext_v2 = load_patchembeddingv2()
    ext_v3 = load_patchembeddingv3()

    for shape in SHAPES:
        benchmark_shape(
            ext_v1,
            ext_v2,
            ext_v3,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()

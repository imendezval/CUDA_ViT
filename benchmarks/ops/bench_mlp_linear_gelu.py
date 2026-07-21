from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmarks.common.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    check_close,
    effective_tflops,
    format_comparison,
    format_correctness,
    format_run_header,
    format_table,
    time_cuda,
)
from cuda_vit.ops.mlp_linear_gelu_ext import load_fused_mlp_linear_gelu


@dataclass(frozen=True)
class MlpShape:
    rows: int
    in_features: int
    out_features: int

    @property
    def label(self) -> str:
        return f"R{self.rows}_In{self.in_features}_Out{self.out_features}"

    @property
    def linear_flops(self) -> int:
        return 2 * self.rows * self.in_features * self.out_features


SHAPES = (
    MlpShape(2, 64, 128),
    MlpShape(8, 128, 512),
    MlpShape(394, 768, 3072),
    MlpShape(1576, 384, 1536),
)


def make_inputs(shape: MlpShape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(
        shape.rows,
        shape.in_features,
        device="cuda",
        dtype=torch.float32,
    )
    weight = torch.randn(
        shape.out_features,
        shape.in_features,
        device="cuda",
        dtype=torch.float32,
    )
    bias = torch.randn(
        shape.out_features,
        device="cuda",
        dtype=torch.float32,
    )
    return x, weight, bias


def pytorch_mlp_linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return F.gelu(F.linear(x, weight, bias), approximate="tanh")


def benchmark_shape(
    ext: object,
    shape: MlpShape,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    x, weight, bias = make_inputs(shape)
    expected = pytorch_mlp_linear_gelu(x, weight, bias)
    correctness = check_close(
        "fused_mlp_linear_gelu",
        ext.fused_MLPlinear_GELU(x, weight, bias),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    timings = (
        time_cuda(
            "pytorch_linear_gelu",
            lambda: pytorch_mlp_linear_gelu(x, weight, bias),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
        time_cuda(
            "fused_mlp_linear_gelu",
            lambda: ext.fused_MLPlinear_GELU(x, weight, bias),
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        ),
    )

    print(f"\nshape={shape.label}")
    print(format_correctness(correctness))
    print(format_table(timings))
    for timing in timings:
        throughput = effective_tflops(shape.linear_flops, timing)
        print(f"{timing.name}: estimated_linear_throughput={throughput:.2f} TFLOP/s")
    print(format_comparison(timings[0], timings[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
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
    print(format_run_header("MLP Linear GELU Benchmark", BenchmarkEnv.current(), config))

    ext = load_fused_mlp_linear_gelu()

    for shape in SHAPES:
        benchmark_shape(
            ext,
            shape,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()

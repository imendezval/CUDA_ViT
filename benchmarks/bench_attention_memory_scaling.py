from __future__ import annotations

import argparse

import torch

from benchmarks.bench_attention_memory import VARIANTS, measure_memory
from benchmarks.core import BenchmarkConfig, BenchmarkEnv, format_run_header
from benchmarks.profile_attention import make_inputs, make_variant, run_warmup
from benchmarks.shapes import AttentionShape


SEQUENCE_SWEEP = (
    AttentionShape(2, 3, 32, 64),
    AttentionShape(2, 3, 64, 64),
    AttentionShape(2, 3, 128, 64),
    AttentionShape(2, 3, 192, 64),
    AttentionShape(2, 3, 256, 64),
    AttentionShape(2, 3, 384, 64),
    AttentionShape(2, 3, 512, 64),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
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
        repeats=1,
    )
    print(format_run_header("Attention Memory Scaling Benchmark", BenchmarkEnv.current(), config))
    print("sweep,shape,variant,peak_allocated_bytes,peak_reserved_bytes,status")

    for shape in SEQUENCE_SWEEP:
        for variant in VARIANTS:
            if variant == "flashattention" and not shape.supports_flashattention:
                print(f"sequence,{shape.label},{variant},skipped,skipped,shape_not_supported")
                continue
            q, k, v = make_inputs(shape)
            op = make_variant(variant, shape, q, k, v)
            run_warmup(op, args.warmup)
            peak_allocated, peak_reserved = measure_memory(op, iterations=args.iterations)
            print(f"sequence,{shape.label},{variant},{peak_allocated},{peak_reserved},ok")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class Timing:
    name: str
    samples_ms: tuple[float, ...]
    warmup: int
    iterations: int

    @property
    def median_ms(self) -> float:
        return median(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return mean(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)


def time_cuda(
    name: str,
    op: Callable[[], object],
    *,
    warmup: int = 25,
    iterations: int = 100,
    repeats: int = 5,
) -> Timing:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA timing")
    if warmup < 0 or iterations <= 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0; iterations and repeats must be > 0")

    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            op()
        torch.cuda.synchronize()

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iterations):
                op()
            end.record()

            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iterations)

    return Timing(
        name=name,
        samples_ms=tuple(samples),
        warmup=warmup,
        iterations=iterations,
    )


def speedup(baseline: Timing, candidate: Timing) -> float:
    return baseline.median_ms / candidate.median_ms


def effective_bandwidth_gbs(bytes_per_call: int, timing: Timing) -> float:
    return bytes_per_call / (timing.median_ms * 1e-3) / 1e9


def format_timing(timing: Timing) -> str:
    return (
        f"{timing.name}: median={timing.median_ms:.4f} ms "
        f"mean={timing.mean_ms:.4f} ms min={timing.min_ms:.4f} ms "
        f"max={timing.max_ms:.4f} ms"
    )


def format_comparison(baseline: Timing, candidate: Timing) -> str:
    return (
        f"{candidate.name} vs {baseline.name}: "
        f"{speedup(baseline, candidate):.2f}x speedup "
        f"({baseline.median_ms:.4f} ms -> {candidate.median_ms:.4f} ms)"
    )


def format_table(rows: Iterable[Timing]) -> str:
    header = "name,median_ms,mean_ms,min_ms,max_ms"
    body = [
        (
            f"{row.name},{row.median_ms:.6f},{row.mean_ms:.6f},"
            f"{row.min_ms:.6f},{row.max_ms:.6f}"
        )
        for row in rows
    ]
    return "\n".join([header, *body])


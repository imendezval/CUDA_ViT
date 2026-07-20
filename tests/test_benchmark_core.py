import pytest

from benchmarks.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    Timing,
    effective_bandwidth_gbs,
    format_comparison,
    format_run_header,
    format_table,
    format_timing,
    speedup,
)


def test_timing_stats_use_samples():
    timing = Timing(
        name="custom",
        samples_ms=(3.0, 1.0, 2.0),
        warmup=10,
        iterations=100,
    )

    assert timing.median_ms == 2.0
    assert timing.mean_ms == 2.0
    assert timing.min_ms == 1.0
    assert timing.max_ms == 3.0


def test_speedup_uses_median_latency():
    baseline = Timing("pytorch", (4.0, 6.0, 8.0), warmup=1, iterations=1)
    candidate = Timing("custom", (2.0, 3.0, 4.0), warmup=1, iterations=1)

    assert speedup(baseline, candidate) == pytest.approx(2.0)


def test_effective_bandwidth_uses_median_latency():
    timing = Timing("custom", (1.0,), warmup=1, iterations=1)

    assert effective_bandwidth_gbs(1_000_000_000, timing) == pytest.approx(1000.0)


def test_format_helpers_are_stable():
    baseline = Timing("pytorch", (2.0,), warmup=1, iterations=1)
    candidate = Timing("custom", (1.0,), warmup=1, iterations=1)

    assert format_timing(candidate) == (
        "custom: median=1.0000 ms mean=1.0000 ms "
        "min=1.0000 ms max=1.0000 ms"
    )
    assert format_comparison(baseline, candidate) == (
        "custom vs pytorch: 2.00x speedup (2.0000 ms -> 1.0000 ms)"
    )
    assert format_table([baseline, candidate]) == "\n".join(
        [
            "name,median_ms,mean_ms,min_ms,max_ms",
            "pytorch,2.000000,2.000000,2.000000,2.000000",
            "custom,1.000000,1.000000,1.000000,1.000000",
        ]
    )


def test_format_run_header_includes_environment_and_config():
    env = BenchmarkEnv(
        gpu="Test GPU",
        pytorch="2.9.0",
        cuda="12.8",
    )
    config = BenchmarkConfig(
        warmup=10,
        iterations=50,
        repeats=3,
    )

    assert format_run_header("Attention Benchmark", env, config) == "\n".join(
        [
            "Attention Benchmark",
            "GPU: Test GPU",
            "PyTorch: 2.9.0",
            "PyTorch CUDA: 12.8",
            "warmup=10 iterations=50 repeats=3",
        ]
    )

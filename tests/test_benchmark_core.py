import pytest
import torch

from benchmarks.common.core import (
    BenchmarkConfig,
    BenchmarkEnv,
    Correctness,
    Timing,
    amdahl_speedup,
    check_close,
    effective_bandwidth_gbs,
    effective_tflops,
    format_comparison,
    format_correctness,
    format_run_header,
    format_table,
    format_timing,
    parallel_efficiency,
    speedup,
    throughput_scale,
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


def test_effective_tflops_uses_median_latency():
    timing = Timing("custom", (2.0,), warmup=1, iterations=1)

    assert effective_tflops(4_000_000_000, timing) == pytest.approx(2.0)


def test_parallel_efficiency_is_speedup_per_parallel_unit():
    assert parallel_efficiency(speedup_value=6.0, parallel_units=8) == pytest.approx(0.75)


def test_parallel_efficiency_rejects_invalid_parallel_units():
    with pytest.raises(ValueError, match="parallel_units"):
        parallel_efficiency(speedup_value=1.0, parallel_units=0)


def test_throughput_scale_compares_work_per_ms():
    baseline = Timing("B1", (1.0,), warmup=1, iterations=1)
    candidate = Timing("B2", (1.5,), warmup=1, iterations=1)

    assert throughput_scale(
        baseline_work=1.0,
        baseline=baseline,
        candidate_work=2.0,
        candidate=candidate,
    ) == pytest.approx(4.0 / 3.0)


def test_throughput_scale_rejects_invalid_work_values():
    timing = Timing("custom", (1.0,), warmup=1, iterations=1)

    with pytest.raises(ValueError, match="work"):
        throughput_scale(0.0, timing, 1.0, timing)


def test_amdahl_speedup_estimates_end_to_end_limit():
    assert amdahl_speedup(runtime_fraction=0.4, optimized_speedup=2.0) == pytest.approx(1.25)


def test_amdahl_speedup_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="runtime_fraction"):
        amdahl_speedup(runtime_fraction=1.1, optimized_speedup=2.0)
    with pytest.raises(ValueError, match="optimized_speedup"):
        amdahl_speedup(runtime_fraction=0.5, optimized_speedup=0.0)


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


def test_check_close_returns_error_stats():
    actual = torch.tensor([1.0, 2.0, 3.25])
    expected = torch.tensor([1.0, 2.5, 3.0])

    result = check_close(
        "custom",
        actual,
        expected,
        rtol=1.0,
        atol=1.0,
    )

    assert result == Correctness(
        name="custom",
        max_abs_error=0.5,
        mean_abs_error=0.25,
        rtol=1.0,
        atol=1.0,
    )


def test_check_close_raises_when_tolerance_fails():
    actual = torch.tensor([1.0])
    expected = torch.tensor([2.0])

    with pytest.raises(AssertionError):
        check_close("custom", actual, expected, rtol=0.0, atol=0.0)


def test_format_correctness_is_stable():
    result = Correctness(
        name="custom",
        max_abs_error=0.000123,
        mean_abs_error=0.0000456,
        rtol=1e-4,
        atol=1e-5,
    )

    assert format_correctness(result) == (
        "custom: max_abs_error=0.000123 mean_abs_error=4.56e-05 "
        "rtol=0.0001 atol=1e-05"
    )

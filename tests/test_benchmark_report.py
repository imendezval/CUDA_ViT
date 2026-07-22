from benchmarks.common.report import (
    PATCH_SCALING_HEADER,
    read_rows,
    report_attention_memory,
    report_attention_scaling,
    report_patch_scaling,
)


def test_report_attention_scaling_skips_metadata(tmp_path):
    path = tmp_path / "attention.csv"
    path.write_text(
        "\n".join(
            [
                "Attention Scaling Benchmark",
                "GPU: Test GPU",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_sdpa,throughput_scale",
                "batch,B1,pytorch_sdpa,1.000000,1.0000,1.0000",
                "batch,B1,custom_3_kernel,2.000000,0.5000,1.0000",
                "batch,B1,fused_attention,0.800000,1.2500,1.0000",
                "batch,B1,flashattention,0.500000,2.0000,1.0000",
            ]
        )
    )

    assert report_attention_scaling(path) == "\n".join(
        [
            "| Sweep | Shape | Fastest | Fastest ms | Custom vs SDPA | Fused vs SDPA | Flash vs SDPA |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| batch | B1 | flashattention | 0.500000 | 0.5000 | 1.2500 | 2.0000 |",
        ]
    )


def test_report_patch_scaling_summarizes_v1_v2_v3(tmp_path):
    path = tmp_path / "patch.csv"
    path.write_text(
        "\n".join(
            [
                "Patch Embedding Scaling Benchmark",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs,throughput_scale",
                "image,S,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
                "image,S,patchembedding,2.000000,0.5000,50.0,1.0000",
                "image,S,patchembeddingv2,0.750000,1.3333,140.0,1.0000",
                "image,S,patchembeddingv3,0.500000,2.0000,200.0,1.0000",
            ]
        )
    )

    assert report_patch_scaling(path) == "\n".join(
        [
            "| Sweep | Shape | Fastest | Fastest ms | v1 vs conv2d | v2 vs conv2d | v3 vs conv2d | v3 GB/s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            "| image | S | patchembeddingv3 | 0.500000 | 0.5000 | 1.3333 | 2.0000 | 200.0 |",
        ]
    )


def test_read_rows_skips_loader_noise_after_header(tmp_path):
    path = tmp_path / "patch.csv"
    path.write_text(
        "\n".join(
            [
                "Patch Embedding Scaling Benchmark",
                ",".join(PATCH_SCALING_HEADER),
                "ninja: no work to do.",
                "batch,S,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
            ]
        )
    )

    assert read_rows(path, PATCH_SCALING_HEADER) == [
        {
            "sweep": "batch",
            "shape": "S",
            "name": "pytorch_conv2d",
            "median_ms": "1.000000",
            "speedup_vs_pytorch_conv2d": "1.0000",
            "logical_bandwidth_gbs": "100.0",
            "throughput_scale": "1.0000",
        }
    ]


def test_report_attention_memory_formats_bytes_as_mib(tmp_path):
    path = tmp_path / "memory.csv"
    path.write_text(
        "\n".join(
            [
                "Attention Memory Benchmark",
                "variant,shape,peak_allocated_bytes,peak_reserved_bytes,status",
                "pytorch_sdpa,S,1048576,2097152,ok",
                "flashattention,S,skipped,skipped,shape_not_supported",
            ]
        )
    )

    assert report_attention_memory(path) == "\n".join(
        [
            "| Variant | Shape | Peak Allocated MiB | Peak Reserved MiB | Status |",
            "| --- | --- | --- | --- | --- |",
            "| pytorch_sdpa | S | 1.00 | 2.00 | ok |",
            "| flashattention | S | skipped | skipped | shape_not_supported |",
        ]
    )

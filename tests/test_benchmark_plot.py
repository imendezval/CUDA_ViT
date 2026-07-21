from benchmarks.reporting.plot import (
    extract_sweep_label,
    extra_memory_series_vs_sdpa,
    metric_value,
    plot_attention_extra_memory_scaling,
    plot_attention_memory,
    plot_attention_memory_scaling,
    plot_attention_scaling,
    plot_patch_scaling,
    plot_vit_scaling,
)


def test_plot_attention_scaling_writes_svg(tmp_path):
    csv_path = tmp_path / "attention.csv"
    svg_path = tmp_path / "attention.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Attention Scaling Benchmark",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_sdpa,throughput_scale",
                "batch,B1,pytorch_sdpa,1.000000,1.0000,1.0000",
                "batch,B1,custom_3_kernel,2.000000,0.5000,1.0000",
                "batch,B2,pytorch_sdpa,1.500000,1.0000,1.3333",
                "batch,B2,custom_3_kernel,3.000000,0.5000,1.3333",
            ]
        )
    )

    plot_attention_scaling(csv_path, svg_path, "median_ms", sweep=None)

    svg = svg_path.read_text()
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Attention Latency" in svg
    assert "Latency (ms)" in svg
    assert "Custom 3 Part Kernel" in svg
    assert "batch:B2" in svg


def test_plot_patch_scaling_writes_svg(tmp_path):
    csv_path = tmp_path / "patch.csv"
    svg_path = tmp_path / "patch.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Patch Embedding Scaling Benchmark",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs,throughput_scale",
                "image,S,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
                "image,S,patchembeddingv2,0.750000,1.3333,140.0,1.1000",
            ]
        )
    )

    plot_patch_scaling(csv_path, svg_path, "throughput_scale", sweep=None)

    svg = svg_path.read_text()
    assert "Patch Embedding Throughput Scaling" in svg
    assert "Throughput Scaling (x)" in svg
    assert "PatchEmbedding v2" in svg


def test_plot_patch_scaling_can_show_images_per_second(tmp_path):
    csv_path = tmp_path / "patch.csv"
    svg_path = tmp_path / "patch.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Patch Embedding Scaling Benchmark",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs,throughput_scale",
                "batch,B2_C3_H224_W224_P16_D384,pytorch_conv2d,2.000000,1.0000,100.0,1.0000",
            ]
        )
    )

    plot_patch_scaling(csv_path, svg_path, "images_per_s", sweep="batch")

    svg = svg_path.read_text()
    assert "Patch Embedding Image Throughput vs Batch Size" in svg
    assert "Images / s" in svg


def test_plot_attention_scaling_can_show_tokens_per_second(tmp_path):
    csv_path = tmp_path / "attention.csv"
    svg_path = tmp_path / "attention.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Attention Scaling Benchmark",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_sdpa,throughput_scale",
                "sequence,B2_H3_T64_Dh64,pytorch_sdpa,1.000000,1.0000,1.0000",
            ]
        )
    )

    plot_attention_scaling(csv_path, svg_path, "tokens_per_s", sweep="sequence")

    svg = svg_path.read_text()
    assert "Attention Token Throughput vs Sequence Length (tokens)" in svg
    assert "Tokens / s" in svg


def test_metric_value_derives_real_throughput_rates():
    assert metric_value({"shape": "B2_C3_H224_W224_P16_D384", "median_ms": "2.0"}, "images_per_s") == 1000.0
    assert metric_value({"shape": "B2_H3_T64_Dh64", "median_ms": "1.0"}, "tokens_per_s") == 128000.0


def test_plot_patch_scaling_filters_one_sweep(tmp_path):
    csv_path = tmp_path / "patch.csv"
    svg_path = tmp_path / "patch.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Patch Embedding Scaling Benchmark",
                "sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs,throughput_scale",
                "batch,B1,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
                "batch,B2,pytorch_conv2d,2.000000,1.0000,100.0,1.0000",
                "embed,D64,pytorch_conv2d,1.500000,1.0000,100.0,1.0000",
                "embed,D384,pytorch_conv2d,3.000000,1.0000,100.0,1.0000",
            ]
        )
    )

    plot_patch_scaling(csv_path, svg_path, "median_ms", sweep="embed")

    svg = svg_path.read_text()
    assert "Patch Embedding Latency vs Embedding Dimension" in svg
    assert "Embedding Dimension" in svg
    assert ">384<" in svg
    assert "D384" not in svg
    assert "B2" not in svg


def test_extract_sweep_label_uses_only_changed_dimension():
    patch_shape = "B2_C3_H224_W224_P16_D1536"
    attention_shape = "B2_H12_T192_Dh64"
    vit_shape = "B2_I256_P16_T256_D192_H3_L2_nocls"

    assert extract_sweep_label("batch", patch_shape) == "2"
    assert extract_sweep_label("image", patch_shape) == "224"
    assert extract_sweep_label("image", vit_shape) == "256"
    assert extract_sweep_label("patch", patch_shape) == "16"
    assert extract_sweep_label("embed", patch_shape) == "1536"
    assert extract_sweep_label("sequence", attention_shape) == "192"
    assert extract_sweep_label("heads", attention_shape) == "12"


def test_plot_attention_memory_writes_svg(tmp_path):
    csv_path = tmp_path / "memory.csv"
    svg_path = tmp_path / "memory.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Attention Memory Benchmark",
                "variant,shape,peak_allocated_bytes,peak_reserved_bytes,status",
                "pytorch_sdpa,S,1048576,2097152,ok",
                "flashattention,S,skipped,skipped,shape_not_supported",
            ]
        )
    )

    plot_attention_memory(csv_path, svg_path)

    svg = svg_path.read_text()
    assert "Attention Peak Memory by Implementation" in svg
    assert "Peak Allocated Memory (MiB)" in svg
    assert "PyTorch SDPA" in svg
    assert "flashattention" not in svg


def test_plot_attention_memory_scaling_writes_svg(tmp_path):
    csv_path = tmp_path / "memory_scaling.csv"
    svg_path = tmp_path / "memory_scaling.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Attention Memory Scaling Benchmark",
                "sweep,shape,variant,peak_allocated_bytes,peak_reserved_bytes,status",
                "sequence,B2_H3_T64_Dh64,pytorch_sdpa,1048576,2097152,ok",
                "sequence,B2_H3_T128_Dh64,pytorch_sdpa,2097152,4194304,ok",
                "sequence,B2_H3_T64_Dh64,flashattention,skipped,skipped,shape_not_supported",
            ]
        )
    )

    plot_attention_memory_scaling(csv_path, svg_path)

    svg = svg_path.read_text()
    assert "Attention Peak Memory vs Sequence Length" in svg
    assert "Peak Allocated Memory (MiB)" in svg
    assert "PyTorch SDPA" in svg
    assert "FlashAttention" not in svg


def test_extra_memory_series_vs_sdpa_subtracts_per_shape_baseline():
    rows = [
        {
            "sweep": "sequence",
            "shape": "B2_H3_T512_Dh64",
            "variant": "pytorch_sdpa",
            "peak_allocated_bytes": "1048576",
            "status": "ok",
        },
        {
            "sweep": "sequence",
            "shape": "B2_H3_T512_Dh64",
            "variant": "pytorch_manual",
            "peak_allocated_bytes": "3145728",
            "status": "ok",
        },
    ]

    series = extra_memory_series_vs_sdpa(rows)

    assert series["pytorch_sdpa"] == [("512", 0.0)]
    assert series["pytorch_manual"] == [("512", 2.0)]


def test_plot_attention_extra_memory_scaling_writes_svg(tmp_path):
    csv_path = tmp_path / "memory_scaling.csv"
    svg_path = tmp_path / "extra_memory.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Attention Memory Scaling Benchmark",
                "sweep,shape,variant,peak_allocated_bytes,peak_reserved_bytes,status",
                "sequence,B2_H3_T512_Dh64,pytorch_sdpa,1048576,2097152,ok",
                "sequence,B2_H3_T512_Dh64,pytorch_manual,3145728,4194304,ok",
            ]
        )
    )

    plot_attention_extra_memory_scaling(csv_path, svg_path)

    svg = svg_path.read_text()
    assert "Attention Extra Peak Memory vs Sequence Length" in svg
    assert "Extra Peak Memory vs PyTorch SDPA (MiB)" in svg
    assert "PyTorch Manual" in svg


def test_plot_vit_scaling_writes_svg(tmp_path):
    csv_path = tmp_path / "vit.csv"
    svg_path = tmp_path / "vit.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Whole ViT Scaling Benchmark",
                "sweep,shape,variant,status,median_ms,mean_ms,min_ms,max_ms,max_abs_error,mean_abs_error",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,pytorch_sdpa,ok,1.000000,1.0,1.0,1.0,0,0",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,custom_v1_3_kernel,ok,4.000000,4.0,4.0,4.0,0,0",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,custom_v2_fused_attention,ok,2.000000,2.0,2.0,2.0,0,0",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,custom_v2_flashattention_torch_linear,ok,3.000000,3.0,3.0,3.0,0,0",
            ]
        )
    )

    plot_vit_scaling(csv_path, svg_path, sweep="patches")

    svg = svg_path.read_text()
    assert "Whole ViT Inference Latency vs Number of Patches" in svg
    assert "Number of Patches" in svg
    assert "PatchEmbedding v2 + FlashAttention + cuBLAS Linear" in svg
    assert "PatchEmbedding v1 + 3 Part Kernel" not in svg
    assert "PatchEmbedding v2 + Fused Attention" not in svg
    assert ">64<" in svg


def test_plot_vit_scaling_can_include_pev1_3part(tmp_path):
    csv_path = tmp_path / "vit.csv"
    svg_path = tmp_path / "vit.svg"
    csv_path.write_text(
        "\n".join(
            [
                "Whole ViT Scaling Benchmark",
                "sweep,shape,variant,status,median_ms,mean_ms,min_ms,max_ms,max_abs_error,mean_abs_error",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,pytorch_sdpa,ok,1.000000,1.0,1.0,1.0,0,0",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,custom_v1_3_kernel,ok,4.000000,4.0,4.0,4.0,0,0",
                "patches,B2_I128_P16_T64_D192_H3_L2_nocls,custom_v2_flashattention_torch_linear,ok,3.000000,3.0,3.0,3.0,0,0",
            ]
        )
    )

    plot_vit_scaling(csv_path, svg_path, sweep="patches", include_pev1_3part=True)

    svg = svg_path.read_text()
    assert "Whole ViT Inference Latency vs Number of Patches with PEv1 3 Part Kernel" in svg
    assert "PatchEmbedding v1 + 3 Part Kernel" in svg

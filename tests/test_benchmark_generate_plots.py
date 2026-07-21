from benchmarks.generate_plots import generate_plots, plot_specs


PATCH_CSV = "\n".join(
    [
        "Patch Embedding Scaling Benchmark",
        "sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs,throughput_scale",
        "batch,B1_C3_H224_W224_P16_D384,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
        "batch,B1_C3_H224_W224_P16_D384,patchembeddingv2,2.000000,0.5000,50.0,1.0000",
        "image,B2_C3_H64_W64_P16_D384,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
        "patch,B2_C3_H224_W224_P16_D384,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
        "embed,B2_C3_H224_W224_P16_D384,pytorch_conv2d,1.000000,1.0000,100.0,1.0000",
    ]
)

ATTENTION_CSV = "\n".join(
    [
        "Attention Scaling Benchmark",
        "sweep,shape,name,median_ms,speedup_vs_pytorch_sdpa,throughput_scale",
        "batch,B1_H3_T192_Dh64,pytorch_sdpa,1.000000,1.0000,1.0000",
        "batch,B1_H3_T192_Dh64,custom_3_kernel,2.000000,0.5000,1.0000",
        "sequence,B2_H3_T64_Dh64,pytorch_sdpa,1.000000,1.0000,1.0000",
        "heads,B2_H3_T192_Dh64,pytorch_sdpa,1.000000,1.0000,1.0000",
    ]
)

ATTENTION_MEMORY_CSV = "\n".join(
    [
        "Attention Memory Scaling Benchmark",
        "sweep,shape,variant,peak_allocated_bytes,peak_reserved_bytes,status",
        "sequence,B2_H3_T64_Dh64,pytorch_sdpa,1048576,2097152,ok",
    ]
)

VIT_CSV = "\n".join(
    [
        "Whole ViT Scaling Benchmark",
        "sweep,shape,variant,status,median_ms,mean_ms,min_ms,max_ms,max_abs_error,mean_abs_error",
        "batch,B1_I128_P16_T64_D192_H3_L2_nocls,pytorch_sdpa,ok,1.000000,1.0,1.0,1.0,0,0",
        "image,B2_I128_P16_T64_D192_H3_L2_nocls,pytorch_sdpa,ok,1.000000,1.0,1.0,1.0,0,0",
        "patches,B2_I128_P16_T64_D192_H3_L2_nocls,pytorch_sdpa,ok,1.000000,1.0,1.0,1.0,0,0",
    ]
)


def write_inputs(tmp_path):
    patch = tmp_path / "patch.csv"
    attention = tmp_path / "attention.csv"
    memory = tmp_path / "memory.csv"
    vit = tmp_path / "vit.csv"
    patch.write_text(PATCH_CSV)
    attention.write_text(ATTENTION_CSV)
    memory.write_text(ATTENTION_MEMORY_CSV)
    vit.write_text(VIT_CSV)
    return patch, attention, memory, vit


def test_plot_specs_exclude_speedup_by_default():
    names = {spec.filename for spec in plot_specs(include_speedup=False)}

    assert "embed_latency.svg" in names
    assert "sequence_tokens_per_s.svg" in names
    assert "embed_images_per_s.svg" in names
    assert not any("speedup" in name for name in names)
    assert not any("throughput_scale" in name for name in names)


def test_generate_plots_removes_stale_speedup_outputs(tmp_path):
    patch, attention, memory, vit = write_inputs(tmp_path)
    output_root = tmp_path / "reports"
    patch_dir = output_root / "patch_embedding" / "plots"
    attention_dir = output_root / "attention" / "plots"
    patch_dir.mkdir(parents=True)
    attention_dir.mkdir(parents=True)
    stale = patch_dir / "embed_speedup_vs_conv2d.svg"
    stale.write_text("old")
    stale_throughput = attention_dir / "sequence_throughput_scale.svg"
    stale_throughput.write_text("old")

    outputs = generate_plots(patch, attention, memory, vit, output_root, include_speedup=False)

    assert not stale.exists()
    assert not stale_throughput.exists()
    assert patch_dir / "embed_latency.svg" in outputs
    assert attention_dir / "sequence_latency.svg" in outputs
    assert attention_dir / "sequence_tokens_per_s.svg" in outputs
    assert attention_dir / "sequence_peak_memory.svg" in outputs
    assert attention_dir / "sequence_extra_peak_memory_vs_sdpa.svg" in outputs
    assert output_root / "vit" / "plots" / "batch_latency.svg" in outputs
    assert not any("speedup" in path.name for path in outputs)


def test_generate_plots_can_include_speedup_outputs(tmp_path):
    patch, attention, memory, vit = write_inputs(tmp_path)
    output_dir = tmp_path / "reports"

    outputs = generate_plots(patch, attention, memory, vit, output_dir, include_speedup=True)

    names = {path.name for path in outputs}
    assert "embed_speedup_vs_conv2d.svg" in names
    assert "sequence_speedup_vs_sdpa.svg" in names

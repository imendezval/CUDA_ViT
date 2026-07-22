from benchmarks.reporting.generate_tables import (
    amdahl_component_limit_table,
    attention_custom_evolution_summary,
    attention_memory_scaling_table,
    attention_performance_summary,
    generate_tables,
    patch_custom_evolution_summary,
    patch_performance_summary,
    vit_component_breakdown_table,
)


PATCH_HEADER = "sweep,shape,name,median_ms,speedup_vs_pytorch_conv2d,logical_bandwidth_gbs,throughput_scale"
ATTENTION_HEADER = "sweep,shape,name,median_ms,speedup_vs_pytorch_sdpa,throughput_scale"
ATTENTION_MEMORY_HEADER = "sweep,shape,variant,peak_allocated_bytes,peak_reserved_bytes,status"
VIT_BREAKDOWN_HEADER = "variant,shape,component,median_ms,mean_ms,share_pct"


def patch_rows(sweep, shape, base):
    return [
        f"{sweep},{shape},pytorch_conv2d,{base:.6f},1.0000,100.0,1.0000",
        f"{sweep},{shape},patchembedding,{base * 4:.6f},0.2500,25.0,1.0000",
        f"{sweep},{shape},patchembeddingv2,{base / 2:.6f},2.0000,200.0,1.0000",
        f"{sweep},{shape},patchembeddingv3,{base / 4:.6f},4.0000,400.0,1.0000",
    ]


def attention_rows(sweep, shape, base, *, flash=True):
    rows = [
        f"{sweep},{shape},pytorch_sdpa,{base:.6f},1.0000,1.0000",
        f"{sweep},{shape},pytorch_manual,{base * 2:.6f},0.5000,1.0000",
        f"{sweep},{shape},custom_3_kernel,{base * 4:.6f},0.2500,1.0000",
        f"{sweep},{shape},fused_attention,{base * 1.5:.6f},0.6667,1.0000",
    ]
    if flash:
        rows.append(f"{sweep},{shape},flashattention,{base / 2:.6f},2.0000,1.0000")
    return rows


PATCH_CSV = "\n".join(
    [
        "Patch Embedding Scaling Benchmark",
        PATCH_HEADER,
        *patch_rows("batch", "B32_C3_H224_W224_P16_D384", 1.0),
        *patch_rows("image", "B2_C3_H512_W512_P16_D384", 2.0),
        *patch_rows("patch", "B2_C3_H224_W224_P56_D384", 3.0),
        *patch_rows("embed", "B2_C3_H224_W224_P16_D384", 4.0),
        *patch_rows("embed", "B2_C3_H224_W224_P16_D3072", 5.0),
    ]
)

ATTENTION_CSV = "\n".join(
    [
        "Attention Scaling Benchmark",
        ATTENTION_HEADER,
        *attention_rows("batch", "B16_H3_T192_Dh64", 1.0),
        *attention_rows("sequence", "B2_H3_T192_Dh64", 2.0),
        *attention_rows("sequence", "B2_H3_T512_Dh64", 3.0),
        *attention_rows("heads", "B2_H24_T192_Dh64", 4.0),
        "head_dim,Dh128,pytorch_sdpa,1.000000,1.0000,1.0000",
        "head_dim,Dh128,pytorch_manual,1.000000,1.0000,1.0000",
        "head_dim,Dh128,custom_3_kernel,1.000000,1.0000,1.0000",
        "head_dim,Dh128,fused_attention,1.000000,1.0000,1.0000",
    ]
)

ATTENTION_MEMORY_CSV = "\n".join(
    [
        "Attention Memory Scaling Benchmark",
        ATTENTION_MEMORY_HEADER,
        "sequence,B2_H3_T64_Dh64,pytorch_manual,1048576,2097152,ok",
        "sequence,B2_H3_T64_Dh64,pytorch_sdpa,2097152,4194304,ok",
        "sequence,B2_H3_T64_Dh64,custom_3_kernel,3145728,4194304,ok",
        "sequence,B2_H3_T64_Dh64,fused_attention,1048576,2097152,ok",
        "sequence,B2_H3_T64_Dh64,flashattention,1048576,2097152,ok",
    ]
)

VIT_BREAKDOWN_CSV = "\n".join(
    [
        "Whole ViT Component Breakdown Benchmark",
        VIT_BREAKDOWN_HEADER,
        "custom_flash_own_linear,S,patch_embedding,1.000000,1.0,25.00",
        "custom_flash_own_linear,S,token_setup,0.100000,0.1,2.50",
        "custom_flash_own_linear,S,attention,3.000000,3.0,75.00",
        "custom_flash_own_linear,S,residual_add,0.010000,0.01,0.25",
        "custom_flash_own_linear,S,total,4.000000,4.0,100.00",
        "custom_flash_cublas_linear,S,patch_embedding,1.000000,1.0,50.00",
        "custom_flash_cublas_linear,S,token_setup,0.100000,0.1,5.00",
        "custom_flash_cublas_linear,S,attention,1.000000,1.0,50.00",
        "custom_flash_cublas_linear,S,residual_add,0.010000,0.01,0.50",
        "custom_flash_cublas_linear,S,total,2.000000,2.0,100.00",
    ]
)


def write_inputs(tmp_path):
    patch = tmp_path / "patch.csv"
    attention = tmp_path / "attention.csv"
    memory = tmp_path / "memory.csv"
    vit_breakdown = tmp_path / "vit_breakdown.csv"
    patch.write_text(PATCH_CSV)
    attention.write_text(ATTENTION_CSV)
    memory.write_text(ATTENTION_MEMORY_CSV)
    vit_breakdown.write_text(VIT_BREAKDOWN_CSV)
    return patch, attention, memory, vit_breakdown


def test_patch_performance_summary(tmp_path):
    patch, _, _, _ = write_inputs(tmp_path)

    summary = patch_performance_summary(patch)

    assert "PatchEmbedding v1" in summary
    assert "PatchEmbedding v2" in summary
    assert "PatchEmbedding v3" in summary
    assert "0/5" in summary
    assert "5/5" in summary
    assert "4.000x" in summary
    assert "0.250x" in summary
    assert "Median Latency" not in summary


def test_patch_custom_evolution_summary(tmp_path):
    patch, _, _, _ = write_inputs(tmp_path)

    summary = patch_custom_evolution_summary(patch)

    assert "PatchEmbedding v2 vs v1" in summary
    assert "PatchEmbedding v3 vs v2" in summary
    assert "PatchEmbedding v3 vs v1" in summary
    assert "5/5" in summary
    assert "2.000x" in summary
    assert "8.000x" in summary


def test_attention_performance_summary_uses_manual_and_sdpa_baselines(tmp_path):
    _, attention, _, _ = write_inputs(tmp_path)

    summary = attention_performance_summary(attention)

    assert "## Against PyTorch Manual" in summary
    assert "## Against PyTorch SDPA" in summary
    assert "FlashAttention" in summary
    assert "Custom 3 Part Kernel" in summary
    assert "Median Latency" not in summary


def test_attention_custom_evolution_summary(tmp_path):
    _, attention, _, _ = write_inputs(tmp_path)

    summary = attention_custom_evolution_summary(attention)

    assert "Fused Attention vs Custom 3 Part Kernel" in summary
    assert "FlashAttention vs Fused Attention" in summary
    assert "FlashAttention vs Custom 3 Part Kernel" in summary
    assert "4/4" in summary
    assert "2.667x" in summary
    assert "3.000x" in summary
    assert "8.000x" in summary


def test_generate_tables_writes_nested_report_files(tmp_path):
    patch, attention, memory, vit_breakdown = write_inputs(tmp_path)
    output_root = tmp_path / "reports"

    outputs = generate_tables(patch, attention, memory, output_root, vit_breakdown)

    assert output_root / "patch_embedding" / "tables" / "scaling_summary.md" in outputs
    assert output_root / "patch_embedding" / "tables" / "performance_summary.md" in outputs
    assert output_root / "patch_embedding" / "tables" / "custom_evolution_summary.md" in outputs
    assert output_root / "patch_embedding" / "tables" / "representative_latency.md" in outputs
    assert output_root / "patch_embedding" / "tables" / "largest_latency.md" in outputs
    assert output_root / "patch_embedding" / "tables" / "largest_throughput.md" in outputs
    assert output_root / "attention" / "tables" / "scaling_summary.md" in outputs
    assert output_root / "attention" / "tables" / "performance_summary.md" in outputs
    assert output_root / "attention" / "tables" / "custom_evolution_summary.md" in outputs
    assert output_root / "attention" / "tables" / "representative_latency.md" in outputs
    assert output_root / "attention" / "tables" / "largest_latency.md" in outputs
    assert output_root / "attention" / "tables" / "largest_throughput.md" in outputs
    assert output_root / "attention" / "tables" / "memory_scaling.md" in outputs
    assert output_root / "vit" / "tables" / "component_breakdown.md" in outputs
    assert output_root / "vit" / "tables" / "amdahl_limits.md" in outputs
    assert output_root / "takeaways.md" in outputs


def test_generate_tables_excludes_attention_head_dim_sweep(tmp_path):
    patch, attention, memory, _ = write_inputs(tmp_path)
    output_root = tmp_path / "reports"

    generate_tables(patch, attention, memory, output_root)

    scaling = (output_root / "attention" / "tables" / "scaling_summary.md").read_text()
    performance = (output_root / "attention" / "tables" / "performance_summary.md").read_text()
    assert "head_dim" not in scaling
    assert "Dh128" not in scaling
    assert "5/5" not in performance


def test_generate_tables_writes_presentation_table_contents(tmp_path):
    patch, attention, memory, vit_breakdown = write_inputs(tmp_path)
    output_root = tmp_path / "reports"

    generate_tables(patch, attention, memory, output_root, vit_breakdown)

    patch_latency = (
        output_root / "patch_embedding" / "tables" / "representative_latency.md"
    ).read_text()
    attention_throughput = (
        output_root / "attention" / "tables" / "largest_throughput.md"
    ).read_text()
    takeaways = (output_root / "takeaways.md").read_text()

    assert "Latency (ms)" in patch_latency
    assert "B2_H3_T512_Dh64" in attention_throughput
    assert "tokens/s" in attention_throughput
    assert "Does PatchEmbedding v2 improve over v1?" in takeaways
    assert "Does PatchEmbedding v3 improve over v2?" in takeaways


def test_attention_memory_scaling_table_formats_mib(tmp_path):
    _, _, memory, _ = write_inputs(tmp_path)

    table = attention_memory_scaling_table(memory)

    assert "PyTorch Manual (MiB)" in table
    assert "B2_H3_T64_Dh64" in table
    assert "1.00" in table
    assert "3.00" in table


def test_vit_component_breakdown_table_formats_component_rows(tmp_path):
    _, _, _, vit_breakdown = write_inputs(tmp_path)

    table = vit_component_breakdown_table(vit_breakdown)

    assert "Custom FlashAttention + Custom Linear (ms)" in table
    assert "Custom FlashAttention + cuBLAS Linear (%)" in table
    assert "Patch Embedding" in table
    assert "Attention" in table
    assert "Token Setup" not in table
    assert "Residual Adds" in table
    assert "75.00" in table


def test_amdahl_component_limit_table_uses_breakdown_shares(tmp_path):
    _, _, _, vit_breakdown = write_inputs(tmp_path)

    table = amdahl_component_limit_table(vit_breakdown)

    assert "Whole Speedup if 10x" in table
    assert "Whole Speedup if inf" in table
    assert "Patch Embedding" in table
    assert "Attention" in table
    assert "Token Setup" not in table
    assert "Residual Adds" not in table
    assert "1.600x" in table
    assert "4.000x" in table

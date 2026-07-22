from benchmarks.vit.bench_vit_breakdown import COMPONENTS, SHAPES, VARIANTS


def test_vit_breakdown_has_custom_linear_and_cublas_variants():
    names = {variant.name for variant in VARIANTS}

    assert names == {
        "pytorch_sdpa",
        "custom_flash_own_linear",
        "custom_flash_cublas_linear",
        "custom_v3_flash_own_linear",
        "custom_v3_flash_cublas_linear",
    }


def test_vit_breakdown_shapes_support_flashattention():
    for shape in SHAPES.values():
        assert shape.supports_flashattention
        assert shape.head_dim == 64


def test_vit_breakdown_components_are_presentation_ordered():
    assert COMPONENTS == (
        "patch_embedding",
        "token_setup",
        "layernorm",
        "qkv_projection",
        "attention",
        "output_projection",
        "mlp",
        "residual_add",
    )

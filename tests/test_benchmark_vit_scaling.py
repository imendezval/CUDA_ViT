from benchmarks.bench_vit_scaling import BATCH_SWEEP, IMAGE_SWEEP, PATCHES_SWEEP, VARIANTS


def test_vit_scaling_sweeps_change_one_dimension():
    assert [shape.batch for shape in BATCH_SWEEP] == [1, 2, 4, 8]
    assert len({shape.image_size for shape in BATCH_SWEEP}) == 1
    assert len({shape.patch for shape in BATCH_SWEEP}) == 1

    assert [shape.image_size for shape in IMAGE_SWEEP] == [128, 256, 384, 512]
    assert len({shape.batch for shape in IMAGE_SWEEP}) == 1
    assert len({shape.patch for shape in IMAGE_SWEEP}) == 1

    assert [shape.patches for shape in PATCHES_SWEEP] == [64, 256, 1024]
    assert len({shape.batch for shape in PATCHES_SWEEP}) == 1
    assert len({shape.image_size for shape in PATCHES_SWEEP}) == 1


def test_vit_scaling_uses_head_dim_64():
    for shape in (*BATCH_SWEEP, *IMAGE_SWEEP, *PATCHES_SWEEP):
        assert shape.head_dim == 64


def test_vit_image_and_patch_sweeps_start_where_flashattention_is_supported():
    for shape in (*IMAGE_SWEEP, *PATCHES_SWEEP):
        assert shape.supports_flashattention


def test_vit_scaling_includes_torch_linear_custom_attention_variants():
    names = {variant.name for variant in VARIANTS}

    assert names == {
        "pytorch_manual",
        "pytorch_sdpa",
        "custom_v1_3_kernel",
        "custom_v2_flashattention",
        "custom_v2_flashattention_torch_linear",
    }

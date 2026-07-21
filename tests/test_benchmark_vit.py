import pytest

from benchmarks.bench_vit import SHAPES, VARIANTS, VitShape, format_rows, selected_shapes
from benchmarks.core import Timing


def test_vit_shape_properties():
    shape = VitShape(2, 3, 128, 16, 192, 3, 4, 2, False)

    assert shape.patches == 64
    assert shape.tokens == 64
    assert shape.patch_elements == 768
    assert shape.head_dim == 64
    assert shape.mlp_dim == 768
    assert shape.supports_flashattention
    assert shape.label == "B2_I128_P16_T64_D192_H3_L2_nocls"


def test_vit_shape_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="image_size"):
        VitShape(1, 3, 65, 16, 192, 3, 4, 1, False)

    with pytest.raises(ValueError, match="embed_dim"):
        VitShape(1, 3, 64, 16, 190, 3, 4, 1, False)


def test_selected_shapes():
    assert selected_shapes("flash") == (SHAPES["flash"],)
    assert selected_shapes("all") == tuple(SHAPES.values())


def test_variants_include_whole_vit_comparisons():
    names = {variant.name for variant in VARIANTS}

    assert "pytorch_manual" in names
    assert "pytorch_sdpa" in names
    assert "custom_v1_3_kernel" in names
    assert "custom_v2_3_kernel" in names
    assert "custom_v2_fused_attention" in names
    assert "custom_v2_flashattention" in names


def test_format_rows_includes_speedups_and_skips():
    rows = [
        (
            "pytorch_manual",
            "shape",
            Timing("pytorch_manual", (4.0, 4.0, 4.0), 1, 1),
            0.0,
            0.0,
            None,
            None,
        ),
        (
            "pytorch_sdpa",
            "shape",
            Timing("pytorch_sdpa", (2.0, 2.0, 2.0), 1, 1),
            0.0,
            0.0,
            None,
            None,
        ),
        (
            "custom_v2_flashattention",
            "shape",
            None,
            None,
            None,
            None,
            None,
        ),
    ]

    text = format_rows(rows)

    assert "variant,shape,status,median_ms" in text
    assert "pytorch_sdpa,shape,ok,2.000000" in text
    assert "2.000000,1.000000" in text
    assert "custom_v2_flashattention,shape,skipped" in text

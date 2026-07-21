from benchmarks.common.shapes import ATTENTION_OP_SHAPES, ATTENTION_SHAPES, AttentionShape


def test_attention_shape_label_is_stable():
    shape = AttentionShape(batch=2, heads=3, tokens=197, head_dim=64)

    assert shape.label == "B2_H3_T197_Dh64"


def test_attention_matmul_flops_counts_multiply_adds():
    shape = AttentionShape(batch=2, heads=3, tokens=197, head_dim=64)

    assert shape.attention_matmul_flops == 2 * 2 * 3 * 197 * 197 * 64


def test_flashattention_support_matches_current_kernel_constraints():
    assert AttentionShape(2, 3, 192, 64).supports_flashattention
    assert not AttentionShape(2, 3, 197, 64).supports_flashattention
    assert not AttentionShape(2, 3, 192, 32).supports_flashattention


def test_attention_presets_include_flash_and_non_flash_shapes():
    assert any(shape.supports_flashattention for shape in ATTENTION_SHAPES)
    assert any(not shape.supports_flashattention for shape in ATTENTION_SHAPES)
    assert any(shape.head_dim != 64 for shape in ATTENTION_OP_SHAPES)

from benchmarks.profiling import profile_attention_nsys, profile_patch_embedding_nsys


def test_patch_embedding_nsys_profile_uses_presentation_shape_and_variants():
    shape = profile_patch_embedding_nsys.PRESENTATION_SHAPE

    assert shape.label == "B2_C3_H224_W224_P16_D768"
    assert profile_patch_embedding_nsys.PROFILE_VARIANTS == (
        "PyTorch Conv2d",
        "PatchEmbedding v3",
    )


def test_attention_nsys_profile_uses_flash_compatible_presentation_shape_and_variants():
    shape = profile_attention_nsys.PRESENTATION_SHAPE

    assert shape.label == "B2_H3_T512_Dh64"
    assert shape.supports_flashattention
    assert profile_attention_nsys.PROFILE_VARIANTS == (
        "PyTorch SDPA",
        "Custom 3 Part Kernel",
        "Fused Attention",
        "FlashAttention",
    )

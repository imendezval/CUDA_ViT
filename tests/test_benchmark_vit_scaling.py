from benchmarks.bench_vit_scaling import BATCH_SWEEP, IMAGE_SWEEP, PATCHES_SWEEP


def test_vit_scaling_sweeps_change_one_dimension():
    assert [shape.batch for shape in BATCH_SWEEP] == [1, 2, 4, 8]
    assert len({shape.image_size for shape in BATCH_SWEEP}) == 1
    assert len({shape.patch for shape in BATCH_SWEEP}) == 1

    assert [shape.image_size for shape in IMAGE_SWEEP] == [64, 128, 192, 256]
    assert len({shape.batch for shape in IMAGE_SWEEP}) == 1
    assert len({shape.patch for shape in IMAGE_SWEEP}) == 1

    assert [shape.patches for shape in PATCHES_SWEEP] == [16, 64, 256]
    assert len({shape.batch for shape in PATCHES_SWEEP}) == 1
    assert len({shape.image_size for shape in PATCHES_SWEEP}) == 1


def test_vit_scaling_uses_head_dim_64():
    for shape in (*BATCH_SWEEP, *IMAGE_SWEEP, *PATCHES_SWEEP):
        assert shape.head_dim == 64

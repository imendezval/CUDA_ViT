import pytest
import torch
import torch.nn.functional as F

from cuda_vit.ops.attention_v_ext import load_attention_v


@pytest.fixture(scope="session")
def ext():
    return load_attention_v()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (1, 1, 4, 8),
        (1, 2, 16, 32),
        (2, 3, 32, 64),
        (2, 4, 64, 64),
        (2, 3, 197, 64),  # ViT-like
    ],
)
def test_attention_v_matches_pytorch(ext, shape):
    torch.manual_seed(123)

    B, H, T, Dh = shape

    att_scores = torch.randn(
        B, H, T, T,
        device="cuda",
        dtype=torch.float32,
    )

    # Make it realistic: attention scores are usually softmax probabilities.
    att_scores = F.softmax(att_scores, dim=-1).contiguous()

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    actual = ext.attention_v(att_scores, V)

    expected = torch.matmul(att_scores, V)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_attention_v_rejects_non_contiguous_attention_scores(ext):
    B, H, T, Dh = 2, 3, 16, 32

    att_scores = torch.randn(
        B, H, T, T,
        device="cuda",
        dtype=torch.float32,
    )

    # Same shape [B, H, T, T], but non-contiguous.
    att_scores = att_scores.transpose(-1, -2)

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.attention_v(att_scores, V)


def test_attention_v_rejects_non_contiguous_v(ext):
    B, H, T, Dh = 2, 3, 16, 32

    att_scores = torch.randn(
        B, H, T, T,
        device="cuda",
        dtype=torch.float32,
    )
    att_scores = F.softmax(att_scores, dim=-1).contiguous()

    # Create non-contiguous V while keeping shape [B, H, T, Dh].
    V = torch.randn(
        B, H, Dh, T,
        device="cuda",
        dtype=torch.float32,
    ).transpose(-1, -2)

    assert V.shape == (B, H, T, Dh)
    assert not V.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.attention_v(att_scores, V)


def test_attention_v_rejects_wrong_attention_shape(ext):
    B, H, T, Dh = 2, 3, 16, 32

    # Wrong: [B, H, T, T + 1]
    att_scores = torch.randn(
        B, H, T, T + 1,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="last 2 dimensions"):
        ext.attention_v(att_scores, V)


def test_attention_v_rejects_shape_mismatch(ext):
    B, H, T, Dh = 2, 3, 16, 32

    att_scores = torch.randn(
        B, H, T, T,
        device="cuda",
        dtype=torch.float32,
    )

    # Wrong T dimension in V.
    V = torch.randn(
        B, H, T + 1, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="must match"):
        ext.attention_v(att_scores, V)


def test_attention_v_rejects_wrong_dtype(ext):
    B, H, T, Dh = 2, 3, 16, 32

    att_scores = torch.randn(
        B, H, T, T,
        device="cuda",
        dtype=torch.float64,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="float32"):
        ext.attention_v(att_scores, V)
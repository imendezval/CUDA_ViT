import pytest
import torch
import torch.nn.functional as F

from cuda_vit.ops.fused_attention_ext import load_fused_attention


@pytest.fixture(scope="session")
def ext():
    return load_fused_attention()


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
def test_fused_attention_matches_pytorch(ext, shape):
    torch.manual_seed(123)

    B, H, T, Dh = shape

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    actual = ext.fused_attention(Q, K, V)

    scale = Dh ** -0.5
    att_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    att_probs = F.softmax(att_scores, dim=-1)
    expected = torch.matmul(att_probs, V)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_fused_attention_matches_torch_sdpa(ext):
    torch.manual_seed(123)

    B, H, T, Dh = 2, 3, 32, 64

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    actual = ext.fused_attention(Q, K, V)

    expected = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        dropout_p=0.0,
        is_causal=False,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_fused_attention_rejects_non_contiguous_q(ext):
    B, H, T, Dh = 2, 3, 16, 32

    # Same shape [B, H, T, Dh], but non-contiguous.
    Q = torch.randn(
        B, H, Dh, T,
        device="cuda",
        dtype=torch.float32,
    ).transpose(-1, -2)

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    assert Q.shape == (B, H, T, Dh)
    assert not Q.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_non_contiguous_k(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    # Same shape [B, H, T, Dh], but non-contiguous.
    K = torch.randn(
        B, H, Dh, T,
        device="cuda",
        dtype=torch.float32,
    ).transpose(-1, -2)

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    assert K.shape == (B, H, T, Dh)
    assert not K.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_non_contiguous_v(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    # Same shape [B, H, T, Dh], but non-contiguous.
    V = torch.randn(
        B, H, Dh, T,
        device="cuda",
        dtype=torch.float32,
    ).transpose(-1, -2)

    assert V.shape == (B, H, T, Dh)
    assert not V.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_wrong_q_shape(ext):
    B, H, T, Dh = 2, 3, 16, 32

    # Wrong: Q must be 4D.
    Q = torch.randn(
        B,
        H,
        T,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="shape"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_wrong_k_shape(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    # Wrong: K must be 4D.
    K = torch.randn(
        B,
        H,
        T,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="shape"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_wrong_v_shape(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    # Wrong: V must be 4D.
    V = torch.randn(
        B,
        H,
        T,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="shape"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_q_k_shape_mismatch(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    # Wrong T dimension.
    K = torch.randn(
        B, H, T + 1, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="same shape"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_q_v_shape_mismatch(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    # Wrong head_dim.
    V = torch.randn(
        B, H, T, Dh + 1,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="same shape"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_wrong_q_dtype(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float64,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="float32"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_wrong_k_dtype(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float64,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="float32"):
        ext.fused_attention(Q, K, V)


def test_fused_attention_rejects_wrong_v_dtype(ext):
    B, H, T, Dh = 2, 3, 16, 32

    Q = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float32,
    )

    V = torch.randn(
        B, H, T, Dh,
        device="cuda",
        dtype=torch.float64,
    )

    with pytest.raises(RuntimeError, match="float32"):
        ext.fused_attention(Q, K, V)
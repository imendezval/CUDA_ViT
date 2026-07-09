import pytest
import torch

from cuda_vit.ops.scaled_qk_ext import load_scaled_qk


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.fixture(scope="session")
def ext():
    return load_scaled_qk()


@pytest.mark.parametrize(
    "B,H,T,Dh",
    [
        # tiny correctness cases
        (1, 1, 1, 1),
        (1, 1, 2, 4),
        (1, 2, 3, 8),

        # small normal cases
        (2, 1, 8, 16),
        (2, 2, 16, 32),
        (4, 3, 32, 64),

        # ViT-like cases
        # B=2, heads=3, tokens=197, head_dim=64
        (2, 3, 197, 64),

        # smaller ViT-like project size
        # D = H * Dh = 384
        (2, 6, 197, 64),

        # non-power-of-two-ish token count/head dim
        (2, 4, 65, 48),
    ],
)
def test_scaled_qk_matches_pytorch(
    ext,
    B,
    H,
    T,
    Dh,
):
    torch.manual_seed(123)

    # Force PyTorch reference to use full FP32 math
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    Q = torch.randn(
        B,
        H,
        T,
        Dh,
        device="cuda",
        dtype=torch.float32,
    )

    K = torch.randn(
        B,
        H,
        T,
        Dh,
        device="cuda",
        dtype=torch.float32,
    )

    custom_out = ext.scaled_qk(Q, K)

    torch_out = torch.matmul(
        Q,
        K.transpose(-2, -1),
    ) / (Dh ** 0.5)

    assert custom_out.shape == (B, H, T, T)

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-4,
        atol=1e-4,
    )


def test_scaled_qk_output_shape(ext):
    torch.manual_seed(123)

    B = 2
    H = 3
    T = 197
    Dh = 64

    Q = torch.randn(B, H, T, Dh, device="cuda")
    K = torch.randn(B, H, T, Dh, device="cuda")

    out = ext.scaled_qk(Q, K)

    assert out.shape == (B, H, T, T)


def test_scaled_qk_rejects_non_contiguous_Q(ext):
    torch.manual_seed(123)

    B = 2
    H = 3
    T = 8
    Dh = 16

    # Shape is still [B, H, T, Dh], but tensor is non-contiguous
    Q = torch.randn(B, H, Dh, T, device="cuda").transpose(2, 3)

    assert Q.shape == (B, H, T, Dh)
    assert not Q.is_contiguous()

    K = torch.randn(B, H, T, Dh, device="cuda")

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.scaled_qk(Q, K)


def test_scaled_qk_rejects_non_contiguous_K(ext):
    torch.manual_seed(123)

    B = 2
    H = 3
    T = 8
    Dh = 16

    Q = torch.randn(B, H, T, Dh, device="cuda")

    # Shape is still [B, H, T, Dh], but tensor is non-contiguous.
    K = torch.randn(B, H, Dh, T, device="cuda").transpose(2, 3)

    assert K.shape == (B, H, T, Dh)
    assert not K.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.scaled_qk(Q, K)


def test_scaled_qk_rejects_wrong_shape(ext):
    torch.manual_seed(123)

    Q = torch.randn(2, 197, 64, device="cuda")
    K = torch.randn(2, 197, 64, device="cuda")

    with pytest.raises(RuntimeError, match="shape"):
        ext.scaled_qk(Q, K)


def test_scaled_qk_rejects_mismatched_shapes(ext):
    torch.manual_seed(123)

    Q = torch.randn(2, 3, 197, 64, device="cuda")
    K = torch.randn(2, 3, 196, 64, device="cuda")

    with pytest.raises(RuntimeError, match="same shape"):
        ext.scaled_qk(Q, K)


def test_scaled_qk_numerical_error_report(ext):
    torch.manual_seed(0)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    B = 2
    H = 3
    T = 197
    Dh = 64

    Q = torch.randn(B, H, T, Dh, device="cuda")
    K = torch.randn(B, H, T, Dh, device="cuda")

    custom_out = ext.scaled_qk(Q, K)

    torch_out = torch.matmul(
        Q,
        K.transpose(-2, -1),
    ) / (Dh ** 0.5)

    max_abs_error = (custom_out - torch_out).abs().max().item()

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-4,
        atol=1e-4,
    )

    # Useful when running pytest -s
    print("B:", B)
    print("H:", H)
    print("T:", T)
    print("Dh:", Dh)
    print("custom_out shape:", custom_out.shape)
    print("torch_out shape:", torch_out.shape)
    print("max absolute error:", max_abs_error)
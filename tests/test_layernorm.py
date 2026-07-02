import pytest
import torch
import torch.nn.functional as F

from cuda_vit.ops.layernorm_ext import load_layernorm


@pytest.fixture(scope="session")
def ext():
    return load_layernorm()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (1, 1, 31),
        (1, 2, 255),
        (2, 3, 256),
        (2, 3, 257),
        (8, 197, 384),  # ViT-like
    ],
)
def test_layernorm_matches_pytorch(ext, shape):
    torch.manual_seed(123)

    *_, d = shape
    eps = 1e-5

    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    gamma = torch.randn(d, device="cuda", dtype=torch.float32)
    beta = torch.randn(d, device="cuda", dtype=torch.float32)

    actual = ext.layernorm(x, gamma, beta, eps)

    expected = F.layer_norm(
        x,
        normalized_shape=(d,),
        weight=gamma,
        bias=beta,
        eps=eps,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_layernorm_rejects_non_contiguous_input(ext):
    eps = 1e-5
    d = 8

    # Shape remains [2, 8, 8], but transpose makes it non-contiguous.
    x = torch.randn(2, 8, 8, device="cuda").transpose(1, 2)

    gamma = torch.randn(d, device="cuda", dtype=torch.float32)
    beta = torch.randn(d, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.layernorm(x, gamma, beta, eps)
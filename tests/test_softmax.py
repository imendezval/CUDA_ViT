import pytest
import torch
import torch.nn.functional as F

from cuda_vit.ops.softmax_ext import load_softmax


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.fixture(scope="session")
def ext():
    return load_softmax()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1, 31),
        (1, 255),
        (2, 256),
        (2, 257),
        (8, 197),

        # Attention-like flattened case:
        # B=2, heads=4, T=197
        # scores [B, heads, T, T] -> [B * heads * T, T]
        (2 * 4 * 197, 197),
    ],
)
def test_softmax_matches_pytorch(ext, shape):
    torch.manual_seed(123)

    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    custom_out = ext.softmax(x)

    torch_out = F.softmax(x, dim=-1)

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-5,
        atol=1e-6,
    )


def test_softmax_matches_pytorch_large_values(ext):
    torch.manual_seed(123)

    x = torch.randn(4, 257, device="cuda", dtype=torch.float32) * 100.0

    custom_out = ext.softmax(x)

    torch_out = F.softmax(x, dim=-1)

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-5,
        atol=1e-6,
    )


def test_softmax_attention_scores_flattened(ext):
    torch.manual_seed(123)

    B = 2
    heads = 4
    T = 197

    scores = torch.randn(
        B,
        heads,
        T,
        T,
        device="cuda",
        dtype=torch.float32,
    ) * 100.0

    # Your CUDA softmax expects [rows, num_features].
    # For attention, each row is one query token's scores over all key tokens.
    x = scores.reshape(B * heads * T, T).contiguous()

    custom_out = ext.softmax(x)

    torch_out = torch.softmax(x, dim=-1)

    assert custom_out.shape == x.shape

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-5,
        atol=1e-6,
    )


def test_softmax_rows_sum_to_one(ext):
    torch.manual_seed(123)

    x = torch.randn(8, 197, device="cuda", dtype=torch.float32)

    custom_out = ext.softmax(x)

    row_sums = custom_out.sum(dim=-1)
    expected = torch.ones_like(row_sums)

    torch.testing.assert_close(
        row_sums,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_softmax_rejects_non_contiguous_input(ext):
    torch.manual_seed(123)

    # Shape is [8, 16], but layout is non-contiguous
    x = torch.randn(16, 8, device="cuda").transpose(0, 1)

    assert x.shape == (8, 16)
    assert not x.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.softmax(x)


def test_softmax_rejects_non_2d_input(ext):
    torch.manual_seed(123)

    x = torch.randn(2, 4, 197, device="cuda")

    with pytest.raises(RuntimeError, match="shape"):
        ext.softmax(x)
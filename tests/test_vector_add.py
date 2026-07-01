import pytest
import torch

from src.cuda_vit.ops.vector_add_ext import load_vector_add


@pytest.fixture(scope="session")
def ext():
    return load_vector_add()


@pytest.mark.parametrize("n", [1, 31, 255, 256, 257, 1_000_003])
def test_vector_add_matches_pytorch(ext, n):
    torch.manual_seed(123)

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    actual = ext.vector_add(a, b)
    expected = torch.add(a, b)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_vector_add_rejects_non_contiguous_input(ext):
    a = torch.randn(8, 8, device="cuda").t()
    b = torch.randn(8, 8, device="cuda")

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.vector_add(a, b)
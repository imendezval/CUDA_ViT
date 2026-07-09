import pytest
import torch
import torch.nn.functional as F

from cuda_vit.ops.linear_forward_ext import load_linear_forward


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.fixture(scope="session")
def ext():
    return load_linear_forward()


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        # tiny correctness cases
        (1, 1, 1),
        (1, 31, 7),

        # small normal cases
        (2, 64, 128),
        (8, 128, 512),

        # ViT-like projection dimensions
        (2, 768, 3072),

        # flattened ViT tokens:
        # B=2, T=197, C=768 -> rows = B*T = 394
        (394, 768, 3072),

        # smaller ViT-like project size:
        # B=8, T=197, C=384 -> rows = 1576
        (1576, 384, 1536),
    ],
)
def test_linear_matches_pytorch(
    ext,
    rows,
    in_features,
    out_features,
):
    torch.manual_seed(123)

    # Force PyTorch reference to use full FP32 math.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    x = torch.randn(
        rows,
        in_features,
        device="cuda",
        dtype=torch.float32,
    )

    W = torch.randn(
        out_features,
        in_features,
        device="cuda",
        dtype=torch.float32,
    )

    custom_out = ext.linear_forward(x, W)

    torch_out = F.linear(x, W, bias=None)

    assert custom_out.shape == (rows, out_features)

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-4,
        atol=1e-4,
    )


def test_linear_output_shape(ext):
    torch.manual_seed(123)

    rows = 2
    in_features = 768
    out_features = 3072

    x = torch.randn(rows, in_features, device="cuda")
    W = torch.randn(out_features, in_features, device="cuda")

    out = ext.linear_forward(x, W)

    assert out.shape == (rows, out_features)


def test_linear_rejects_non_contiguous_input(ext):
    torch.manual_seed(123)

    rows = 4
    in_features = 8
    out_features = 16

    # Shape is still [rows, in_features], but tensor is non-contiguous.
    x = torch.randn(in_features, rows, device="cuda").transpose(0, 1)

    assert x.shape == (rows, in_features)
    assert not x.is_contiguous()

    W = torch.randn(out_features, in_features, device="cuda")

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.linear_forward(x, W)


def test_linear_rejects_non_contiguous_weight(ext):
    torch.manual_seed(123)

    rows = 4
    in_features = 8
    out_features = 16

    x = torch.randn(rows, in_features, device="cuda")

    # Shape is still [out_features, in_features], but tensor is non-contiguous.
    W = torch.randn(in_features, out_features, device="cuda").transpose(0, 1)

    assert W.shape == (out_features, in_features)
    assert not W.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.linear_forward(x, W)


def test_linear_rejects_wrong_dtype(ext):
    torch.manual_seed(123)

    rows = 4
    in_features = 8
    out_features = 16

    x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
    W = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="float32"):
        ext.linear_forward(x, W)


def test_linear_rejects_wrong_shape(ext):
    torch.manual_seed(123)

    rows = 4
    in_features = 8
    out_features = 16

    x = torch.randn(rows, in_features, device="cuda")
    W = torch.randn(out_features, in_features + 1, device="cuda")

    with pytest.raises(RuntimeError, match="in_features"):
        ext.linear_forward(x, W)


def test_linear_numerical_error_report(ext):
    torch.manual_seed(0)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    rows = 2
    in_features = 768
    out_features = 3072

    x = torch.randn(rows, in_features, device="cuda")
    W = torch.randn(out_features, in_features, device="cuda")

    custom_out = ext.linear_forward(x, W)

    torch_out = F.linear(x, W, bias=None)

    max_abs_error = (custom_out - torch_out).abs().max().item()

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-4,
        atol=1e-4,
    )

    # Useful when running pytest -s
    print("rows:", rows)
    print("in_features:", in_features)
    print("out_features:", out_features)
    print("custom_out shape:", custom_out.shape)
    print("torch_out shape:", torch_out.shape)
    print("max absolute error:", max_abs_error)
import pytest
import torch
import torch.nn.functional as F

from cuda_vit.ops.patchembedding_ext import load_patchembedding


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.fixture(scope="session")
def ext():
    return load_patchembedding()


@pytest.mark.parametrize(
    "B,C,H,W_img,patch_size,emb_size",
    [
        (1, 1, 16, 16, 16, 1),
        (1, 3, 32, 32, 16, 8),
        (2, 3, 32, 32, 16, 64),
        (4, 3, 32, 32, 8, 64),
        (2, 3, 224, 224, 16, 384),  # ViT-like
    ],
)
def test_patchembedding_matches_pytorch(
    ext,
    B,
    C,
    H,
    W_img,
    patch_size,
    emb_size,
):
    torch.manual_seed(123)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    assert H % patch_size == 0
    assert W_img % patch_size == 0

    num_patch_el = C * patch_size * patch_size
    num_patches_h = H // patch_size
    num_patches_w = W_img // patch_size
    num_patches = num_patches_h * num_patches_w

    x = torch.randn(
        B,
        C,
        H,
        W_img,
        device="cuda",
        dtype=torch.float32,
    )

    W = torch.randn(
        emb_size,
        num_patch_el,
        device="cuda",
        dtype=torch.float32,
    )

    custom_out = ext.patchembedding(x, W)

    W_conv = W.view(emb_size, C, patch_size, patch_size)

    conv_out = F.conv2d(
        x,
        W_conv,
        bias=None,
        stride=patch_size,
    )

    torch_out = (
        conv_out
        .permute(0, 2, 3, 1)
        .reshape(B, num_patches, emb_size)
        .contiguous()
    )

    torch.testing.assert_close(
        custom_out,
        torch_out,
        rtol=1e-4,
        atol=1e-4,
    )


def test_patchembedding_rejects_non_contiguous_input(ext):
    torch.manual_seed(123)

    B = 2
    C = 3
    H = 32
    W_img = 32
    patch_size = 16
    emb_size = 64

    num_patch_el = C * patch_size * patch_size

    x = torch.randn(B, C, H, W_img, device="cuda").transpose(2, 3)

    W = torch.randn(
        emb_size,
        num_patch_el,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="contiguous"):
        ext.patchembedding(x, W)
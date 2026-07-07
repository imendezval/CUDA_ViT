import torch
import torch.nn.functional as F

from cuda_vit.ops.patchembedding_ext import load_patchembedding

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

# Force the PyTorch convolution reference to use full FP32 math
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

ext = load_patchembedding()

torch.manual_seed(0)

# B, C, H, W
B = 2
C = 3
H = 32
W_img = 32

# PatchEmbed config
patch_size = 16
emb_size = 64

assert H % patch_size == 0
assert W_img % patch_size == 0

num_patch_el = C * patch_size * patch_size
num_patches_h = H // patch_size
num_patches_w = W_img // patch_size
num_patches = num_patches_h * num_patches_w

x = torch.randn(
    B, C, H, W_img,
    device="cuda",
    dtype=torch.float32,
)

# W: [emb_size, C * patch_size * patch_size]
W = torch.randn(
    emb_size, num_patch_el,
    device="cuda",
    dtype=torch.float32,
)

# CUDA extension
custom_out = ext.patchembedding(x, W)

# PyTorch equivalent
W_conv = W.view(emb_size, C, patch_size, patch_size)

# conv_out: [B, emb_size, num_patches_h, num_patches_w]
conv_out = F.conv2d(
    x,
    W_conv,
    bias=None,
    stride=patch_size,
)

# Convert to B, T, D
torch_out = (
    conv_out
    .permute(0, 2, 3, 1)   # [B, patches_h, patches_w, emb_size]
    .reshape(B, num_patches, emb_size)
    .contiguous()
)

torch.testing.assert_close(
    custom_out,
    torch_out,
    rtol=1e-4,
    atol=1e-4,
)

print("Success.")
print("Input shape:", x.shape)
print("Weight shape:", W.shape)
print("Custom output shape:", custom_out.shape)
print("Expected output shape:", torch_out.shape)
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First embedding values:", custom_out[0, 0, :5].cpu())
import torch
import torch.nn.functional as F

from cuda_vit.ops.mlp_linear_gelu_ext import (
    load_fused_mlp_linear_gelu,
)

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

# Force the PyTorch linear reference to use full FP32 math
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

ext = load_fused_mlp_linear_gelu()

torch.manual_seed(0)

# B, F (= in_features)
# In ViT: [B, T, C] -> [B * T, C]
B = 2
in_features = 768
out_features = 3072

x = torch.randn(
    B, in_features,
    device="cuda",
    dtype=torch.float32,
)

# W: [out_features, in_features]
W = torch.randn(
    out_features, in_features,
    device="cuda",
    dtype=torch.float32,
)

# b: [out_features]
b = torch.randn(
    out_features,
    device="cuda",
    dtype=torch.float32,
)

# CUDA extension
custom_out = ext.fused_MLPlinear_GELU(x, W, b)

# PyTorch equivalent
linear_out = F.linear(
    x,
    W,
    b,
)

# Must use because CUDA kernel uses gelu_tanh
torch_out = F.gelu(
    linear_out,
    approximate="tanh",
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
print("Bias shape:", b.shape)
print("Custom output shape:", custom_out.shape)
print("Expected output shape:", torch_out.shape)
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First output values:", custom_out[0, :5].cpu())
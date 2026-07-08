import torch

from cuda_vit.ops.softmax_ext import load_softmax

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

ext = load_softmax()

torch.manual_seed(0)

# B, heads, T, T
B = 2
heads = 4
T = 197

scores = torch.randn(
    B, heads, T, T,
    device="cuda",
    dtype=torch.float32,
) * 100

# Convert to 2D rows for CUDA extension
x = scores.reshape(B * heads * T, T).contiguous()

# CUDA extension
custom_out = ext.softmax(x)

# PyTorch equivalent
torch_out = torch.softmax(x, dim=-1)

torch.testing.assert_close(
    custom_out,
    torch_out,
    rtol=1e-5,
    atol=1e-6,
)

print("Success.")
print("Input shape:", x.shape)
print("Custom output shape:", custom_out.shape)
print("Expected output shape:", torch_out.shape)
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First softmax values:", custom_out[0, :5].cpu())
print("First row sum:", custom_out[0].sum().item())
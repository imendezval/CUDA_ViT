import torch

from cuda_vit.ops.scaled_qk_ext import load_scaled_qk


assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

# Force PyTorch matmul reference to use full FP32 math
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

ext = load_scaled_qk()

torch.manual_seed(0)

# ViT attention shape
B = 2
H = 3          # num_heads
T = 197        # num_tokens
Dh = 64        # head_dim

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

# CUDA extension
custom_out = ext.scaled_qk(Q, K)

# PyTorch equivalent:
# Q: [B, H, T, Dh]
# K.transpose(-2, -1): [B, H, Dh, T]
# output: [B, H, T, T]
torch_out = torch.matmul(
    Q,
    K.transpose(-2, -1),
) / (Dh ** 0.5)

torch.cuda.synchronize()

torch.testing.assert_close(
    custom_out,
    torch_out,
    rtol=1e-4,
    atol=1e-4,
)

print("Success.")
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("Custom output shape:", custom_out.shape)
print("Expected output shape:", torch_out.shape)
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First output values:", custom_out[0, 0, 0, :5].cpu())
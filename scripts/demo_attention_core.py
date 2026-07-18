import math

import torch
import torch.nn.functional as F

from cuda_vit.ops.flashattention_ext import load_flashattention


assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

# Force PyTorch reference toward full FP32 math.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

ext = load_flashattention()

torch.manual_seed(0)

# Current kernel requirements:
# - Dh == 64
# - T divisible by Br=16 and Bc=32
B = 2
H = 3
T = 192
Dh = 64

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

V = torch.randn(
    B, H, T, Dh,
    device="cuda",
    dtype=torch.float32,
)

Q = Q.contiguous()
K = K.contiguous()
V = V.contiguous()

# Custom fused FlashAttention-style kernel.
# Rename this function if your binding uses a different name.
custom_out = ext.FlashAttention(Q, K, V)

# Explicit PyTorch reference:
#
# Q:                       [B, H, T, Dh]
# K.transpose(-2, -1):     [B, H, Dh, T]
# scores:                  [B, H, T, T]
# probabilities:           [B, H, T, T]
# output:                  [B, H, T, Dh]
scale = 1.0 / math.sqrt(Dh)

scores = torch.matmul(
    Q,
    K.transpose(-2, -1),
) * scale

probabilities = F.softmax(scores, dim=-1)

torch_out = torch.matmul(
    probabilities,
    V,
)

torch.cuda.synchronize()

torch.testing.assert_close(
    custom_out,
    torch_out,
    rtol=1e-4,
    atol=1e-4,
)

absolute_error = (custom_out - torch_out).abs()

print("Success.")
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("Custom output shape:", custom_out.shape)
print("Expected output shape:", torch_out.shape)
print("Maximum absolute error:", absolute_error.max().item())
print("Mean absolute error:", absolute_error.mean().item())
print("First custom values:", custom_out[0, 0, 0, :5].cpu())
print("First expected values:", torch_out[0, 0, 0, :5].cpu())
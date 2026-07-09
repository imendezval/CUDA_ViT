import torch
import torch.nn.functional as F

from cuda_vit.ops.attention_v_ext import load_attention_v


assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

# Force PyTorch reference to use full FP32 math
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

ext = load_attention_v()

torch.manual_seed(0)

# ViT attention shape
B = 2
H = 3          # num_heads
T = 197        # num_tokens
Dh = 64        # head_dim

attention_scores = torch.randn(
    B, H, T, T,
    device="cuda",
    dtype=torch.float32,
)

# Usually already softmaxed attention probs
attention_scores = F.softmax(attention_scores, dim=-1).contiguous()

V = torch.randn(
    B, H, T, Dh,
    device="cuda",
    dtype=torch.float32,
)

# CUDA extension
custom_out = ext.attention_v(attention_scores, V)

# PyTorch equivalent:
# attention_scores: [B, H, T, T]
# V:                [B, H, T, Dh]
# output:           [B, H, T, Dh]
torch_out = torch.matmul(attention_scores, V)

torch.cuda.synchronize()

torch.testing.assert_close(
    custom_out,
    torch_out,
    rtol=1e-4,
    atol=1e-4,
)

print("Success.")
print("attention_scores shape:", attention_scores.shape)
print("V shape:", V.shape)
print("Custom output shape:", custom_out.shape)
print("Expected output shape:", torch_out.shape)
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First output values:", custom_out[0, 0, 0, :5].cpu())
import torch
import torch.nn.functional as F

from cuda_vit.ops.layernorm_ext import load_layernorm

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

ext = load_layernorm()

torch.manual_seed(0)

# B, C, T
B = 8
T = 197
D = 384
eps = 1e-5

x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
gamma = torch.randn(D, device="cuda", dtype=torch.float32)
beta = torch.randn(D, device="cuda", dtype=torch.float32)

# CUDA extension
custom_out = ext.layernorm(x, gamma, beta, eps)

# PyTorch reference: normalize only the final dim D
torch_out = F.layer_norm(
    x,
    normalized_shape=(D,),
    weight=gamma,
    bias=beta,
    eps=eps,
)

torch.testing.assert_close(
    custom_out,
    torch_out,
    rtol=1e-5,
    atol=1e-5,
)

print("Success.")
print("Input shape:", x.shape)
print("Gamma/beta shape:", gamma.shape)
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First five output values:", custom_out[0, 0, :5].cpu())
import torch

from cuda_vit.ops.vector_add_ext import load_vector_add

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU."

ext = load_vector_add()

torch.manual_seed(0)

a = torch.randn(1_000_000, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)

custom_out = ext.vector_add(a, b)
torch_out = a + b

torch.testing.assert_close(custom_out, torch_out, rtol=1e-6, atol=1e-6)

print("Success.")
print("Max absolute error:", (custom_out - torch_out).abs().max().item())
print("First five values:", custom_out[:5].cpu())
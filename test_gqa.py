import sys, os, warnings
os.environ["VKFLAME_DEBUG"] = "0"
sys.path.insert(0, "python")
sys.path.insert(0, ".")
warnings.simplefilter("always")

import torch
import torch.nn.functional as F
import vkflame
from torch_backend.mode import VKFlameMode

# Small GQA: Q=(1,4,16,32), K/V=(1,2,16,32)
Q = torch.randn(1, 4, 16, 32, dtype=torch.float16, device="cuda")
K = torch.randn(1, 2, 16, 32, dtype=torch.float16, device="cuda")
V = torch.randn(1, 2, 16, 32, dtype=torch.float16, device="cuda")
print("tensors is_cuda:", Q.is_cuda, K.is_cuda, V.is_cuda)

import torch._C
try:
    with VKFlameMode():
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    print("GQA (small) vkf OK shape:", out.shape)
except Exception as e:
    print(f"GQA (small) EXCEPTION: {type(e).__name__}: {e}")

# Compare with expanded ROCm reference
Kr = K.repeat_interleave(2, dim=1)
Vr = V.repeat_interleave(2, dim=1)
ref = F.scaled_dot_product_attention(Q.float(), Kr.float(), Vr.float()).half()
print("ROCm ref shape:", ref.shape)

# Test the handler _directly_ to see what error it raises
try:
    from torch_backend.ops import _sdpa_handler
    result = _sdpa_handler(Q, K, V, is_causal=False)
    print("_sdpa_handler direct OK:", result.shape)
except Exception as e:
    print(f"_sdpa_handler direct EXCEPTION: {type(e).__name__}: {e}")

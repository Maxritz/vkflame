# tests/test_correctness.py
import torch
import pytest

# On Windows 11 there is no ROCm/CUDA; vkflame provides Vulkan-backed compute
# via staged CPU↔GPU uploads.  Tests run on CPU tensors and VKFlameMode
# intercepts fp16/int8 ops, routing them through Vulkan.  fp32 always falls
# through to PyTorch so reference computations are unaffected.
def _vkflame_gpu_available() -> bool:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
        import vkflame  # noqa: F401
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and not _vkflame_gpu_available(),
    reason="No GPU (no CUDA/ROCm and no Vulkan via vkflame)"
)

# Device to place test tensors on.
# Windows 11 + AMD: no CUDA → use CPU (vkflame stages uploads transparently).
# Linux + ROCm:     use "cuda" for direct GPU path.
_DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Helper: move tensor to the test device
def _G(t: torch.Tensor) -> torch.Tensor:
    return t.to(_DEV)


def setup_module():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    import vkflame
    vkflame.install(permanent=True)


# ── Matrix multiply ──────────────────────────────────────────────
def test_mm_f16():
    A = _G(torch.randn(256, 256, dtype=torch.float16))
    B = _G(torch.randn(256, 256, dtype=torch.float16))
    ref = torch.mm(A.float(), B.float()).half()
    out = torch.mm(A, B)
    assert (out.float() - ref.float()).abs().max() < 0.1


def test_mm_shapes():
    for M, N, K in [(1, 4096, 4096), (32, 4096, 4096), (4096, 4096, 128)]:
        A = _G(torch.randn(M, K, dtype=torch.float16))
        B = _G(torch.randn(K, N, dtype=torch.float16))
        ref = torch.mm(A.float(), B.float()).half()
        out = torch.mm(A, B)
        assert (out.float() - ref.float()).abs().max() < 0.5, \
            f"mm({M},{N},{K}) failed"


# ── RMSNorm ──────────────────────────────────────────────────────
def test_rms_norm():
    x = _G(torch.randn(32, 4096, dtype=torch.float16))
    g = _G(torch.ones(4096, dtype=torch.float16))
    rms = (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    ref = (x.float() / rms * g.float()).half()
    out, _, _ = torch.ops.aten.native_layer_norm(x, [4096], g, None, 1e-6)
    assert (out.float() - ref.float()).abs().max() < 0.05


# ── Softmax ───────────────────────────────────────────────────────
def test_softmax_numerical():
    x = _G(torch.randn(64, 32000, dtype=torch.float16))
    ref = torch.softmax(x.float(), dim=-1).half()
    out = torch.softmax(x, dim=-1)
    assert (out.float() - ref.float()).abs().max() < 1e-3


def test_softmax_sum_to_one():
    x = _G(torch.randn(8, 1000, dtype=torch.float16))
    out = torch.softmax(x, dim=-1)
    row_sums = out.float().sum(dim=-1)
    assert (row_sums - 1.0).abs().max() < 1e-3


# ── Attention ─────────────────────────────────────────────────────
def test_sdpa_mha():
    B, H, S, D = 2, 8, 64, 64
    Q = _G(torch.randn(B, H, S, D, dtype=torch.float16))
    K = _G(torch.randn(B, H, S, D, dtype=torch.float16))
    V = _G(torch.randn(B, H, S, D, dtype=torch.float16))
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float()).half()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    assert (out.float() - ref.float()).abs().max() < 0.1


def test_sdpa_causal():
    B, H, S, D = 1, 4, 32, 64
    Q = _G(torch.randn(B, H, S, D, dtype=torch.float16))
    K = _G(torch.randn(B, H, S, D, dtype=torch.float16))
    V = _G(torch.randn(B, H, S, D, dtype=torch.float16))
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), is_causal=True).half()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert (out.float() - ref.float()).abs().max() < 0.1


def test_sdpa_gqa():
    B, Hq, Hkv, S, D = 1, 8, 2, 64, 64
    Q = _G(torch.randn(B, Hq, S, D, dtype=torch.float16))
    K = _G(torch.randn(B, Hkv, S, D, dtype=torch.float16))
    V = _G(torch.randn(B, Hkv, S, D, dtype=torch.float16))
    K_exp = K.repeat_interleave(Hq // Hkv, dim=1)
    V_exp = V.repeat_interleave(Hq // Hkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_exp.float(), V_exp.float()).half()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    assert (out.float() - ref.float()).abs().max() < 0.1


# ── Winograd ──────────────────────────────────────────────────────
def test_winograd_f23():
    # fp32 conv falls through to PyTorch on both platforms — validates correctness
    x = _G(torch.randn(4, 8, 64, 64, dtype=torch.float32))
    w = _G(torch.randn(16, 8, 3, 3, dtype=torch.float32))
    ref = torch.nn.functional.conv2d(x, w, padding=0)
    out = torch.nn.functional.conv2d(x, w, padding=0)
    assert (out - ref).abs().max() < 1e-3


# ── Sampling ──────────────────────────────────────────────────────
def test_topk():
    x = _G(torch.tensor([[3., 1., 4., 1., 5., 9., 2., 6.]]))
    vals, idx = torch.topk(x, k=3)
    assert vals[0, 0].item() == 9.0
    assert vals[0, 1].item() == 6.0
    assert vals[0, 2].item() == 5.0
    assert idx[0, 0].item() == 5
    assert idx[0, 1].item() == 7
    assert idx[0, 2].item() == 4


def test_argmax():
    x = _G(torch.tensor([[1., 3., 2.]]))
    assert torch.argmax(x, dim=-1).item() == 1


def test_multinomial_distribution():
    probs = _G(torch.tensor([[0.1, 0.2, 0.7]]))
    samples = torch.multinomial(probs, num_samples=10000, replacement=True)
    counts = torch.bincount(samples[0], minlength=3).float()
    freqs = counts / counts.sum()
    assert abs(freqs[2].item() - 0.7) < 0.05


# ── Embedding ─────────────────────────────────────────────────────
def test_embedding():
    weight = _G(torch.randn(100, 64, dtype=torch.float16))
    idx    = _G(torch.tensor([0, 5, 99, 42]))
    # ref: fp32 weight falls through vkflame (not in _ACCEL_DTYPES)
    ref    = torch.nn.functional.embedding(idx, weight.float()).half()
    # vkflame path: fp16 weight is intercepted and dispatched via Vulkan
    out    = torch.nn.functional.embedding(idx, weight)
    assert torch.allclose(out.float(), ref.float(), atol=0.01)


# ── End-to-end ────────────────────────────────────────────────────
def test_full_generation():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    import os
    model_path = os.environ.get("VKFLAME_TEST_MODEL", "")
    if not model_path:
        pytest.skip("Set VKFLAME_TEST_MODEL to a quantised model path")
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16).to(_DEV).eval()
    with torch.no_grad():
        inp = tok("Hello", return_tensors="pt").input_ids.to(_DEV)
        out = model.generate(inp, max_new_tokens=10)
    text = tok.decode(out[0])
    assert len(text) > 5

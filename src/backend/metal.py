"""Metal backend: hand-written MSL kernels (src/kernels/*.metal) compiled at runtime and dispatched on the GPU.

Decode (one token) runs entirely through custom kernels: matvec, RMSNorm, RoPE, attention, SwiGLU.
Prefill (many tokens) keeps PyTorch's tuned GEMM and attention for the matrix-matrix work.
"""
from pathlib import Path

import torch

from backend.torch_ref import TorchBackend

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
TG = 256                                                   # threads per threadgroup for row-parallel kernels


def load_library():
    """Concatenate every .metal file and compile it once (runtime compiler; no Xcode needed)."""
    src = "\n".join(p.read_text() for p in sorted(KERNEL_DIR.glob("*.metal")))
    return torch.mps.compile_shader(src)


class MetalBackend(TorchBackend):
    def __init__(self):
        super().__init__("mps", torch.bfloat16)
        self.name = "metal-kernels-bf16"
        self.lib = load_library()
        self._no_bias = torch.zeros(1, dtype=torch.bfloat16, device=self.device)
        self._no_res = torch.zeros(1, device=self.device)

    def linear(self, x, w, b=None, residual=None):
        if x.shape[0] != 1 or w.dtype != torch.bfloat16:
            return super().linear(x, w, b, residual)       # prefill: tuned GEMM
        N, K = w.shape
        y = torch.empty(1, N, device=self.device)
        res = residual.float().contiguous() if residual is not None else self._no_res
        self.lib.matvec_bf16(y, w, x.float().contiguous(), b if b is not None else self._no_bias, res,
                             K, N, int(b is not None), int(residual is not None), threads=N * 32, group_size=TG)
        return y

    def gemm(self, x, w):
        """Tiled-GEMM kernel for prefill (kept for study/benchmarks; linear() uses PyTorch's tuned GEMM)."""
        x = x.float().contiguous()
        T, K = x.shape
        N = w.shape[0]
        y = torch.empty(T, N, device=self.device)
        gx, gy = -(-N // 16), -(-T // 16)
        self.lib.gemm_bf16(y, x, w, T, N, K, threads=(gx * 16, gy * 16), group_size=(16, 16))
        return y

    def swiglu(self, x, w_gate_up):
        if x.shape[0] != 1 or w_gate_up.dtype != torch.bfloat16:
            return super().swiglu(x, w_gate_up)
        F, K = w_gate_up.shape[0] // 2, w_gate_up.shape[1]
        out = torch.empty(1, F, device=self.device)
        self.lib.matvec_swiglu_bf16(out, w_gate_up, x.float().contiguous(), K, F, threads=F * 32, group_size=TG)
        return out

    def rms_norm(self, x, w, eps):
        x = x.float().contiguous()
        out = torch.empty_like(x)
        d = x.shape[-1]
        rows = x.numel() // d
        self.lib.rms_norm(out, x, w, float(eps), d, threads=rows * TG, group_size=TG)
        return out

    def rope(self, x, positions, theta):
        x = x.float().contiguous()
        T, H, d = x.shape
        out = torch.empty_like(x)
        n = T * H * (d // 2)
        self.lib.rope(out, x, positions.to(device=self.device, dtype=torch.int32), float(theta), H, d, n,
                      threads=n, group_size=min(TG, n))
        return out

    def attention(self, q, k, v, causal=True):
        if q.shape[0] != 1:
            return super().attention(q, k, v, causal)       # prefill: many queries, masked
        _, Hq, d = q.shape
        S, Hkv, _ = k.shape
        out = torch.empty(1, Hq, d, device=self.device)
        scores = torch.empty(Hq, S, device=self.device)
        self.lib.attention_decode(out, q.float().contiguous(), k.float().contiguous(), v.float().contiguous(),
                                  scores, S, Hkv, Hq // Hkv, d, threads=Hq * TG, group_size=TG)
        return out

    def silu_mul(self, gate, up):
        gate, up = gate.float().contiguous(), up.float().contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        self.lib.silu_mul(out, gate, up, n, threads=n, group_size=min(TG, n))
        return out

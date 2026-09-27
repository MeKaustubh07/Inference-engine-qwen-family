"""Reference backend: the hand-written ops from ops.py, on CPU in fp32 (the oracle) or on the GPU (MPS)."""
import torch

import ops


class TorchBackend:
    """device='cpu', weight_dtype=float32  -> exact reference path used for HF comparisons.
    device='mps', weight_dtype=bfloat16    -> fast path: bf16 weights resident on the GPU, fp32 activations."""

    def __init__(self, device: str = "cpu", weight_dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.name = f"torch-{self.device.type}-{str(weight_dtype).removeprefix('torch.')}"

    def prepare(self, w: torch.Tensor) -> torch.Tensor:
        return w.to(device=self.device, dtype=self.weight_dtype)

    def linear(self, x, w, b=None):
        y = (x.to(w.dtype) @ w.T).float()
        return y + b.float() if b is not None else y

    def rms_norm(self, x, w, eps):
        return ops.rms_norm(x, w, eps)

    def rope(self, x, positions, theta):
        return ops.rope(x, positions, theta)

    def attention(self, q, k, v, causal=True):
        return ops.attention(q, k, v, causal)

    def silu_mul(self, gate, up):
        return ops.silu_mul(gate, up)

    def sync(self) -> None:
        """Wait for queued GPU work (needed for honest timing on MPS)."""
        if self.device.type == "mps":
            torch.mps.synchronize()

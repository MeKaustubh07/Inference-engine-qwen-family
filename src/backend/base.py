"""The backend protocol: the fixed set of operations the model is built from.

The model only ever calls these. Swapping the implementation (fp32 CPU reference, bf16 on the GPU,
hand-written Metal kernels, quantized weights) never touches model code.
"""
from typing import Protocol

import torch


class Backend(Protocol):
    name: str
    device: torch.device

    def prepare(self, w: torch.Tensor) -> torch.Tensor:
        """Move/convert one weight tensor into this backend's resident format (called once per weight)."""

    def linear(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None,
               residual: torch.Tensor | None = None) -> torch.Tensor:
        """x [T, K] @ w[N, K].T (+ b) (+ residual) -> [T, N] in fp32. T == 1 is the decode matvec."""

    def rms_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """[..., D] -> [..., D]"""

    def rope(self, x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
        """x [T, H, d] rotated by absolute positions [T]."""

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """q [T, Hq, d], k/v [S, Hkv, d] -> [T, Hq, d]; query i sits at position S - T + i."""

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """silu(gate) * up"""

    def swiglu(self, x: torch.Tensor, w_gate_up: torch.Tensor) -> torch.Tensor:
        """w_gate_up = [gate; up] stacked [2F, K]: returns silu(x @ gate.T) * (x @ up.T), [T, F]."""

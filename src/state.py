"""Per-sequence decode state: the KV cache.

Keys and values of past tokens never change, so each layer stores them once and every later
step reads them back instead of recomputing the whole prefix (O(n^3) total work -> O(n^2)).
"""
import torch


class ContiguousKVCache:
    """One preallocated buffer per layer: k/v [max_len, n_kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_len: int,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.max_len = max_len
        self.k = torch.zeros(n_layers, max_len, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0                                   # tokens committed so far

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k/v [T, Hkv, d] for positions start .. start+T-1."""
        end = start + k.shape[0]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, capacity {self.max_len}")
        self.k[layer, start:end] = k.to(self.k.dtype)
        self.v[layer, start:end] = v.to(self.v.dtype)

    def read(self, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """All keys/values for positions 0 .. end-1 (views, no copy)."""
        return self.k[layer, :end], self.v[layer, :end]

    def advance(self, n: int) -> None:
        self.length += n

    def bytes_used(self) -> int:
        return 2 * self.k[:, : self.length].numel() * self.k.element_size()

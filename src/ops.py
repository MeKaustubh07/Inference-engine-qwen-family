"""Hand-written ops on torch primitives. No torch.nn, no torch.nn.functional model ops."""
import math

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: scale each token's vector to unit root-mean-square, then per-channel weight.

    x:      [..., D]   the residual stream (one row per token)
    weight: [D]        learned per-channel volume knobs
    """
    x32 = x.float()                                     # do the math in fp32 for accuracy
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 / rms) * weight.float()                 # broadcast weight across tokens


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary position embedding: rotate pairs of numbers by a position-dependent angle.

    x:         [T, H, d]  queries or keys, split into H heads of d numbers
    positions: [T]        position of each token in the sequence (0, 1, 2, ...)
    theta:     base frequency from the config (1e6 for Qwen2.5)

    Pairing convention (Qwen2 / HF "rotate_half"): number i pairs with number i + d/2.
    """
    d = x.shape[-1]
    half = d // 2
    # One rotation speed per pair: fast for the first pairs, very slow for the last.
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))  # [d/2]
    angles = positions.float()[:, None] * freqs[None, :]                          # [T, d/2]
    cos = torch.cos(angles)[:, None, :]                                           # [T, 1, d/2]
    sin = torch.sin(angles)[:, None, :]                                           # broadcast over heads

    x32 = x.float()
    a, b = x32[..., :half], x32[..., half:]         # the two members of every pair
    return torch.cat((a * cos - b * sin,            # standard 2-D rotation of (a, b)
                      a * sin + b * cos), dim=-1)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Turn scores into weights that are positive and sum to 1 along `dim`."""
    x32 = x.float()
    m = x32.max(dim=dim, keepdim=True).values      # subtract the max first: exp() of big numbers overflows
    e = torch.exp(x32 - m)
    return e / e.sum(dim=dim, keepdim=True)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Grouped-query attention.

    q:    [T, Hq, d]   queries for the T new tokens (already rotated by RoPE)
    k, v: [S, Hkv, d]  keys/values for all S tokens seen so far (S == T until the KV cache exists)
    returns [T, Hq, d]
    """
    T, Hq, d = q.shape
    S, Hkv, _ = k.shape
    group = Hq // Hkv                                   # 14 // 2 = 7 query heads share each key/value head
    k = k.float().repeat_interleave(group, dim=1)       # [S, Hq, d]: KV head 0 serves query heads 0-6, head 1 serves 7-13
    v = v.float().repeat_interleave(group, dim=1)

    qh, kh, vh = q.float().transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)   # heads first: [H, tokens, d]
    scores = (qh @ kh.transpose(1, 2)) / math.sqrt(d)   # [Hq, T, S]: every query dotted with every key

    if causal:
        # query i sits at absolute position S - T + i and may only see keys at positions <= that
        future = torch.ones(T, S, dtype=torch.bool).triu(diagonal=S - T + 1)
        scores = scores.masked_fill(future, float("-inf"))   # exp(-inf) = 0 -> zero weight

    weights = softmax(scores, dim=-1)                   # [Hq, T, S]: each row sums to 1
    out = weights @ vh                                  # [Hq, T, d]: weighted mix of values
    return out.transpose(0, 1)                          # back to [T, Hq, d]


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU core: silu(gate) * up, where silu(g) = g * sigmoid(g) = g / (1 + e^-g)."""
    g = gate.float()
    return (g / (1.0 + torch.exp(-g))) * up.float()

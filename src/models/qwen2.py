"""Qwen2.5 model assembled from the loader's tensors."""
import torch

from config import ModelConfig
from ops import attention, rms_norm, rope, silu_mul
from weight_loader import SafetensorsFile


class Qwen2Model:
    def __init__(self, config: ModelConfig, weights: SafetensorsFile):
        self.config = config
        self.weights = weights
        self.embed_table = weights.get("model.embed_tokens.weight")   # [vocab, hidden], bf16 view

    def _w(self, name: str) -> torch.Tensor:
        """Fetch a weight by name and upcast it for the fp32 reference path."""
        return self.weights.get(name).float()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids [T] -> vectors [T, hidden]: row lookup in the embedding table."""
        return self.embed_table[ids].float()

    def self_attn(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """Attention half of one station. x: normalized stream [T, hidden] -> [T, hidden]."""
        cfg = self.config
        p = f"model.layers.{layer}.self_attn."
        T = x.shape[0]

        # 1. Projections: every token's row -> a query, a key, a value (Qwen adds biases to all three)
        q = x @ self._w(p + "q_proj.weight").T + self._w(p + "q_proj.bias")   # [T, 896]
        k = x @ self._w(p + "k_proj.weight").T + self._w(p + "k_proj.bias")   # [T, 128]
        v = x @ self._w(p + "v_proj.weight").T + self._w(p + "v_proj.bias")   # [T, 128]

        # 2. Split into heads of 64 numbers each
        q = q.view(T, cfg.num_attention_heads, cfg.head_dim)    # [T, 14, 64]
        k = k.view(T, cfg.num_key_value_heads, cfg.head_dim)    # [T,  2, 64]
        v = v.view(T, cfg.num_key_value_heads, cfg.head_dim)    # [T,  2, 64]

        # 3. Positions: rotate queries and keys (values are never rotated)
        positions = torch.arange(T)
        q = rope(q, positions, cfg.rope_theta)
        k = rope(k, positions, cfg.rope_theta)

        # 4. Attention: score, mask the future, softmax, mix values
        out = attention(q, k, v, causal=True)                  # [T, 14, 64]

        # 5. Merge heads and project back into the residual stream (no bias here)
        return out.reshape(T, -1) @ self._w(p + "o_proj.weight").T   # [T, 896]

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward half of one station (SwiGLU). x: normalized stream [T, hidden] -> [T, hidden]."""
        p = f"model.layers.{layer}.mlp."
        gate = x @ self._w(p + "gate_proj.weight").T      # [T, 4864]  decides how much of each feature passes
        up = x @ self._w(p + "up_proj.weight").T          # [T, 4864]  the features themselves
        return silu_mul(gate, up) @ self._w(p + "down_proj.weight").T   # [T, 896] back to stream width

    def block(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """One full station: pre-norm attention and pre-norm MLP, each added back into the residual stream."""
        eps = self.config.rms_norm_eps
        p = f"model.layers.{layer}."
        h = h + self.self_attn(layer, rms_norm(h, self._w(p + "input_layernorm.weight"), eps))
        h = h + self.mlp(layer, rms_norm(h, self._w(p + "post_attention_layernorm.weight"), eps))
        return h

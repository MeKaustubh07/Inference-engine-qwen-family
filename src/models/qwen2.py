"""Qwen2.5 model assembled from the loader's tensors, running on a pluggable backend."""
import torch

from backend.torch_ref import TorchBackend
from config import ModelConfig
from state import ContiguousKVCache
from weight_loader import SafetensorsFile


class Qwen2Model:
    def __init__(self, config: ModelConfig, weights: SafetensorsFile, backend=None):
        self.config = config
        self.weights = weights
        self.b = backend or TorchBackend()                              # default: fp32 CPU reference
        self.embed_table = weights.get("model.embed_tokens.weight")   # [vocab, hidden], bf16 view
        self._cache: dict[str, torch.Tensor] = {}

    def _w(self, name: str) -> torch.Tensor:
        """Fetch a weight by name, convert it once into the backend's resident format, and keep it."""
        if name not in self._cache:
            self._cache[name] = self.b.prepare(self.weights.get(name))
        return self._cache[name]

    def new_state(self, max_len: int) -> ContiguousKVCache:
        cfg = self.config
        return ContiguousKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, max_len,
                                 device=self.b.device, dtype=torch.float32)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids [T] -> vectors [T, hidden]: row lookup in the embedding table."""
        return self.embed_table[ids.cpu()].float().to(self.b.device)

    def self_attn(self, layer: int, x: torch.Tensor, positions: torch.Tensor | None = None,
                  state: ContiguousKVCache | None = None) -> torch.Tensor:
        """Attention half of one station. x: normalized stream [T, hidden] -> [T, hidden].

        positions: absolute positions of the T tokens (default 0..T-1).
        state:     KV cache; the new keys/values are written at `positions` and attention reads
                   every cached position up to the last new one.
        """
        cfg, b = self.config, self.b
        p = f"model.layers.{layer}.self_attn."
        T = x.shape[0]
        if positions is None:
            positions = torch.arange(T, device=b.device)

        # 1. Projections (Qwen adds biases to q, k and v)
        q = b.linear(x, self._w(p + "q_proj.weight"), self._w(p + "q_proj.bias"))   # [T, 896]
        k = b.linear(x, self._w(p + "k_proj.weight"), self._w(p + "k_proj.bias"))   # [T, 128]
        v = b.linear(x, self._w(p + "v_proj.weight"), self._w(p + "v_proj.bias"))   # [T, 128]

        # 2. Split into heads
        q = q.view(T, cfg.num_attention_heads, cfg.head_dim)
        k = k.view(T, cfg.num_key_value_heads, cfg.head_dim)
        v = v.view(T, cfg.num_key_value_heads, cfg.head_dim)

        # 3. Rotate queries and keys by position
        q = b.rope(q, positions, cfg.rope_theta)
        k = b.rope(k, positions, cfg.rope_theta)

        # 4. KV cache: store the new keys/values, then attend over everything cached so far
        if state is not None:
            start = int(positions[0])
            state.write(layer, start, k, v)
            k, v = state.read(layer, start + T)

        out = b.attention(q, k, v, causal=True)                   # [T, 14, 64]
        return b.linear(out.reshape(T, -1), self._w(p + "o_proj.weight"))   # no bias on o_proj

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward half of one station (SwiGLU). x: normalized stream [T, hidden] -> [T, hidden]."""
        b, p = self.b, f"model.layers.{layer}.mlp."
        gate = b.linear(x, self._w(p + "gate_proj.weight"))      # [T, 4864]
        up = b.linear(x, self._w(p + "up_proj.weight"))          # [T, 4864]
        return b.linear(b.silu_mul(gate, up), self._w(p + "down_proj.weight"))

    def block(self, layer: int, h: torch.Tensor, positions: torch.Tensor | None = None,
              state: ContiguousKVCache | None = None) -> torch.Tensor:
        """One full station: pre-norm attention and pre-norm MLP, each added back into the residual stream."""
        eps, b, p = self.config.rms_norm_eps, self.b, f"model.layers.{layer}."
        h = h + self.self_attn(layer, b.rms_norm(h, self._w(p + "input_layernorm.weight"), eps), positions, state)
        h = h + self.mlp(layer, b.rms_norm(h, self._w(p + "post_attention_layernorm.weight"), eps))
        return h

    def forward(self, ids: torch.Tensor, state: ContiguousKVCache | None = None,
                last_only: bool = False) -> torch.Tensor:
        """Token ids [T] -> logits [T, vocab] (or [1, vocab] with last_only).

        Without a state the whole sequence is processed from position 0.
        With a state, the tokens continue from state.length and their keys/values are cached.
        """
        start = state.length if state is not None else 0
        positions = torch.arange(start, start + len(ids), device=self.b.device)
        h = self.embed(ids)
        for layer in range(self.config.num_hidden_layers):
            h = self.block(layer, h, positions, state)
        if state is not None:
            state.advance(len(ids))
        if last_only:
            h = h[-1:]
        h = self.b.rms_norm(h, self._w("model.norm.weight"), self.config.rms_norm_eps)
        # Tied embeddings: the table that turned ids into vectors now scores vectors against every token.
        return self.b.linear(h, self._w("model.embed_tokens.weight"))

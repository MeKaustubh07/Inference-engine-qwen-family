"""Qwen2.5 model assembled from the loader's tensors, running on a pluggable backend."""
import torch

from backend.torch_ref import TorchBackend
from config import ModelConfig
from quant import concat_rows
from state import ContiguousKVCache, PagedKVPool
from weight_loader import SafetensorsFile


class Qwen2Model:
    def __init__(self, config: ModelConfig, weights: SafetensorsFile, backend=None):
        self.config = config
        self.weights = weights
        self.b = backend or TorchBackend()                              # default: fp32 CPU reference
        self._cache: dict[str, torch.Tensor] = {}

    def _w(self, name: str) -> torch.Tensor:
        """Fetch a weight by name, convert it once into the backend's resident format, and keep it."""
        if name not in self._cache:
            self._cache[name] = self.b.prepare(self.weights.get(name), name)
        return self._cache[name]

    def _fused(self, key: str, names: list[str]) -> torch.Tensor:
        """Stack several weights row-wise into one resident tensor (e.g. q/k/v -> one projection)."""
        if key not in self._cache:
            self._cache[key] = self.b.prepare(concat_rows([self.weights.get(n) for n in names]), key)
        return self._cache[key]

    def new_state(self, max_len: int) -> ContiguousKVCache:
        cfg = self.config
        return ContiguousKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, max_len,
                                 device=self.b.device, dtype=torch.float32)

    def new_paged_pool(self, num_blocks: int, block_size: int = 16) -> PagedKVPool:
        cfg = self.config
        return PagedKVPool(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, num_blocks, block_size,
                           device=self.b.device, dtype=torch.float32)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids [T] -> vectors [T, hidden]: row lookup in the resident (tied) embedding table.

        The lookup runs where the table lives, so on the GPU no row is copied from the CPU
        (a CPU->GPU copy of fresh data waits for all queued GPU work)."""
        return self.b.embedding(self._w("model.embed_tokens.weight"), ids.to(self.b.device, non_blocking=True))

    def self_attn(self, layer: int, x: torch.Tensor, positions: torch.Tensor | None = None,
                  state: ContiguousKVCache | None = None, residual: torch.Tensor | None = None,
                  start: int | None = None) -> torch.Tensor:
        """Attention half of one station. x: normalized stream [T, hidden] -> [T, hidden].

        positions: absolute positions of the T tokens (default 0..T-1).
        state:     KV cache; the new keys/values are written at `positions` and attention reads
                   every cached position up to the last new one.
        residual:  if given, added inside the output projection (fused residual connection).
        start:     first position as a plain int. Pass it whenever you can: reading it back from a GPU
                   tensor (int(positions[0])) forces the CPU to wait for all queued GPU work.
        """
        cfg, b = self.config, self.b
        p = f"model.layers.{layer}.self_attn."
        T = x.shape[0]
        Hq, Hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        if positions is None:
            positions = torch.arange(T, device=b.device)

        # 1. One fused projection for q, k and v (Qwen adds biases to all three)
        names = ["q_proj", "k_proj", "v_proj"]
        qkv = b.linear(x, self._fused(p + "qkv.weight", [p + n + ".weight" for n in names]),
                       self._fused(p + "qkv.bias", [p + n + ".bias" for n in names]))       # [T, 896+128+128]

        # 2. Split into heads
        q = qkv[:, : Hq * d].reshape(T, Hq, d)
        k = qkv[:, Hq * d : (Hq + Hkv) * d].reshape(T, Hkv, d)
        v = qkv[:, (Hq + Hkv) * d :].reshape(T, Hkv, d)

        # 3. Rotate queries and keys by position in one call (RoPE is independent per head)
        qk = b.rope(torch.cat([q, k], dim=1), positions, cfg.rope_theta)
        q, k = qk[:, :Hq], qk[:, Hq:]

        # 4. KV cache: store the new keys/values, then attend over everything cached so far
        if state is not None:
            if start is None:
                start = int(positions[0])                          # host sync; forward() passes start instead
            state.write(layer, start, k, v)
            k, v = state.read(layer, start + T)

        out = b.attention(q, k, v, causal=True)                   # [T, 14, 64]
        return b.linear(out.reshape(T, -1), self._w(p + "o_proj.weight"), residual=residual)   # no bias

    def mlp(self, layer: int, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """Feed-forward half of one station (SwiGLU). x: normalized stream [T, hidden] -> [T, hidden]."""
        b, p = self.b, f"model.layers.{layer}.mlp."
        w_gate_up = self._fused(p + "gate_up.weight", [p + "gate_proj.weight", p + "up_proj.weight"])  # [2*4864, 896]
        hidden = b.swiglu(x, w_gate_up)                           # silu(x @ gate.T) * (x @ up.T), [T, 4864]
        return b.linear(hidden, self._w(p + "down_proj.weight"), residual=residual)

    def block(self, layer: int, h: torch.Tensor, positions: torch.Tensor | None = None,
              state: ContiguousKVCache | None = None, start: int | None = None) -> torch.Tensor:
        """One full station: pre-norm attention and pre-norm MLP, each added back into the residual stream."""
        eps, b, p = self.config.rms_norm_eps, self.b, f"model.layers.{layer}."
        h = self.self_attn(layer, b.rms_norm(h, self._w(p + "input_layernorm.weight"), eps), positions, state,
                           residual=h, start=start)
        h = self.mlp(layer, b.rms_norm(h, self._w(p + "post_attention_layernorm.weight"), eps), residual=h)
        return h

    def forward(self, ids: torch.Tensor, state: ContiguousKVCache | None = None,
                last_only: bool = False) -> torch.Tensor:
        """Token ids [T] -> logits [T, vocab] (or [1, vocab] with last_only).

        Without a state the whole sequence is processed from position 0.
        With a state, the tokens continue from state.length and their keys/values are cached.
        """
        start = state.length if state is not None else 0
        if state is not None:
            state.reserve(start + len(ids))          # fail before any layer mutates state (atomic forward)
        positions = torch.arange(start, start + len(ids), device=self.b.device, dtype=torch.int32)
        h = self.embed(ids)
        for layer in range(self.config.num_hidden_layers):
            h = self.block(layer, h, positions, state, start)
        if state is not None:
            state.advance(len(ids))
        if last_only:
            h = h[-1:]
        h = self.b.rms_norm(h, self._w("model.norm.weight"), self.config.rms_norm_eps)
        # Tied embeddings: the table that turned ids into vectors now scores vectors against every token.
        return self.b.linear(h, self._w("model.embed_tokens.weight"))

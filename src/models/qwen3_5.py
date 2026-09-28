"""Qwen3.5 (text path): 18 Gated DeltaNet layers + 6 gated full-attention layers, on a pluggable backend.

Follows the verified spec of HF transformers' modeling_qwen3_5.py. Traps handled explicitly (vs Qwen2):
RMSNorm scales by (1 + w); the DeltaNet output norm uses plain w; q_proj rows interleave [query | gate] per head;
q/k are RMS-normed per head before a PARTIAL RoPE (first 64 of 256 dims, theta 1e7); attention output is gated by
sigmoid(gate) before o_proj; the causal conv runs before the q/k/v split; no biases anywhere; head_dim is explicit.
"""
import math

import torch

import ops
from backend.torch_ref import TorchBackend
from config import Qwen35Config
from quant import QuantTensor, concat_rows
from state import ContiguousKVCache, HybridPool, HybridState, PagedKVPool, PagedSequence

P = "model.language_model."


def _rows(w, idx: torch.Tensor):
    """Select rows of a plain or quantized weight (quantization blocks never cross rows)."""
    if isinstance(w, QuantTensor):
        mins = w.mins[idx] if w.mins is not None else None
        return QuantTensor(w.scheme, w.data[idx], w.scales[idx], (len(idx), w.shape[1]), w.block, mins)
    return w[idx]


class Qwen35Model:
    def __init__(self, config: Qwen35Config, weights, backend=None):
        self.config = config
        self.weights = weights
        self.b = backend or TorchBackend()
        self._cache: dict[str, torch.Tensor] = {}
        c = config
        self.attn_layers = [i for i, t in enumerate(c.layer_types) if t == "full_attention"]
        self.linear_layers = [i for i, t in enumerate(c.layer_types) if t == "linear_attention"]
        self.slot = {i: n for n, i in enumerate(self.attn_layers)} | {i: n for n, i in enumerate(self.linear_layers)}
        self.key_dim = c.linear_num_key_heads * c.linear_key_head_dim           # 2048
        self.value_dim = c.linear_num_value_heads * c.linear_value_head_dim     # 2048
        self.conv_dim = 2 * self.key_dim + self.value_dim                        # 6144

    # ---------------------------------------------------------------- weights
    def _w(self, name: str):
        """A 2-D weight in the backend's resident format (bf16 / quantized / fp32)."""
        if name not in self._cache:
            self._cache[name] = self.b.prepare(self.weights.get(P + name), name)
        return self._cache[name]

    def _fused(self, key: str, build):
        if key not in self._cache:
            self._cache[key] = self.b.prepare(build(), key)
        return self._cache[key]

    def _vec(self, name: str, one_plus: bool = False) -> torch.Tensor:
        """A small parameter kept in fp32 on the device (norm weights, A_log, dt_bias, conv taps)."""
        key = name + ("+1" if one_plus else "")
        if key not in self._cache:
            t = self.weights.get(P + name).float()
            self._cache[key] = (1.0 + t if one_plus else t).to(self.b.device)
        return self._cache[key]

    # ---------------------------------------------------------------- state
    def new_state(self, max_len: int, kv=None) -> HybridState:
        c = self.config
        if kv is None:
            kv = ContiguousKVCache(len(self.attn_layers), c.num_key_value_heads, c.head_dim, max_len,
                                   device=self.b.device, dtype=torch.float32)
        return HybridState(kv, len(self.linear_layers), c.linear_num_value_heads, c.linear_key_head_dim,
                           c.linear_value_head_dim, self.conv_dim, c.linear_conv_kernel_dim, device=self.b.device)

    def new_paged_pool(self, num_blocks: int, block_size: int = 16, max_seqs: int = 8) -> HybridPool:
        """Paged KV blocks for the attention layers + `max_seqs` DeltaNet state slots (~19 MB each on 0.8B/2B)."""
        c = self.config
        kv = PagedKVPool(len(self.attn_layers), c.num_key_value_heads, c.head_dim, num_blocks, block_size,
                         device=self.b.device, dtype=torch.float32)
        return HybridPool(kv, max_seqs, len(self.linear_layers), c.linear_num_value_heads, c.linear_key_head_dim,
                          c.linear_value_head_dim, self.conv_dim, c.linear_conv_kernel_dim, device=self.b.device)

    def new_paged_state(self, pool: HybridPool) -> HybridState:
        return pool.new_sequence()

    # ---------------------------------------------------------------- layers
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.b.embedding(self._w("embed_tokens.weight"), ids.to(self.b.device, non_blocking=True))

    def _qkvg(self, i: int):
        """Fused [query(Hq*d) ; k ; v ; gate(Hq*d)] projection. q_proj interleaves [query_h | gate_h] per head."""
        c, p = self.config, f"layers.{i}.self_attn."
        Hq, d = c.num_attention_heads, c.head_dim

        def build():
            wq = self.weights.get(P + p + "q_proj.weight")
            rows = torch.arange(2 * Hq * d).view(Hq, 2, d)
            return concat_rows([_rows(wq, rows[:, 0].flatten()), self.weights.get(P + p + "k_proj.weight"),
                                self.weights.get(P + p + "v_proj.weight"), _rows(wq, rows[:, 1].flatten())])
        return self._fused(p + "qkvg.weight", build)

    def full_attention(self, i: int, x, positions, state: HybridState | None, start: int, residual):
        c, b = self.config, self.b
        p = f"layers.{i}.self_attn."
        T, Hq, Hkv, d = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim

        y = b.linear(x, self._qkvg(i))
        q = y[:, : Hq * d].reshape(T, Hq, d)
        k = y[:, Hq * d:(Hq + Hkv) * d].reshape(T, Hkv, d)
        v = y[:, (Hq + Hkv) * d:(Hq + 2 * Hkv) * d].reshape(T, Hkv, d)
        gate = y[:, (Hq + 2 * Hkv) * d:]
        q = b.rms_norm(q, self._vec(p + "q_norm.weight", one_plus=True), c.rms_norm_eps)   # per head, before RoPE
        k = b.rms_norm(k, self._vec(p + "k_norm.weight", one_plus=True), c.rms_norm_eps)
        r = c.rotary_dim                                                                    # only 64 of 256 dims rotate
        qk = b.rope(torch.cat([q[..., :r], k[..., :r]], dim=1).contiguous(), positions, c.rope_theta)
        q = torch.cat([qk[:, :Hq], q[..., r:]], dim=-1)
        k = torch.cat([qk[:, Hq:], k[..., r:]], dim=-1)
        if state is not None:
            slot = self.slot[i]
            state.kv.write(slot, start, k, v)
            k, v = state.kv.read(slot, start + T)
        o = b.attention(q, k, v, causal=True).reshape(T, Hq * d)                           # scale 1/sqrt(256)
        o = o * torch.sigmoid(gate)                                                         # output gate, BEFORE o_proj
        return b.linear(o, self._w(p + "o_proj.weight"), residual=residual)

    def linear_attention(self, i: int, x, state: HybridState | None, residual):
        c, b = self.config, self.b
        p = f"layers.{i}.linear_attn."
        T, H, dk, dv = x.shape[0], c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
        y = b.linear(x, self._fused(p + "in_proj.weight",
                                    lambda: concat_rows([self.weights.get(P + p + n + ".weight") for n in names])))
        qkv, z = y[:, : self.conv_dim], y[:, self.conv_dim:self.conv_dim + self.value_dim]
        beta_logit, a = y[:, -2 * H:-H], y[:, -H:]

        slot = self.slot[i]
        if T == 1 and state is not None:                          # decode: one fused step (kernels on Metal)
            o = b.deltanet_decode(qkv, z, beta_logit, a, state, slot, self._vec(p + "conv1d.weight").squeeze(1),
                                  self._vec(p + "A_log"), self._vec(p + "dt_bias"), self._vec(p + "norm.weight"),
                                  c.rms_norm_eps, (H, dk, dv, self.key_dim))
            return b.linear(o, self._w(p + "out_proj.weight"), residual=residual)
        tail = state.conv_tail[slot] if state is not None else torch.zeros(c.linear_conv_kernel_dim - 1, self.conv_dim, device=b.device)
        conv_w = self._vec(p + "conv1d.weight").squeeze(1)                                  # [6144, 4]
        u, new_tail = b.causal_conv1d(qkv, tail, conv_w)                                     # conv BEFORE the split

        q = u[:, : self.key_dim].reshape(T, H, dk)
        k = u[:, self.key_dim:2 * self.key_dim].reshape(T, H, dk)
        v = u[:, 2 * self.key_dim:].reshape(T, H, dv)
        beta = torch.sigmoid(beta_logit.float())
        ab = a.float() + self._vec(p + "dt_bias")
        g = -torch.exp(self._vec(p + "A_log")) * torch.where(ab > 20, ab, torch.log1p(torch.exp(ab)))   # decay <= 0
        q = ops.l2norm(q.float()) * (1.0 / math.sqrt(dk))
        k = ops.l2norm(k.float())

        S = state.S[slot] if state is not None else torch.zeros(H, dk, dv, device=b.device)
        o, S = b.gated_delta(q, k, v.float(), beta, g, S)
        if state is not None:
            state.S[slot] = S
            state.conv_tail[slot] = new_tail
        o = b.rms_norm_gated(o.reshape(T * H, dv), z.reshape(T * H, dv), self._vec(p + "norm.weight"), c.rms_norm_eps)
        return b.linear(o.reshape(T, H * dv), self._w(p + "out_proj.weight"), residual=residual)

    def mlp(self, i: int, x, residual):
        b, p = self.b, f"layers.{i}.mlp."
        w = self._fused(p + "gate_up.weight", lambda: concat_rows(
            [self.weights.get(P + p + "gate_proj.weight"), self.weights.get(P + p + "up_proj.weight")]))
        return b.linear(b.swiglu(x, w), self._w(p + "down_proj.weight"), residual=residual)

    def block(self, i: int, h, positions, state, start: int):
        c, b = self.config, self.b
        x = b.rms_norm(h, self._vec(f"layers.{i}.input_layernorm.weight", one_plus=True), c.rms_norm_eps)
        if c.layer_types[i] == "full_attention":
            h = self.full_attention(i, x, positions, state, start, residual=h)
        else:
            h = self.linear_attention(i, x, state, residual=h)
        x = b.rms_norm(h, self._vec(f"layers.{i}.post_attention_layernorm.weight", one_plus=True), c.rms_norm_eps)
        return self.mlp(i, x, residual=h)

    def forward(self, ids: torch.Tensor, state: HybridState | None = None, last_only: bool = False,
                capture: dict | None = None) -> torch.Tensor:
        """Token ids [T] -> logits [T, vocab]. With a state, continues from state.length and updates it."""
        start = state.length if state is not None else 0
        if state is not None:
            state.reserve(start + len(ids))          # fail before any layer mutates state (atomic forward)
        positions = torch.arange(start, start + len(ids), device=self.b.device, dtype=torch.int32)
        h = self.embed(ids)
        if capture is not None:
            capture["embed"] = h
        for i in range(self.config.num_hidden_layers):
            h = self.block(i, h, positions, state, start)
            if capture is not None:
                capture[f"l{i}_out"] = h
        if state is not None:
            state.advance(len(ids))
        if last_only:
            h = h[-1:]
        h = self.b.rms_norm(h, self._vec("norm.weight", one_plus=True), self.config.rms_norm_eps)
        if capture is not None:
            capture["final_norm"] = h
        return self.b.linear(h, self._w("embed_tokens.weight"))                            # tied output head

    def decode_batch(self, tokens: list[int], states: list) -> torch.Tensor:
        """One decode step for B independent sequences -> logits [B, vocab].

        Projections and MLPs run batched ([B, K] x [K, N]: each weight read once per step for everyone).
        Per-sequence parts: attention over each KV history, and each DeltaNet layer's recurrent/conv state.
        """
        c, b = self.config, self.b
        B, Hq, Hkv, d = len(tokens), c.num_attention_heads, c.num_key_value_heads, c.head_dim
        H, dk, dv = c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        for st in states:
            st.reserve(st.length + 1)
        starts = [st.length for st in states]
        positions = torch.tensor(starts, device=b.device, dtype=torch.int32)
        kvs = [st.kv for st in states]
        paged = all(isinstance(kv, PagedSequence) and kv.pool is kvs[0].pool for kv in kvs)
        if paged:                                       # one KV write and one attention dispatch per layer for all
            kv_pool = kvs[0].pool
            blocks, offsets, tables, lens = kv_pool.batch_layout(kvs, starts)
        h = self.embed(torch.tensor(tokens))
        for i in range(c.num_hidden_layers):
            x = b.rms_norm(h, self._vec(f"layers.{i}.input_layernorm.weight", one_plus=True), c.rms_norm_eps)
            if c.layer_types[i] == "full_attention":
                p = f"layers.{i}.self_attn."
                y = b.linear(x, self._qkvg(i))
                q = b.rms_norm(y[:, : Hq * d].reshape(B, Hq, d), self._vec(p + "q_norm.weight", one_plus=True), c.rms_norm_eps)
                k = b.rms_norm(y[:, Hq * d:(Hq + Hkv) * d].reshape(B, Hkv, d), self._vec(p + "k_norm.weight", one_plus=True), c.rms_norm_eps)
                v = y[:, (Hq + Hkv) * d:(Hq + 2 * Hkv) * d].reshape(B, Hkv, d)
                gate = y[:, (Hq + 2 * Hkv) * d:]
                r = c.rotary_dim
                qk = b.rope(torch.cat([q[..., :r], k[..., :r]], dim=1).contiguous(), positions, c.rope_theta)
                q = torch.cat([qk[:, :Hq], q[..., r:]], dim=-1)
                k = torch.cat([qk[:, Hq:], k[..., r:]], dim=-1)
                if paged:
                    L = self.slot[i]
                    kv_pool.write_batch(L, blocks, offsets, k, v)
                    o = b.paged_attention(q, kv_pool.k[L], kv_pool.v[L], tables, lens, kv_pool.block_size)
                else:
                    outs = []
                    for j, st in enumerate(states):
                        st.kv.write(self.slot[i], starts[j], k[j:j + 1], v[j:j + 1])
                        K, V = st.kv.read(self.slot[i], starts[j] + 1)
                        outs.append(b.attention(q[j:j + 1].contiguous(), K, V, causal=True))
                    o = torch.cat(outs)
                o = o.reshape(B, Hq * d) * torch.sigmoid(gate)
                h = b.linear(o, self._w(p + "o_proj.weight"), residual=h)
            else:
                p = f"layers.{i}.linear_attn."
                names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                y = b.linear(x, self._fused(p + "in_proj.weight",
                                            lambda: concat_rows([self.weights.get(P + p + n + ".weight") for n in names])))
                qkv, z = y[:, : self.conv_dim], y[:, self.conv_dim:self.conv_dim + self.value_dim]
                beta_logit, a = y[:, -2 * H:-H], y[:, -H:]
                conv_w = self._vec(p + "conv1d.weight").squeeze(1)
                o = b.deltanet_decode_batch(qkv, z, beta_logit, a, states, self.slot[i], conv_w, self._vec(p + "A_log"),
                                            self._vec(p + "dt_bias"), self._vec(p + "norm.weight"), c.rms_norm_eps,
                                            (H, dk, dv, self.key_dim))
                h = b.linear(o, self._w(p + "out_proj.weight"), residual=h)
            x = b.rms_norm(h, self._vec(f"layers.{i}.post_attention_layernorm.weight", one_plus=True), c.rms_norm_eps)
            h = self.mlp(i, x, residual=h)
        for st in states:
            st.advance(1)
        h = b.rms_norm(h, self._vec("norm.weight", one_plus=True), c.rms_norm_eps)
        return b.linear(h, self._w("embed_tokens.weight"))

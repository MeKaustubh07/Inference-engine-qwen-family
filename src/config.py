"""ModelConfig: the recipe card, read from config.json."""
import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int          # width of the residual stream (896 for Qwen2.5-0.5B)
    intermediate_size: int    # width of the MLP's middle layer (4864)
    num_hidden_layers: int    # number of stations (24)
    num_attention_heads: int  # query heads (14)
    num_key_value_heads: int  # key/value heads (2)  -> GQA
    rms_norm_eps: float       # tiny constant inside RMSNorm (1e-6)
    rope_theta: float         # RoPE base frequency (1e6)
    tie_word_embeddings: bool # reuse the embedding table as the output head

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads   # 896 // 14 = 64

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        c = json.load(open(path))
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )


@dataclass
class Qwen35Config:
    """Qwen3.5 text model (hybrid Gated DeltaNet + gated full attention). Fields live under config["text_config"]."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int          # 8 query heads (each with a same-size output gate)
    num_key_value_heads: int          # 2
    head_dim: int                     # 256, explicit (NOT hidden_size // heads)
    rms_norm_eps: float
    rope_theta: float                 # 1e7
    partial_rotary_factor: float      # 0.25 -> only the first 64 of 256 dims rotate
    layer_types: list[str]            # "linear_attention" | "full_attention", one per layer
    linear_conv_kernel_dim: int       # 4
    linear_key_head_dim: int          # 128
    linear_value_head_dim: int        # 128
    linear_num_key_heads: int         # 16
    linear_num_value_heads: int       # 16
    tie_word_embeddings: bool

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_json(cls, path: str) -> "Qwen35Config":
        top = json.load(open(path))
        c = top["text_config"]
        rope = c["rope_parameters"]
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"], rms_norm_eps=c["rms_norm_eps"],
            rope_theta=float(rope["rope_theta"]), partial_rotary_factor=rope["partial_rotary_factor"],
            layer_types=list(c["layer_types"]), linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
            linear_key_head_dim=c["linear_key_head_dim"], linear_value_head_dim=c["linear_value_head_dim"],
            linear_num_key_heads=c["linear_num_key_heads"], linear_num_value_heads=c["linear_num_value_heads"],
            tie_word_embeddings=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
        )

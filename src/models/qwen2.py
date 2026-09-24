"""Qwen2.5 model assembled from the loader's tensors. Week 3: embedding only."""
import torch

from config import ModelConfig
from weight_loader import SafetensorsFile


class Qwen2Model:
    def __init__(self, config: ModelConfig, weights: SafetensorsFile):
        self.config = config
        self.weights = weights
        self.embed_table = weights.get("model.embed_tokens.weight")   # [vocab, hidden], bf16 view

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids [T] -> vectors [T, hidden]: row lookup in the embedding table."""
        return self.embed_table[ids].float()

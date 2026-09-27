"""Turn the last position's logits into one chosen token id."""
from dataclasses import dataclass

import torch

from ops import softmax


@dataclass
class SamplingParams:
    temperature: float = 0.7        # <1 sharpens the distribution, >1 flattens it; 0 means greedy
    top_k: int = 20                 # keep only the k most likely tokens (0 = off)
    top_p: float = 0.8              # keep the smallest set whose probabilities add up to p (1.0 = off)
    repetition_penalty: float = 1.1 # >1 discourages tokens that already appeared (1.0 = off)
    seed: int | None = None


def sample(logits: torch.Tensor, prev_ids: list[int], params: SamplingParams,
           generator: torch.Generator | None = None) -> int:
    """logits: [vocab] for the next position. Order matches HF: penalty -> temperature -> top-k -> top-p."""
    x = logits.float().clone()

    if params.repetition_penalty != 1.0 and prev_ids:
        seen = torch.tensor(sorted(set(prev_ids)))
        s = x[seen]
        x[seen] = torch.where(s > 0, s / params.repetition_penalty, s * params.repetition_penalty)

    if params.temperature <= 0:
        return int(x.argmax())
    x = x / params.temperature

    if params.top_k > 0:
        kth = torch.topk(x, min(params.top_k, x.numel())).values[-1]
        x = x.masked_fill(x < kth, float("-inf"))

    if params.top_p < 1.0:
        order = torch.argsort(x, descending=True)
        probs = softmax(x[order])
        cum = torch.cumsum(probs, dim=0)
        drop = cum - probs > params.top_p          # drop tokens once the tokens before them already cover p
        x[order[drop]] = float("-inf")

    probs = softmax(x)
    return int(torch.multinomial(probs, 1, generator=generator))

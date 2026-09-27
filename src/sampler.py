"""Turn the last position's logits into one chosen token id."""
from dataclasses import dataclass

import torch

from ops import softmax

MIN_TEMPERATURE = 1e-5


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

    if params.temperature < MIN_TEMPERATURE:       # 0 means greedy; so does anything small enough to overflow x / t
        return int(x.argmax())
    x = x / params.temperature

    ids = None
    if 0 < params.top_k < x.numel():
        # keep the candidates (ties at the k-th value included, like HF) and work on them only: sorting the full
        # 248k vocabulary for top-p on every token would dominate the step time
        kth = torch.topk(x, params.top_k).values[-1]
        ids = torch.nonzero(x >= kth).flatten()
        x = x[ids]

    if params.top_p < 1.0:
        order = torch.argsort(x, descending=True)
        probs = softmax(x[order])
        cum = torch.cumsum(probs, dim=0)
        drop = cum - probs >= params.top_p         # drop a token once the tokens ranked above it already cover p
        drop[0] = False                            # always keep the most likely token (HF: min_tokens_to_keep=1)
        x[order[drop]] = float("-inf")

    probs = softmax(x)
    i = int(torch.multinomial(probs, 1, generator=generator))
    return int(ids[i]) if ids is not None else i

"""Autoregressive generation loops."""
from collections.abc import Iterator

import torch

from sampler import SamplingParams, sample


def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int, eos_ids: set[int]) -> list[int]:
    """No cache: re-run the whole sequence each step and take the highest-scoring next token."""
    ids = tokenizer.encode(prompt)
    new: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.forward(torch.tensor(ids))       # [T, vocab]
        next_id = int(logits[-1].argmax())              # only the last position predicts the next token
        if next_id in eos_ids:
            break
        ids.append(next_id)
        new.append(next_id)
    return new


def generate_stream(model, tokenizer, prompt_ids: list[int], params: SamplingParams,
                    max_new_tokens: int, eos_ids: set[int]) -> Iterator[str]:
    """Sample token by token and yield text as soon as it forms complete characters (no cache yet)."""
    gen = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    ids = list(prompt_ids)
    new: list[int] = []
    emitted = ""
    for _ in range(max_new_tokens):
        logits = model.forward(torch.tensor(ids))[-1]
        next_id = sample(logits, ids, params, gen)
        if next_id in eos_ids:
            break
        ids.append(next_id)
        new.append(next_id)
        text = tokenizer.decode(new)
        if not text.endswith("\ufffd"):          # wait until a multi-byte character is complete
            yield text[len(emitted):]
            emitted = text

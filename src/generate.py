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
        # only the last position predicts the next token; ids past the tokenizer's vocab are padding rows
        next_id = int(logits[-1, : tokenizer.vocab_size()].argmax())
        if next_id in eos_ids:
            break
        ids.append(next_id)
        new.append(next_id)
    return new


def generate_stream(model, tokenizer, prompt_ids: list[int], params: SamplingParams,
                    max_new_tokens: int, eos_ids: set[int], record: dict | None = None) -> Iterator[str]:
    """Sample token by token and yield text as soon as it forms complete characters (no cache yet).

    record (optional): filled with {"ids": generated ids, "stop": "eos" | "length"}.
    """
    gen = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    ids = list(prompt_ids)
    new: list[int] = []
    emitted = ""
    stop = "length"
    for _ in range(max_new_tokens):
        logits = model.forward(torch.tensor(ids))[-1, : tokenizer.vocab_size()]   # drop padding ids
        next_id = sample(logits, ids, params, gen)
        if next_id in eos_ids:
            stop = "eos"
            break
        ids.append(next_id)
        new.append(next_id)
        text = tokenizer.decode(new)
        if not text.endswith("\ufffd"):          # wait until a multi-byte character is complete
            yield text[len(emitted):]
            emitted = text
    text = tokenizer.decode(new)                   # flush anything still held back
    if len(text) > len(emitted):
        yield text[len(emitted):]
    if record is not None:
        record["ids"], record["stop"] = new, stop

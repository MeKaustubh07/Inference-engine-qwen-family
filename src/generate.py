"""Autoregressive generation loops."""
import torch


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

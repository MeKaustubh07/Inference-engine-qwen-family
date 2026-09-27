"""ChatML prompt formatting (the Qwen2.5 chat template, without tool calling)."""

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def format_chat(messages: list[dict], add_generation_prompt: bool = True,
                default_system: str = DEFAULT_SYSTEM) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> one prompt string."""
    out = []
    if not messages or messages[0]["role"] != "system":
        out.append(f"{IM_START}system\n{default_system}{IM_END}\n")
    for m in messages:
        out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)

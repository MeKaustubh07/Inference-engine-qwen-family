"""ChatML prompt formatting for the Qwen2.5 and Qwen3.5 chat templates (without tool calling)."""
import re

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def format_chat(messages: list[dict], add_generation_prompt: bool = True, default_system: str = DEFAULT_SYSTEM,
                style: str = "qwen2.5", enable_thinking: bool = False) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> one prompt string.

    qwen2.5: a default system prompt is inserted when none is given.
    qwen3.5: no default system prompt; earlier assistant turns drop their <think>...</think> reasoning; the
             generation prompt opens a thinking block (enable_thinking) or closes an empty one (thinking off).
    """
    out = []
    if style == "qwen2.5" and (not messages or messages[0]["role"] != "system"):
        out.append(f"{IM_START}system\n{default_system}{IM_END}\n")
    for m in messages:
        content = m["content"]
        if style == "qwen3.5" and m["role"] == "assistant":
            content = _THINK.sub("", content)
        out.append(f"{IM_START}{m['role']}\n{content}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
        if style == "qwen3.5":
            out.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(out)

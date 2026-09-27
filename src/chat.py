"""ChatML prompt formatting for the Qwen2.5 and Qwen3.5 chat templates (without tool calling)."""
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def format_chat(messages: list[dict], add_generation_prompt: bool = True, default_system: str = DEFAULT_SYSTEM,
                style: str = "qwen2.5", enable_thinking: bool = False) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> one prompt string.

    qwen2.5: a default system prompt is inserted when none is given; content is used verbatim.
    qwen3.5 (mirrors chat_template.jinja): no default system prompt; every message is trimmed; an assistant turn's
             text before its last </think> is reasoning (the model's own replies have no opening <think> because
             the prompt already opened it); reasoning is kept only for assistant turns after the last user query;
             the generation prompt opens a thinking block (enable_thinking) or closes an empty one.
    """
    if style == "qwen2.5":
        out = []
        if not messages or messages[0]["role"] != "system":
            out.append(f"{IM_START}system\n{default_system}{IM_END}\n")
        out += [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
        if add_generation_prompt:
            out.append(f"{IM_START}assistant\n")
        return "".join(out)

    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        content = m["content"].strip()
        if m["role"] == "assistant":
            reasoning = ""
            if "</think>" in content:
                reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
                content = content.split("</think>")[-1].lstrip("\n")
            if i > last_user:
                out.append(f"{IM_START}assistant\n<think>\n{reasoning}\n</think>\n\n{content}{IM_END}\n")
            else:
                out.append(f"{IM_START}assistant\n{content}{IM_END}\n")
        else:
            out.append(f"{IM_START}{m['role']}\n{content}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)

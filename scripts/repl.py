"""Terminal chat with the engine: multi-turn, streaming."""
import sys

sys.path.insert(0, "src")
from chat import format_chat
from config import ModelConfig
from generate import generate_stream
from models.qwen2 import Qwen2Model
from sampler import SamplingParams
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

MODEL_DIR = "models/qwen2.5-0.5b"
EOS = {151645, 151643}


def main() -> None:
    cfg = ModelConfig.from_json(f"{MODEL_DIR}/config.json")
    model = Qwen2Model(cfg, SafetensorsFile(f"{MODEL_DIR}/model.safetensors"))
    tok = Tokenizer(f"{MODEL_DIR}/tokenizer.json")
    params = SamplingParams()
    history: list[dict] = []
    print("Chat with Qwen2.5-0.5B (your engine). Empty line or Ctrl-D to quit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        ids = tok.encode(format_chat(history))
        print("qwen> ", end="", flush=True)
        reply = ""
        for piece in generate_stream(model, tok, ids, params, max_new_tokens=256, eos_ids=EOS):
            print(piece, end="", flush=True)
            reply += piece
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

"""Answer key: tokenize test strings with the official HF tokenizer, write JSON goldens.

usage: golden_tokens.py [model_dir] [out.json]      (default: Qwen2.5-0.5B -> tests/golden_tokens.json)
"""
import json
import random
import sys

from tokenizers import Tokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen2.5-0.5b"
out_path = sys.argv[2] if len(sys.argv) > 2 else "tests/golden_tokens.json"
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
cases = [
    "The capital of France is",
    "Hello world",
    "hello   world  ",
    "I'm here, you're there. Don't stop!",
    "Numbers: 12345 and 3.14159, year 2026",
    "Kaustubh builds an inference engine in TypeScript.",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "नमस्ते दुनिया — こんにちは世界 — 🚀🔥",
    "def f(x):\n    return x * 2\n\n\n",
    "   leading spaces and\ttabs\tand trailing   ",
    "",
    # scripts with combining marks (Qwen3.5's pre-tokenizer lists \p{M}; see src/tokenizer.py)
    "தமிழ் மொழி", "สวัสดีครับ", "مَرْحَبًا بِكُمْ", "שָׁלוֹם", "Tiếng Việt có dấu", "éclair", "❤️ 👍🏽",
    "ক্ষমা করুন", "ಕನ್ನಡ ಭಾಷೆ", "ພາສາລາວ", "Ελληνικά ά", "ǟb", "Zalgo z̷̢a̶l̵g̸o", "日本語のテキスト", "한국어 텍스트",
]
random.seed(0)
alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789.,!?'\n-_()[]{}<>|éü你好😀्ािु"
for _ in range(40):
    cases.append("".join(random.choice(alphabet) for _ in range(random.randint(1, 60))))

out = [{"text": c, "ids": tok.encode(c).ids} for c in cases]
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(out)} cases to {out_path}")

"""Compare our tokenizer against the official answer key (tests/golden_tokens.json)."""
import json
import sys

sys.path.insert(0, "src")
from tokenizer import Tokenizer

tok = Tokenizer("models/qwen2.5-0.5b/tokenizer.json")
cases = json.load(open("tests/golden_tokens.json", encoding="utf-8"))

passed = 0
for c in cases:
    ids = tok.encode(c["text"])
    same = ids == c["ids"]
    round_trip = tok.decode(ids) == c["text"]
    if same and round_trip:
        passed += 1
    else:
        print("FAIL", repr(c["text"]), "\n  ours:", ids, "\n  gold:", c["ids"], "\n  round-trip ok:", round_trip)
print(f"{passed}/{len(cases)} cases match the official tokenizer")

for s in ["The capital of France is", "Kaustubh", "<|im_start|>user\nhi<|im_end|>"]:
    ids = tok.encode(s)
    print(repr(s), "->", ids, "->", [tok.decode([i]) for i in ids])
print("vocab size:", tok.vocab_size())
sys.exit(0 if passed == len(cases) else 1)

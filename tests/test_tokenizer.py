"""Compare our tokenizer against the official answer keys for the Qwen3.5 tokenizer (shared by 0.8B and 2B)."""
import json
import sys
import unicodedata

sys.path.insert(0, "src")
from tokenizer import Tokenizer

MODELS = [("Qwen3.5-0.8B", "models/qwen3.5-0.8b", "tests/golden_tokens_qwen35.json")]

all_ok = True
for name, model_dir, golden in MODELS:
    tok = Tokenizer(f"{model_dir}/tokenizer.json")
    cases = json.load(open(golden, encoding="utf-8"))
    passed = 0
    for c in cases:
        ids = tok.encode(c["text"])
        round_trip = tok.decode(ids) == unicodedata.normalize("NFC", c["text"])    # the normalizer applies NFC
        if ids == c["ids"] and round_trip:
            passed += 1
        else:
            print("FAIL", name, repr(c["text"]), "\n  ours:", ids, "\n  gold:", c["ids"], "\n  round-trip ok:", round_trip)
    all_ok &= passed == len(cases)
    print(f"{name}: {passed}/{len(cases)} cases match the official tokenizer (vocab {tok.vocab_size()})")

# the 2B ships the same tokenizer files, so the 0.8B's answer keys cover it too
same = all(open(f"models/qwen3.5-2b/{f}", "rb").read() == open(f"models/qwen3.5-0.8b/{f}", "rb").read()
           for f in ("tokenizer.json", "tokenizer_config.json"))
print(f"{'PASS' if same else 'FAIL'}  Qwen3.5-2B tokenizer.json and tokenizer_config.json are byte-equal to the 0.8B's")
all_ok &= same
sys.exit(0 if all_ok else 1)

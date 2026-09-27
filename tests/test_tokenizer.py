"""Compare our tokenizer against the official answer keys for every model (Qwen2.5 and Qwen3.5)."""
import json
import sys
import unicodedata

sys.path.insert(0, "src")
from tokenizer import Tokenizer

MODELS = [("Qwen2.5-0.5B", "models/qwen2.5-0.5b", "tests/golden_tokens.json"),
          ("Qwen3.5-0.8B", "models/qwen3.5-0.8b", "tests/golden_tokens_qwen35.json")]

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
sys.exit(0 if all_ok else 1)

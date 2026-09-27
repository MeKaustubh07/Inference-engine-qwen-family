"""Week 6: sampler behaviour, chat template vs HF, and a streamed chat reply."""
import sys
import torch

sys.path.insert(0, "src")
from chat import format_chat
from config import ModelConfig
from generate import generate_stream
from models.qwen2 import Qwen2Model
from sampler import SamplingParams, sample
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

results = []
def check(name, ok):
    results.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name}")

# 1. temperature 0 is greedy
logits = torch.tensor([1.0, 5.0, 2.0, 4.9])
check("temperature 0 picks argmax", sample(logits, [], SamplingParams(temperature=0, repetition_penalty=1.0)) == 1)

# 2. top-k = 1 is greedy regardless of temperature
g = torch.Generator().manual_seed(0)
check("top_k=1 always picks argmax", all(sample(logits, [], SamplingParams(temperature=2.0, top_k=1, top_p=1.0, repetition_penalty=1.0), g) == 1 for _ in range(50)))

# 3. top-p keeps only the head of the distribution: probs 0.5, 0.3, 0.15, 0.05 with p=0.75 -> only tokens 0 and 1
lp = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
picks = {sample(lp, [], SamplingParams(temperature=1.0, top_k=0, top_p=0.75, repetition_penalty=1.0), g) for _ in range(400)}
check(f"top_p=0.75 only samples tokens {{0, 1}} (got {sorted(picks)})", picks == {0, 1})

# 4. repetition penalty: a token that was seen loses its lead
rp = torch.tensor([3.0, 3.2])
check("repetition penalty flips the choice away from a repeated token",
      sample(rp, [1], SamplingParams(temperature=0, repetition_penalty=1.1)) == 0)

# 5. same seed -> same output
a = [sample(lp, [], SamplingParams(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0), torch.Generator().manual_seed(7)) for _ in range(20)]
b = [sample(lp, [], SamplingParams(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0), torch.Generator().manual_seed(7)) for _ in range(20)]
check("seeded sampling is reproducible", a == b)

# 6. chat template matches HF's own renderer
from transformers import AutoTokenizer
hf_tok = AutoTokenizer.from_pretrained("models/qwen2.5-0.5b")
convs = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "and 3+3?"}],
]
for c in convs:
    check(f"chat template matches HF ({len(c)} messages)", format_chat(c) == hf_tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True))

# 7. a real streamed chat reply that ends at EOS
cfg = ModelConfig.from_json("models/qwen2.5-0.5b/config.json")
model = Qwen2Model(cfg, SafetensorsFile("models/qwen2.5-0.5b/model.safetensors"))
tok = Tokenizer("models/qwen2.5-0.5b/tokenizer.json")
ids = tok.encode(format_chat([{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]))
reply = "".join(generate_stream(model, tok, ids, SamplingParams(seed=0), max_new_tokens=40, eos_ids={151645, 151643}))
print(f"      reply: {reply!r}")
check("chat reply mentions Paris", "Paris" in reply)

print(f"\n{sum(results)}/{len(results)} checks passed")

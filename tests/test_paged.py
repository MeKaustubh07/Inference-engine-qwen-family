"""Week 8: paged KV cache == contiguous cache, sequence isolation, allocator behaviour, memory accounting."""
import sys
import torch

sys.path.insert(0, "src")
from config import ModelConfig
from models.qwen2 import Qwen2Model
from state import BlockAllocator, OutOfBlocks
from weight_loader import SafetensorsFile

D = "models/qwen2.5-0.5b"
cfg = ModelConfig.from_json(f"{D}/config.json")
model = Qwen2Model(cfg, SafetensorsFile(f"{D}/model.safetensors"))

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

# 1. allocator
a = BlockAllocator(4)
got = [a.allocate() for _ in range(4)]
check("allocator hands out 4 distinct blocks", sorted(got) == [0, 1, 2, 3] and a.num_free == 0)
try:
    a.allocate(); check("allocator raises when empty", False)
except OutOfBlocks:
    check("allocator raises OutOfBlocks when empty", True)
a.free(got[:2])
check("freed blocks are reusable", a.num_free == 2 and {a.allocate(), a.allocate()} == set(got[:2]))

def run(ids, state, split):
    """prefill `split` tokens, then decode the rest one at a time; return all logits."""
    parts = [model.forward(ids[:split], state=state)]
    for t in range(split, len(ids)):
        parts.append(model.forward(ids[t:t + 1], state=state))
    return torch.cat(parts)

# 2. paged (block_size 4, so every prompt spans several blocks with ragged ends) == contiguous
pool = model.new_paged_pool(num_blocks=64, block_size=4)
for i in range(5):
    ids = torch.load(f"tests/golden/{i}.pt")["ids"]
    split = max(1, len(ids) // 2)
    ref = run(ids, model.new_state(len(ids)), split)
    seq = pool.new_sequence()
    paged = run(ids, seq, split)
    d = (paged - ref).abs().max().item()
    check(f"prompt {i} ({len(ids)} tokens, {len(seq.block_table)} blocks): paged == contiguous, max|diff|={d:.1e}", d < 1e-4)
    seq.free()
check("all blocks returned to the pool after freeing", pool.allocator.num_free == 64)

# 3. two sequences interleaved in one pool don't disturb each other
ids_a = torch.load("tests/golden/3.pt")["ids"]; ids_b = torch.load("tests/golden/4.pt")["ids"]
alone_a = run(ids_a, model.new_state(len(ids_a)), 2)
alone_b = run(ids_b, model.new_state(len(ids_b)), 2)
sa, sb = pool.new_sequence(), pool.new_sequence()
out_a = [model.forward(ids_a[:2], state=sa)]; out_b = [model.forward(ids_b[:2], state=sb)]
for t in range(2, max(len(ids_a), len(ids_b))):          # alternate one decode step each
    if t < len(ids_a): out_a.append(model.forward(ids_a[t:t + 1], state=sa))
    if t < len(ids_b): out_b.append(model.forward(ids_b[t:t + 1], state=sb))
da = (torch.cat(out_a) - alone_a).abs().max().item(); db = (torch.cat(out_b) - alone_b).abs().max().item()
overlap = set(sa.block_table) & set(sb.block_table)
check(f"interleaved sequences match their solo runs (diff {da:.1e}, {db:.1e}), no shared blocks", da < 1e-4 and db < 1e-4 and not overlap)

# 4. memory: paged uses ceil(len/bs) blocks; a contiguous cache reserves max_len up front
per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 4
check(f"paged bytes for {sa.length} tokens = {sa.bytes_used()} (= {len(sa.block_table)} blocks x 4 tokens x {per_tok} B)",
      sa.bytes_used() == len(sa.block_table) * 4 * per_tok)
sa.free(); sb.free()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

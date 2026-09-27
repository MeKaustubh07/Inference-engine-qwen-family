"""Per-sequence decode state: the KV cache.

Keys and values of past tokens never change, so each layer stores them once and every later
step reads them back instead of recomputing the whole prefix (O(n^3) total work -> O(n^2)).
"""
import torch


class ContiguousKVCache:
    """One preallocated buffer per layer: k/v [max_len, n_kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_len: int,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.max_len = max_len
        self.k = torch.zeros(n_layers, max_len, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0                                   # tokens committed so far

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k/v [T, Hkv, d] for positions start .. start+T-1."""
        end = start + k.shape[0]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, capacity {self.max_len}")
        self.k[layer, start:end] = k.to(self.k.dtype)
        self.v[layer, start:end] = v.to(self.v.dtype)

    def read(self, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """All keys/values for positions 0 .. end-1 (views, no copy)."""
        return self.k[layer, :end], self.v[layer, :end]

    def advance(self, n: int) -> None:
        self.length += n

    def bytes_used(self) -> int:
        return 2 * self.k[:, : self.length].numel() * self.k.element_size()


class OutOfBlocks(RuntimeError):
    """The shared pool has no free block left (the scheduler should queue or evict)."""


class BlockAllocator:
    """Hands out fixed-size blocks from a free list, like an OS page allocator."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks - 1, -1, -1))   # pop() returns block 0 first

    def allocate(self) -> int:
        if not self.free_list:
            raise OutOfBlocks(f"all {self.num_blocks} KV blocks are in use")
        return self.free_list.pop()

    def free(self, blocks: list[int]) -> None:
        self.free_list.extend(reversed(blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_list)


class PagedKVPool:
    """One big pool of KV blocks shared by every sequence: k/v [layers, num_blocks, block_size, kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, num_blocks: int, block_size: int = 16,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.block_size = block_size
        self.k = torch.zeros(n_layers, num_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.allocator = BlockAllocator(num_blocks)

    def new_sequence(self) -> "PagedSequence":
        return PagedSequence(self)


class PagedSequence:
    """One sequence's view of the pool: a block table mapping logical positions to physical blocks.

    Same interface as ContiguousKVCache (write / read / advance / length), so the model can't tell them apart.
    """

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.block_table: list[int] = []      # logical block i -> physical block id
        self.length = 0

    def _ensure(self, end: int) -> None:
        """Grow the block table to cover `end` positions. All-or-nothing: if the pool can't supply every block
        needed, raise before taking any, so a failed request never holds blocks (no hold-and-wait deadlock)."""
        bs = self.pool.block_size
        need = -(-end // bs) - len(self.block_table)
        if need <= 0:
            return
        if need > self.pool.allocator.num_free:
            raise OutOfBlocks(f"need {need} more KV blocks, only {self.pool.allocator.num_free} free")
        self.block_table.extend(self.pool.allocator.allocate() for _ in range(need))

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[0]
        self._ensure(end)
        pos = torch.arange(start, end, device=self.pool.k.device)
        table = torch.tensor(self.block_table, device=self.pool.k.device)
        blocks, offsets = table[pos // self.pool.block_size], pos % self.pool.block_size
        self.pool.k[layer, blocks, offsets] = k.to(self.pool.k.dtype)
        self.pool.v[layer, blocks, offsets] = v.to(self.pool.v.dtype)

    def read(self, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather positions 0 .. end-1 from their blocks into contiguous [end, kv_heads, head_dim]."""
        nb = -(-end // self.pool.block_size)                  # ceil(end / block_size)
        table = torch.tensor(self.block_table[:nb], device=self.pool.k.device)
        k = self.pool.k[layer, table].flatten(0, 1)[:end]     # [nb*bs, H, d] -> first `end` rows
        v = self.pool.v[layer, table].flatten(0, 1)[:end]
        return k, v

    def advance(self, n: int) -> None:
        self.length += n

    def free(self) -> None:
        """Return this sequence's blocks to the pool (called when the request finishes)."""
        self.pool.allocator.free(self.block_table)
        self.block_table = []
        self.length = 0

    def bytes_used(self) -> int:
        p = self.pool
        return 2 * len(self.block_table) * p.k[0, 0].numel() * p.k.shape[0] * p.k.element_size()

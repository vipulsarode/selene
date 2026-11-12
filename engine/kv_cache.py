"""
The memory the engine writes into: one K tensor and one V tensor per layer,
each shaped [num_blocks, block_size, num_kv_heads, head_dim].

A block ID from the BlockManager is an index into dim 0 of these tensors.
A slot is (block ID, offset inside the block): where ONE token's K/V lives.

Get the config without downloading weights:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config.num_hidden_layers, config.num_key_value_heads, config.head_dim
"""
import torch


# ---------------------------------------------------------------------------
# Slot mapping: token position -> (block ID, offset)
# ---------------------------------------------------------------------------
def slot_for(block_table: list[int], pos: int, block_size: int = 16) -> tuple[int, int]:
    """Where does the token at position `pos` of this request live?
    Example: table [7, 2, 9], pos 40 -> (9, 8)."""
    ...

    block_idx = pos // block_size
    offset = pos % block_size

    return (block_table[block_idx], offset)

def slots_for_range(block_table: list[int], start: int, end: int,
                    block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Slots for positions start .. end-1, as two 1-D long tensors (block_ids, offsets).
    Prefill writes many tokens at once (start=0, end=prompt length);
    decode writes one (start=pos, end=pos+1)."""
    ...

    positions = torch.arange(start, end)

    block_idxs = positions // block_size
    offsets = positions % block_size

    return (torch.tensor(block_table)[block_idxs], offsets)


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------
class KVCache:
    def __init__(self, config, num_blocks: int, block_size: int = 16,
                 dtype: torch.dtype = torch.float32, device: str = "cpu"):
        """Allocate K and V for every layer, all zeros.
        self.k[layer] and self.v[layer]: [num_blocks, block_size, num_kv_heads, head_dim]"""
        ...

        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        
        self.k = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(config.num_hidden_layers)]
    
    @staticmethod
    def bytes_per_block(config, block_size: int = 16,
                        dtype: torch.dtype = torch.float32) -> int:
        """2 (K and V) x layers x block_size x kv_heads x head_dim x bytes per element.
        Hint: torch.tensor([], dtype=dtype).element_size() gives bytes per element."""
        ...

        return 2 * config.num_key_value_heads * config.num_hidden_layers * config.head_dim * block_size * dtype.itemsize

    @classmethod
    def num_blocks_for_budget(cls, config, budget_bytes: int, block_size: int = 16,
                              dtype: torch.dtype = torch.float32) -> int:
        """How many whole blocks fit in this many bytes?"""
        ...

        return budget_bytes // cls.bytes_per_block(config, block_size, dtype)


    def write(self, layer: int, block_ids: torch.Tensor, offsets: torch.Tensor,
              k: torch.Tensor, v: torch.Tensor) -> None:
        """Store new tokens' K/V for one layer.
        k, v: [num_new_tokens, num_kv_heads, head_dim]; block_ids, offsets: [num_new_tokens].
        Token i goes to self.k[layer][block_ids[i], offsets[i]].
        Hint: one line each with advanced indexing, no Python loop."""
        ...

        self.k[layer][block_ids, offsets] = k
        self.v[layer][block_ids, offsets] = v
    
    def gather(self, layer: int, block_table: list[int],
               num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read back one request's K/V for one layer as contiguous tensors,
        [num_tokens, num_kv_heads, head_dim] each, in position order.
        This is what model runner v1 hands to SDPA.
        Careful: the last block may be only partly filled. Which tokens do you keep?"""
        ...
        idx = torch.tensor(block_table)
        k = self.k[layer][idx]          # [num_blocks_in_table, 16, kv_heads, head_dim]
        v = self.v[layer][idx]
        k = k.reshape(-1, k.shape[2], k.shape[3])[:num_tokens]   # glue blocks, drop unused slots
        v = v.reshape(-1, v.shape[2], v.shape[3])[:num_tokens]
        return k, v


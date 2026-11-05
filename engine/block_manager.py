from collections import deque
from itertools import islice

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16, watermark: int = 0):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.watermark = watermark          # free blocks held back for running requests
        self.free_blocks = deque(range(num_blocks))       # every block ID starts free
        self.block_tables = {}              # req_id -> [physical block IDs], in order
        self.num_tokens = {}

    def num_free_blocks(self) -> int:
        """How many blocks are free right now."""
        ...
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """Would a new request with this many prompt tokens fit,
        while leaving `watermark` blocks untouched?"""
        ...
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        remaining_blocks = self.num_free_blocks() - blocks_needed

        return remaining_blocks >= self.watermark
        
    def allocate(self, req_id, num_tokens: int) -> list[int]:
        """Reserve every block the prompt needs, all at once.
        Returns the request's block table."""
        ...
        
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if self.can_allocate(num_tokens):
            self.block_tables[req_id] = [self.free_blocks.popleft() for _ in  range(blocks_needed)] 
            self.num_tokens[req_id] = num_tokens
        else:
            return None
        
        return self.block_tables[req_id]

    def append_slot(self, req_id):
        """Called after each new token is generated.
        Returns a new block ID if the last block was full, otherwise None."""
        ...

        c = self.num_tokens[req_id]

        if c%self.block_size==0:
            self.block_tables[req_id].append(self.free_blocks.popleft())
            self.num_tokens[req_id] = c + 1   
            return self.block_tables[req_id][-1]
        else:
            self.num_tokens[req_id] = c + 1   
            return None

    def can_append_slot(self, req_id) -> bool:
        """Can this running request take its next token without preemption?
        Ignores the watermark: running requests may use reserved blocks."""
        c = self.num_tokens[req_id]

        return not (c%self.block_size==0 and self.num_free_blocks() == 0)

    def free(self, req_id) -> None:
        """Give every block of this request back to the free list."""
        ...
        blocks = self.block_tables[req_id]
        self.free_blocks.extend(blocks)

        del self.block_tables[req_id]
        del self.num_tokens[req_id]
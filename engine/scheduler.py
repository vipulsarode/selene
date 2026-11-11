from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from engine.block_manager import BlockManager


class Status(Enum):          # named states, so a typo crashes instead of silently failing
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass                   # auto-generates __init__ from the fields below
class Request:
    req_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    eos_token_id: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING

    @property
    def num_tokens(self) -> int:
        """Prompt + generated so far. This is what you allocate for,
        on first admission AND when a preempted request comes back."""
        ...
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def is_finished(self) -> bool:
        """Hit EOS, or generated max_tokens."""
        ...
        if len(self.output_token_ids) == 0:
            return False
        
        return (self.output_token_ids[-1]==self.eos_token_id) or (len(self.output_token_ids) >= self.max_tokens)


@dataclass
class SchedulerOutput:
    prefill: list[Request]     # admitted this step: process all their tokens
    decode: list[Request]      # already running: one token each
    preempted: list[Request]   # sent back to waiting this step (for logging)


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_num_seqs: int = 64):
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs     # cap on how many run at once
        self.waiting = deque()               # front = next to admit
        self.running = []                    # oldest first, newest last

    def add_request(self, req: Request) -> None:
        """New requests join the BACK of waiting."""
        ...
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        """Decide what runs this step. v1: prefill-only OR decode-only.

        1. PREFILL: admit from the FRONT of waiting while
           - running is under max_num_seqs, and
           - the block manager says it fits.
           Stop at the first one that doesn't fit (no skipping ahead).
           Admitted -> allocate, mark RUNNING, add to running.
           If anyone was admitted, return a prefill-only batch.

        2. DECODE (nothing admitted): for each running request, oldest first,
           check it can get its next slot. If not, preempt the NEWEST
           running request and check again; stop if it preempted itself.
           Once it can, reserve the slot -> it goes in the decode batch.

        Careful in 2: you're removing from the list you're looping over.
        """
        ...

        so = SchedulerOutput(prefill=[], decode=[], preempted=[])

        while (len(self.running) < self.max_num_seqs) and (len(self.waiting) > 0) and self.bm.can_allocate(self.waiting[0].num_tokens):
            req = self.waiting.popleft()
            self.bm.allocate(req.req_id, req.num_tokens)
            req.status = Status.RUNNING
            self.running.append(req)
            so.prefill.append(req)

        if len(so.prefill) > 0:
            return so

        # safe: victims are always taken from the end
        for req in self.running:

            while not self.bm.can_append_slot(req.req_id):
                victim = self.running[-1]    
                so.preempted.append(victim)
                self._preempt(victim)
                if victim is req:
                    break

            if req.status != Status.RUNNING:
                continue

            self.bm.append_slot(req.req_id)
            so.decode.append(req)

        return so

    def _preempt(self, req: Request) -> None:
        """Recompute-style: give back its blocks, KEEP its tokens,
        mark WAITING, put it at the FRONT of waiting."""
        ...

        self.waiting.appendleft(req)
        req.status = Status.WAITING 
        self.running.remove(req)
        self.bm.free(req.req_id)

    def update(self, out: SchedulerOutput, new_tokens: dict) -> list[Request]:
        """After the model runs. new_tokens: req_id -> sampled token ID.
        Prefill requests get a token too (prefill samples the first one).
        Append each token; if finished -> free blocks, mark FINISHED,
        remove from running. Returns the requests that finished."""
        ...

        finished = []

        for req in out.prefill + out.decode:

            req.output_token_ids.append(new_tokens[req.req_id])
            if req.is_finished():
                finished.append(req)
                req.status = Status.FINISHED 
                self.bm.free(req.req_id)
                self.running.remove(req)

        return finished


    def has_unfinished(self) -> bool:
        """Anything in waiting or running? The engine loops while this is True."""
        ...
        return (len(self.waiting) > 0) or (len(self.running) > 0)
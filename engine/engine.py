import json

import torch
from transformers import AutoModelForCausalLM

from engine.block_manager import BlockManager
from engine.kv_cache import KVCache
from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler


class Engine:
    def __init__(self, hf_model, num_blocks, max_num_seqs=64):
        """Build the four pieces: BlockManager, KVCache, ModelRunner, Scheduler.
        The KVCache and the BlockManager must agree on num_blocks."""

        self.bm = BlockManager(num_blocks)
        self.kv = KVCache(hf_model.config, num_blocks)
        self.mr = ModelRunner(hf_model, self.kv)     
        self.scheduler = Scheduler(self.bm, max_num_seqs) 

    def add_request(self, req):
        """Hand it to the scheduler."""
        self.scheduler.add_request(req)

    def step(self):
        """One step: schedule -> run prefill/decode for each request -> update.
        Collect new tokens in a dict {req_id: token}. Return what finished."""
        
        new_tokens = {}
        scheduler_output= self.scheduler.schedule()

        for prefill_req in scheduler_output.prefill:
            token_ids = prefill_req.prompt_token_ids + prefill_req.output_token_ids
            tok = self.mr.prefill(token_ids, self.bm.block_tables[prefill_req.req_id])
            new_tokens[prefill_req.req_id] = tok
        for decode_req in scheduler_output.decode:
            token_id = decode_req.output_token_ids[-1]
            pos = decode_req.num_tokens - 1
            tok = self.mr.decode(token_id, pos, self.bm.block_tables[decode_req.req_id])
            new_tokens[decode_req.req_id] = tok

        finished_req = self.scheduler.update(scheduler_output, new_tokens)
        return finished_req

    def run(self):
        """Keep stepping while the scheduler has unfinished work.
        Return every finished request."""

        finished = []
        while self.scheduler.has_unfinished():
            finished_req = self.step()
            finished += finished_req

        return finished
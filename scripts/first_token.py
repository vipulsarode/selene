import json

import torch
from transformers import AutoModelForCausalLM

from engine.block_manager import BlockManager
from engine.kv_cache import KVCache
from engine.model_runner import ModelRunner

ref = json.load(open("reference/hf_outputs.json"))

hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
hf.eval()

NUM_BLOCKS = 16
bm = BlockManager(NUM_BLOCKS)
kv = KVCache(hf.config, NUM_BLOCKS)
mr = ModelRunner(hf, kv)

for i, r in enumerate(ref["results"]):
    prompt_ids = r["prompt_token_ids"]
    expected = r["output_token_ids"]
    req_id = f"p{i}"

    bt = bm.allocate(req_id, len(prompt_ids))
    tok = mr.prefill(prompt_ids, bt)        # first generated token
    out = [tok]
    pos = len(prompt_ids)                    # prompt took 0..n-1, so `tok` sits at n

    while len(out) < len(expected):
        bm.append_slot(req_id)               # reserve the slot at `pos` (new block if needed)
        tok = mr.decode(tok, pos, bt)        # feed the newest token in, get the next one
        out.append(tok)
        pos += 1

    bm.free(req_id)

    if out == expected:
        print(f"prompt {i}: MATCH, all {len(out)} tokens")
    else:
        first_bad = next(j for j, (a, b) in enumerate(zip(out, expected)) if a != b)
        print(f"prompt {i}: DIVERGES at token {first_bad}")
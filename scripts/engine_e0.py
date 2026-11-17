import json
import math

import torch
from transformers import AutoModelForCausalLM

from engine.engine import Engine
from engine.scheduler import Request

import sys
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"

ref = json.load(open("reference/hf_outputs.json"))
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
hf.eval()

BLOCK_SIZE = 16


def run_all(num_blocks):
    engine = Engine(hf, num_blocks, device = DEVICE)

    # count preemptions by wrapping the scheduler's _preempt (test-only trick)
    count = {"preempted": 0}
    original = engine.scheduler._preempt
    def counting_preempt(req):
        count["preempted"] += 1
        original(req)
    engine.scheduler._preempt = counting_preempt

    expected = {}
    for i, r in enumerate(ref["results"]):
        req_id = f"p{i}"
        expected[req_id] = r["output_token_ids"]
        engine.add_request(Request(req_id=req_id,
                                   prompt_token_ids=r["prompt_token_ids"],
                                   max_tokens=len(r["output_token_ids"]),
                                   eos_token_id=None))      # stop on length, same count as HF

    finished = engine.run()
    print(f"\nnum_blocks={num_blocks}, preemptions={count['preempted']}")
    for req in sorted(finished, key=lambda r: r.req_id):
        out, exp = req.output_token_ids, expected[req.req_id]
        if out == exp:
            print(f"  {req.req_id}: MATCH, all {len(out)} tokens")
        else:
            first_bad = next(j for j, (a, b) in enumerate(zip(out, exp)) if a != b)
            print(f"  {req.req_id}: DIVERGES at token {first_bad}")


# how many blocks each request needs at its longest (prompt + all outputs)
needs = [math.ceil((len(r["prompt_token_ids"]) + len(r["output_token_ids"])) / BLOCK_SIZE)
         for r in ref["results"]]

run_all(num_blocks=sum(needs))       # everyone fits: no preemption
run_all(num_blocks=max(needs) + 1)   # the biggest fits alone, but not all three: forces preemption
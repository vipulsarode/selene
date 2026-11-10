# Selene

An LLM inference engine built from scratch, with a paged KV cache, continuous batching, and preemption.

Selene is the serving counterpart to [Helios](#), my from-scratch distributed training framework. **Helios trains, Selene serves.** The goal is to understand how engines like vLLM actually work by building the core pieces myself and checking them against a trusted reference.

> 🚧 Work in progress. The block manager is done and tested; the scheduler is next.

## Roadmap

| Component | What it does | Status |
|---|---|---|
| Block manager | Hands out fixed-size KV cache blocks to requests and tracks which request owns which block | ✅ Done (28 tests) |
| Scheduler | Decides which requests run each step: admission, continuous batching, preemption | 🔨 Next |
| KV cache | Preallocated GPU tensors of shape `[num_blocks, block_size, kv_heads, head_dim]` | ⏳ |
| Model runner | Runs prefill and decode forward passes, reading/writing KV through block tables | ⏳ |
| HF reference | Greedy generation with Hugging Face Transformers, used as the correctness oracle | ⏳ |
| Engine | `add_request()` + step loop that ties everything together | ⏳ |
| CUDA graphs | Capture the decode step to cut kernel launch overhead | Stretch |

Development model: **Qwen3-0.6B** (small enough to iterate fast; the design doesn't depend on model size).

## Layout

```
selene/
├── engine/
│   ├── block_manager.py    # ✅ paged KV block allocation
│   ├── scheduler.py        # admission, batching, preemption
│   ├── kv_cache.py         # GPU KV tensors
│   ├── model_runner.py     # prefill / decode forward passes
│   └── engine.py           # add_request() + step loop
├── reference/
│   └── hf_reference.py     # Hugging Face oracle for output parity
└── tests/
    └── test_block_manager.py
```

## Running the tests

From the repo root:

```bash
python -m pytest tests/
```

The suite covers allocation rounding, mid-block vs. block-boundary appends, freeing, the watermark, double-free / unknown-request errors, and a 5,000-step random workload with preemption that checks the invariant after every operation.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023), the vLLM paper
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022), continuous batching
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI 2024), chunked prefill
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI 2024)


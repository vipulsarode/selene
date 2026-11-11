"""
HF reference outputs for E0: the oracle Selene has to match token for token.

Runs Qwen3-0.6B in float32 on CPU with greedy decoding and saves, per prompt:
  - prompt_token_ids   (Selene takes these as input, so both see identical tokens)
  - output_token_ids   (what Selene must reproduce exactly at batch size 1)
  - stopped_on_eos     (HF stops early on EOS; Selene must stop at the same place)
and the raw logits for every generated step, for debugging when tokens diverge.

Usage (from the repo root):
    python -m reference.hf_reference
    python -m reference.hf_reference --max-new-tokens 64
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

PROMPTS = [
    # short: fits in one 16-token block
    "The capital of France is",
    # medium: crosses a block boundary during generation
    "In distributed training, data parallelism splits the batch across GPUs, while tensor parallelism",
    # long (~40 tokens): starts in the 3rd block, exercises multi-block block tables
    "Once upon a time, in a small village at the edge of a dense forest, there lived an old "
    "clockmaker who spent his days repairing watches for the villagers. One morning, a stranger",
]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="reference/hf_outputs.json")
    ap.add_argument("--logits-out", default="reference/hf_logits.pt")
    args = ap.parse_args()

    # float32 on CPU: the most boring, exact version possible. An oracle should not be fast.
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]

    results, all_logits = [], []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt")          # raw text, no chat template
        prompt_ids = enc["input_ids"][0].tolist()

        out = model.generate(
            **enc,
            do_sample=False,                               # greedy: always the argmax token
            max_new_tokens=args.max_new_tokens,
            temperature=None, top_p=None, top_k=None,      # silence sampling-config warnings
            return_dict_in_generate=True,
            output_logits=True,                            # raw logits, before any processors
        )

        # generate returns prompt + new tokens glued together: keep only the new part
        new_ids = out.sequences[0, len(prompt_ids):].tolist()
        stopped_on_eos = len(new_ids) > 0 and new_ids[-1] in eos_ids

        results.append({
            "prompt": prompt,
            "prompt_token_ids": prompt_ids,
            "output_token_ids": new_ids,
            "stopped_on_eos": stopped_on_eos,
        })
        # one [vocab] row per generated token
        all_logits.append(torch.stack([step[0] for step in out.logits]))

        print(f"[{len(prompt_ids):>3} prompt tok -> {len(new_ids):>3} new]  "
              f"{prompt[:40]!r}... -> {tok.decode(new_ids)!r}")

    payload = {
        "model": args.model,
        "dtype": "float32",
        "device": "cpu",
        "max_new_tokens": args.max_new_tokens,
        "eos_token_ids": eos_ids,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    torch.save(all_logits, args.logits_out)
    print(f"\nsaved {args.out} and {args.logits_out}")


if __name__ == "__main__":
    main()
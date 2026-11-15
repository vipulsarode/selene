"""
Model runner v1: runs Qwen3 layer by layer, reading and writing K/V through the paged cache.

Strategy: borrow HF's *weights and small modules* (Linear layers, RMSNorms, MLP, rotary
embedding) but write the *forward pass and attention* yourself, so every K/V goes through
your KVCache. That's the part that makes it your engine.

The HF module names you need (from modeling_qwen3.py):
    hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
    hf.model.embed_tokens            token ids -> [n, hidden]
    hf.model.rotary_emb              (x, position_ids) -> (cos, sin)
    hf.model.layers[i]               one decoder layer:
        .input_layernorm             RMSNorm before attention
        .self_attn.q_proj / k_proj / v_proj / o_proj
        .self_attn.q_norm / k_norm   QK-norm (per head, over head_dim)
        .post_attention_layernorm    RMSNorm before the MLP
        .mlp                         the whole MLP, call it as-is
    hf.model.norm                    final RMSNorm
    hf.lm_head                       [hidden] -> [vocab] logits

Shapes: your cache stores one token as [kv_heads, head_dim].
HF's apply_rotary_pos_emb wants [batch, heads, seq, head_dim]. Transpose at the boundary.

v1 runs ONE request at a time (loop over the batch). Real batching comes later.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine.kv_cache import KVCache, slots_for_range


class ModelRunner:
    def __init__(self, hf_model, kv_cache: KVCache):
        """Keep the HF model (for its weights and modules) and the cache.
        Also save from the config: num_heads, num_kv_heads, head_dim."""
        ...

        self.hf_model = hf_model
        self.kv_cache = kv_cache

        self.num_heads = hf_model.config.num_attention_heads
        self.num_kv_heads = hf_model.config.num_key_value_heads
        self.head_dim = hf_model.config.head_dim

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], block_table: list[int]) -> int:
        """Process ALL of a request's tokens at once (prompt + any kept outputs, if it was
        preempted). Writes every token's K/V into the cache. Returns the next token (greedy).

        Steps:
          1. positions = 0 .. n-1  (absolute positions in this request)
          2. h = embed(token_ids)                          -> [n, hidden]
          3. cos, sin = rotary_emb(h, positions)           (once, shared by all layers)
          4. block_ids, offsets = slots_for_range(block_table, 0, n)
          5. for each layer: h = self._layer(i, h, cos, sin, block_ids, offsets,
                                             block_table, n, causal=True)
          6. final norm -> lm_head on the LAST position only -> argmax
        """
        ...

        n = len(token_ids)
        positions = torch.arange(n)
        h = self.hf_model.model.embed_tokens(torch.tensor(token_ids))
        cos, sin = self.hf_model.model.rotary_emb(h.unsqueeze(0), positions.unsqueeze(0))
        block_ids, offsets = slots_for_range(block_table, 0, n)
        for layer_idx in range(len(self.hf_model.model.layers)): 
            h = self._layer(layer_idx, h, cos, sin, block_ids, offsets,
                                             block_table, n, causal=True)
        h = self.hf_model.model.norm(h)  
        out = self.hf_model.lm_head(h[-1])
        return torch.argmax(out, dim = -1).item()   


    @torch.inference_mode()
    def decode(self, token_id: int, position: int, block_table: list[int]) -> int:
        h = self.hf_model.model.embed_tokens(torch.tensor([token_id]))
        cos, sin = self.hf_model.model.rotary_emb(h.unsqueeze(0), torch.tensor([position]).unsqueeze(0))
        block_ids, offsets = slots_for_range(block_table, position, position+1)
        for layer_idx in range(len(self.hf_model.model.layers)): 
            h = self._layer(layer_idx, h, cos, sin, block_ids, offsets,
                                             block_table, position+1, causal=False)
        h = self.hf_model.model.norm(h)  
        out = self.hf_model.lm_head(h[-1])
        return torch.argmax(out, dim = -1).item()   

    def _layer(self, i, h, cos, sin, block_ids, offsets, block_table,
            num_cached_after_write, causal) -> torch.Tensor:
        layer = self.hf_model.model.layers[i]
        attn = layer.self_attn
        n = h.shape[0]                                    # fix 1

        residual = h
        x = layer.input_layernorm(h)

        q = attn.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = attn.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(x).view(n, self.num_kv_heads, self.head_dim)

        q = attn.q_norm(q)
        k = attn.k_norm(k)

        # fix 2: [n, heads, d] -> [1, n, heads, d], and tell HF that seq is dim 1
        q, k = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q, k = q.squeeze(0), k.squeeze(0)

        self.kv_cache.write(i, block_ids, offsets, k, v)
        K, V = self.kv_cache.gather(i, block_table, num_cached_after_write)
        out = self._attention(q, K, V, causal)            # [n, 16, 128]

        h = residual + attn.o_proj(out.reshape(n, -1))    # fixes 3 + 4
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h


    def _attention(self, q, k, v, causal: bool) -> torch.Tensor:
        """q: [n_q, 16, d]; k, v: [n_kv, 8, d]  ->  [n_q, 16, d]
        Use F.scaled_dot_product_attention with enable_gqa=True.
        SDPA wants [batch, heads, seq, d], so transpose in and out.
        Prefill: causal=True (n_q == n_kv). Decode: causal=False (1 query sees everything)."""
        ...

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal = causal)

        out = out.squeeze(0).transpose(0, 1)

        return out
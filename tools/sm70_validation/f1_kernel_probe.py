"""Standalone probe: TRITON_PAGED kernels on real GPU vs torch reference."""

import torch
import sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
from vllm.v1.attention.ops.triton_paged_attn import (
    store_kvcache_paged,
    paged_decode_attention,
    varlen_paged_prefill_attention,
)

torch.manual_seed(0)
dev = "cuda:0"

B, KVH, H, D, BS = 8, 2, 4, 64, 16
SEQ, NS = 32, 2
SCALE = D**-0.5

k_cache = torch.zeros(B, BS, KVH, D, dtype=torch.float16, device=dev)
v_cache = torch.zeros(B, BS, KVH, D, dtype=torch.float16, device=dev)

slot_mapping = torch.cat(
    [
        torch.arange(0, SEQ),
        torch.arange(2 * BS, 2 * BS + SEQ),
    ]
).to(dev)
k = torch.randn(NS, SEQ, KVH, D, dtype=torch.float16, device=dev) * 0.3
v = torch.randn(NS, SEQ, KVH, D, dtype=torch.float16, device=dev) * 0.3
q = torch.randn(NS * SEQ, H, D, dtype=torch.float16, device=dev) * 0.3
seq_lens = torch.tensor([SEQ, SEQ], dtype=torch.int32, device=dev)
block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, SEQ, 2 * SEQ], dtype=torch.int32, device=dev)

store_kvcache_paged(
    k.view(NS * SEQ, KVH, D),
    v.view(NS * SEQ, KVH, D),
    k_cache,
    v_cache,
    slot_mapping,
    BS,
)
print("store OK — nonzero ratio:", (k_cache != 0).float().mean().item())


def ref_attention(queries, slots_per_seq, ntok):
    out = torch.zeros(NS, ntok, H, D, dtype=torch.float32)
    for s in range(NS):
        slots = slots_per_seq[s]
        blk, off = slots // BS, slots % BS
        Ks = k_cache[blk, off].repeat_interleave(H // KVH, dim=1).float()
        Vs = v_cache[blk, off].repeat_interleave(H // KVH, dim=1).float()
        n = len(slots)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=1)
        for h in range(H):
            att = (queries[s, :, h].float() @ Ks[:, h].T) * SCALE
            att.masked_fill_(mask, float("-inf"))
            out[s, :, h] = torch.softmax(att, -1) @ Vs[:, h]
    return out


out_prefill = torch.zeros(NS * SEQ, H, D, dtype=torch.float16, device=dev)
varlen_paged_prefill_attention(
    out_prefill,
    q,
    k_cache,
    v_cache,
    qsl,
    seq_lens,
    block_tables,
    H,
    KVH,
    D,
    BS,
    SEQ,
    SCALE,
)
ref = ref_attention(
    q.view(NS, SEQ, H, D), [slot_mapping[:SEQ], slot_mapping[SEQ:]], SEQ
).to(dev)
print(
    "prefill varlen max_diff:",
    (out_prefill.view(NS, SEQ, H, D).float() - ref).abs().max().item(),
)

# decode: 1 token nuevo por secuencia
q_dec = torch.randn(NS, H, D, dtype=torch.float16, device=dev) * 0.3
slot_dec = torch.tensor([SEQ, 2 * BS + SEQ], dtype=torch.int32, device=dev)
store_kvcache_paged(k[:, -1], v[:, -1], k_cache, v_cache, slot_dec, BS)
seq_lens_dec = torch.tensor([SEQ + 1, SEQ + 1], dtype=torch.int32, device=dev)
out_dec = torch.zeros(NS, H, D, dtype=torch.float16, device=dev)
paged_decode_attention(
    out_dec,
    q_dec,
    k_cache,
    v_cache,
    block_tables,
    seq_lens_dec,
    torch.tensor([0, 1, 2], dtype=torch.int32, device=dev),
    H,
    KVH,
    D,
    BS,
    SCALE,
)
ref_dec = torch.zeros(NS, H, D, dtype=torch.float32, device=dev)
for s in range(NS):
    slots = torch.cat([slot_mapping[s * SEQ : (s + 1) * SEQ], slot_dec[s : s + 1]])
    blk, off = slots // BS, slots % BS
    Ks = k_cache[blk, off].repeat_interleave(H // KVH, dim=1).float()
    Vs = v_cache[blk, off].repeat_interleave(H // KVH, dim=1).float()
    for h in range(H):
        att = (q_dec[s, h].float() @ Ks[:, h].T) * SCALE
        ref_dec[s, h] = torch.softmax(att, -1) @ Vs[:, h]
print("decode max_diff:", (out_dec.float() - ref_dec).abs().max().item())

# === reproduccion del bug E2E: caches como vistas unbind(1) no contiguas ===
pool = torch.zeros(B, 2, BS, KVH, D, dtype=torch.float16, device=dev)
kc2, vc2 = pool.unbind(1)
store_kvcache_paged(k.view(NS * SEQ, KVH, D), v.view(NS * SEQ, KVH, D), kc2, vc2, slot_mapping, BS)
out2 = torch.zeros(NS * SEQ, H, D, dtype=torch.float16, device=dev)
varlen_paged_prefill_attention(
    out2, q, kc2, vc2, qsl, seq_lens, block_tables, H, KVH, D, BS, SEQ, SCALE,
)
print("prefill con vistas unbind(1) max_diff:", (out2.view(NS, SEQ, H, D).float() - ref).abs().max().item())
print("kc2 contiguo:", kc2.is_contiguous(), "| stride0:", kc2.stride(0), "vs esperado", BS * KVH * D)

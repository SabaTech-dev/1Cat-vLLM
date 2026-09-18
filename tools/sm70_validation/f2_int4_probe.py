"""Sonda F2.1: kernels TRITON_PAGED con cache int4 empaquetada (scatter store)."""

import torch, sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
from vllm.v1.attention.backends.triton_paged import _quantize_int4_pth
from vllm.v1.attention.ops.triton_paged_attn import (
    paged_decode_attention,
    varlen_paged_prefill_attention,
)

torch.manual_seed(0)
dev = "cuda:0"
B, KVH, H, D, BS = 8, 2, 4, 128, 16
SEQ, NS = 32, 2
SCALE = D**-0.5
PHS = D // 2 + 4  # 68: 64 bytes de datos + 4 de escala fp32

k = torch.randn(NS, SEQ, KVH, D, dtype=torch.float16, device=dev) * 0.3
v = torch.randn(NS, SEQ, KVH, D, dtype=torch.float16, device=dev) * 0.3
q = torch.randn(NS * SEQ, H, D, dtype=torch.float16, device=dev) * 0.3
slot_mapping = torch.cat([torch.arange(0, SEQ), torch.arange(2 * BS, 2 * BS + SEQ)]).to(
    dev
)
seq_lens = torch.tensor([SEQ, SEQ], dtype=torch.int32, device=dev)
block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, SEQ, 2 * SEQ], dtype=torch.int32, device=dev)


def scatter(cache, packed, slots):
    ok = slots >= 0
    cache[slots[ok] // BS, slots[ok] % BS] = packed[ok]


def dequant(cache):
    data = cache[..., : D // 2]
    sb = cache[..., D // 2 : PHS].contiguous().view(torch.float32)
    lo = (data & 0xF).float() - 8.0
    hi = ((data >> 4) & 0xF).float() - 8.0
    return torch.stack([lo, hi], dim=-1).reshape(*cache.shape[:-1], D) * sb


def ref_attention(queries, slots_per_seq, ntok):
    out = torch.zeros(NS, ntok, H, D, dtype=torch.float32)
    for s in range(NS):
        slots = slots_per_seq[s]
        blk, off = slots // BS, slots % BS
        Ks = dequant(kc4[blk, off]).repeat_interleave(H // KVH, dim=1)
        Vs = dequant(vc4[blk, off]).repeat_interleave(H // KVH, dim=1)
        n = len(slots)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1)
        for h in range(H):
            att = (queries[s, :, h].float() @ Ks[:, h].T) * SCALE
            att.masked_fill_(mask, float("-inf"))
            out[s, :, h] = torch.softmax(att, -1) @ Vs[:, h]
    return out


# caches int4 + store via scatter (los slots -1 se saltan)
kc4 = torch.zeros(B, BS, KVH, PHS, dtype=torch.uint8, device=dev)
vc4 = torch.zeros(B, BS, KVH, PHS, dtype=torch.uint8, device=dev)
kq = _quantize_int4_pth(k.view(NS * SEQ, KVH, D))
vq = _quantize_int4_pth(v.view(NS * SEQ, KVH, D))
scatter(kc4, kq, slot_mapping)
scatter(vc4, vq, slot_mapping)
print("store int4 OK — bytes no nulos:", round((kc4 != 0).float().mean().item(), 3))

# prefill int4
out_p = torch.zeros(NS * SEQ, H, D, dtype=torch.float16, device=dev)
varlen_paged_prefill_attention(
    out_p,
    q,
    kc4,
    vc4,
    qsl,
    seq_lens,
    block_tables,
    H,
    KVH,
    D,
    BS,
    SEQ,
    SCALE,
    kv_int4=True,
)
ref = ref_attention(
    q.view(NS, SEQ, H, D), [slot_mapping[:SEQ], slot_mapping[SEQ:]], SEQ
).to(dev)
d1 = (out_p.view(NS, SEQ, H, D).float() - ref).abs().max().item()
print("prefill int4 max_diff vs ref-dequant:", round(d1, 4))

# decode int4 (1 token nuevo por secuencia)
q_dec = torch.randn(NS, H, D, dtype=torch.float16, device=dev) * 0.3
slot_dec = torch.tensor([SEQ, 2 * BS + SEQ], dtype=torch.int32, device=dev)
scatter(kc4, _quantize_int4_pth(k[:, -1].contiguous().view(NS, KVH, D)), slot_dec)
scatter(vc4, _quantize_int4_pth(v[:, -1].contiguous().view(NS, KVH, D)), slot_dec)
seq_lens_dec = torch.tensor([SEQ + 1, SEQ + 1], dtype=torch.int32, device=dev)
out_d = torch.zeros(NS, H, D, dtype=torch.float16, device=dev)
paged_decode_attention(
    out_d,
    q_dec,
    kc4,
    vc4,
    block_tables,
    seq_lens_dec,
    torch.tensor([0, 1, 2], dtype=torch.int32, device=dev),
    H,
    KVH,
    D,
    BS,
    SCALE,
    kv_int4=True,
)
ref_d = torch.zeros(NS, H, D, dtype=torch.float32, device=dev)
for s in range(NS):
    slots = torch.cat([slot_mapping[s * SEQ : (s + 1) * SEQ], slot_dec[s : s + 1]])
    blk, off = slots // BS, slots % BS
    Ks = dequant(kc4[blk, off]).repeat_interleave(H // KVH, dim=1)
    Vs = dequant(vc4[blk, off]).repeat_interleave(H // KVH, dim=1)
    for h in range(H):
        att = (q_dec[s, h].float() @ Ks[:, h].T) * SCALE
        ref_d[s, h] = torch.softmax(att, -1) @ Vs[:, h]
d2 = (out_d.float() - ref_d).abs().max().item()
print("decode int4 max_diff vs ref-dequant:", round(d2, 4))
print("sonda completada")

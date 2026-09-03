"""Sonda F2.1 hibrido: K int8-PTH + V int4-PTH en TRITON_PAGED.
Incluye OUTLIER RoPE-style en K (dim 0 y 63 con magnitud 20x) — el caso
que rompio el int4 simetrico puro."""

import torch, sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
from vllm.v1.attention.backends.triton_paged import (
    _quantize_int8_pth,
    _quantize_int4_pth,
)
from vllm.v1.attention.ops.triton_paged_attn import (
    store_kvcache_paged,
    paged_decode_attention,
    varlen_paged_prefill_attention,
)

torch.manual_seed(0)
dev = "cuda:0"
B, KVH, H, D, BS = 8, 2, 4, 128, 16
SEQ, NS = 32, 2
SCALE = D**-0.5
K_HS = D + 4  # int8 + fp32 scale
V_HS = D // 2 + 4  # int4 packed + fp32 scale

k = torch.randn(NS, SEQ, KVH, D, dtype=torch.float16, device=dev) * 0.3
v = torch.randn(NS, SEQ, KVH, D, dtype=torch.float16, device=dev) * 0.3
# outlier RoPE-style: dos dims con magnitud 20x (post-rotacion)
k[..., :, 0] *= 20.0
k[..., :, 63] *= 20.0
q = torch.randn(NS * SEQ, H, D, dtype=torch.float16, device=dev) * 0.3
slot_mapping = torch.cat([torch.arange(0, SEQ), torch.arange(2 * BS, 2 * BS + SEQ)]).to(
    dev
)
seq_lens = torch.tensor([SEQ, SEQ], dtype=torch.int32, device=dev)
block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=dev)
qsl = torch.tensor([0, SEQ, 2 * SEQ], dtype=torch.int32, device=dev)


def dequant_k(cache):
    data = cache[..., :D].contiguous().view(torch.int8).float()
    sb = cache[..., D:K_HS].contiguous().view(torch.float32)
    return data * sb


def dequant_v(cache):
    data = cache[..., : D // 2]
    sb = cache[..., D // 2 : V_HS].contiguous().view(torch.float32)
    lo = (data & 0xF).float() - 8.0
    hi = ((data >> 4) & 0xF).float() - 8.0
    return torch.stack([lo, hi], dim=-1).reshape(*cache.shape[:-1], D) * sb


def ref_attention(queries, slots_per_seq, ntok):
    out = torch.zeros(NS, ntok, H, D, dtype=torch.float32)
    for s in range(NS):
        slots = slots_per_seq[s]
        blk, off = slots // BS, slots % BS
        Ks = dequant_k(kc4[blk, off]).repeat_interleave(H // KVH, dim=1)
        Vs = dequant_v(vc4[blk, off]).repeat_interleave(H // KVH, dim=1)
        n = len(slots)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1)
        for h in range(H):
            att = (queries[s, :, h].float() @ Ks[:, h].T) * SCALE
            att.masked_fill_(mask, float("-inf"))
            out[s, :, h] = torch.softmax(att, -1) @ Vs[:, h]
    return out


kc4 = torch.zeros(B, BS, KVH, K_HS, dtype=torch.uint8, device=dev)
vc4 = torch.zeros(B, BS, KVH, V_HS, dtype=torch.uint8, device=dev)
kq = _quantize_int8_pth(k.view(NS * SEQ, KVH, D))
vq = _quantize_int4_pth(v.view(NS * SEQ, KVH, D))
store_kvcache_paged(kq, vq, kc4, vc4, slot_mapping, BS)
print("store hibrido OK — K no nulos:", round((kc4 != 0).float().mean().item(), 3))

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
    k_int8=True,
    v_int4=True,
)
ref = ref_attention(
    q.view(NS, SEQ, H, D), [slot_mapping[:SEQ], slot_mapping[SEQ:]], SEQ
).to(dev)
d1 = (out_p.view(NS, SEQ, H, D).float() - ref).abs().max().item()
print("prefill hibrido max_diff (con outliers K):", round(d1, 4))

q_dec = torch.randn(NS, H, D, dtype=torch.float16, device=dev) * 0.3
slot_dec = torch.tensor([SEQ, 2 * BS + SEQ], dtype=torch.int32, device=dev)
store_kvcache_paged(
    _quantize_int8_pth(k[:, -1].contiguous().view(NS, KVH, D)),
    _quantize_int4_pth(v[:, -1].contiguous().view(NS, KVH, D)),
    kc4,
    vc4,
    slot_dec,
    BS,
)
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
    k_int8=True,
    v_int4=True,
)
ref_d = torch.zeros(NS, H, D, dtype=torch.float32, device=dev)
for s in range(NS):
    slots = torch.cat([slot_mapping[s * SEQ : (s + 1) * SEQ], slot_dec[s : s + 1]])
    blk, off = slots // BS, slots % BS
    Ks = dequant_k(kc4[blk, off]).repeat_interleave(H // KVH, dim=1)
    Vs = dequant_v(vc4[blk, off]).repeat_interleave(H // KVH, dim=1)
    for h in range(H):
        att = (q_dec[s, h].float() @ Ks[:, h].T) * SCALE
        ref_d[s, h] = torch.softmax(att, -1) @ Vs[:, h]
d2 = (out_d.float() - ref_d).abs().max().item()
print("decode hibrido max_diff (con outliers K):", round(d2, 4))
print("sonda hibrida completada")

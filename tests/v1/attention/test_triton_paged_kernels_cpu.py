# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical tests for the TRITON_PAGED kernels.

Runs the Triton kernels through the interpreter
(``TRITON_INTERPRET=1``), so the whole suite executes on machines
without a GPU. Tolerances account for fp16 accumulation differences
between the kernels (fp32 online softmax) and the dense fp16 reference.
"""

import os

# Must be set before the kernels (and Triton) are imported.
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

from vllm.v1.attention.ops.triton_paged_attn import (
    paged_decode_attention,
    store_kvcache_paged,
    varlen_paged_prefill_attention,
)

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
NUM_HEADS = 4
GQA = NUM_HEADS // NUM_KV_HEADS
SCALE = 1.0 / (8.0**0.5)
TOL = 3e-2


def _build_tables(seq_lens, query_lens):
    """Allocate a block table + slot mapping with disjoint blocks."""
    num_seqs = len(seq_lens)
    max_blocks_per_seq = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens)
    block_table = torch.full((num_seqs, max_blocks_per_seq), -1, dtype=torch.int32)
    next_block = 0
    for i, s in enumerate(seq_lens):
        for b in range((s + BLOCK_SIZE - 1) // BLOCK_SIZE):
            block_table[i, b] = next_block
            next_block += 1

    slots = []
    for i in range(num_seqs):
        ctx = seq_lens[i] - query_lens[i]
        for j in range(query_lens[i]):
            pos = ctx + j
            slots.append(
                int(block_table[i, pos // BLOCK_SIZE]) * BLOCK_SIZE + pos % BLOCK_SIZE
            )
    return block_table, torch.tensor(slots, dtype=torch.int64), next_block


def _reference(q, k_cache, v_cache, block_table, seq_lens, query_lens):
    """Dense PyTorch reference with paged gather + causal masking."""
    out = torch.zeros_like(q)
    qo = 0
    for i, qlen in enumerate(query_lens):
        ctx = seq_lens[i] - qlen
        ks, vs = [], []
        for pos in range(seq_lens[i]):
            pb = int(block_table[i, pos // BLOCK_SIZE])
            off = pos % BLOCK_SIZE
            ks.append(k_cache[pb, off])
            vs.append(v_cache[pb, off])
        k = torch.stack(ks).repeat_interleave(GQA, dim=1)
        v = torch.stack(vs).repeat_interleave(GQA, dim=1)
        qi = q[qo : qo + qlen]
        att = torch.einsum("lhd,shd->lhs", qi.float(), k.float()) * SCALE
        mask = (
            torch.arange(seq_lens[i])[None, :] > torch.arange(ctx, ctx + qlen)[:, None]
        )
        p = att.masked_fill(mask.unsqueeze(1), float("-inf")).softmax(-1)
        out[qo : qo + qlen] = torch.einsum("lhs,shd->lhd", p.to(v.dtype), v)
        qo += qlen
    return out


def test_store_kvcache_exact_roundtrip():
    torch.manual_seed(0)
    seq_lens, query_lens = [1, 17, 33, 64], [1, 1, 1, 1]
    num_tokens = sum(query_lens)
    k_cache = torch.zeros(16, BLOCK_SIZE, NUM_KV_HEADS, 64).half()
    v_cache = torch.zeros_like(k_cache)
    k_new = torch.randn(num_tokens, NUM_KV_HEADS, 64).half()
    v_new = torch.randn(num_tokens, NUM_KV_HEADS, 64).half()
    block_table, slot_mapping, _ = _build_tables(seq_lens, query_lens)

    store_kvcache_paged(k_new, v_new, k_cache, v_cache, slot_mapping, BLOCK_SIZE)
    for t in range(num_tokens):
        slot = int(slot_mapping[t])
        pb, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
        assert torch.equal(k_cache[pb, off], k_new[t])
        assert torch.equal(v_cache[pb, off], v_new[t])


def test_paged_decode_attention_matches_reference():
    torch.manual_seed(0)
    head_dim = 64
    seq_lens, query_lens = [1, 17, 33, 64], [1, 1, 1, 1]
    num_tokens = sum(query_lens)
    block_table, slot_mapping, num_blocks = _build_tables(seq_lens, query_lens)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim).half()
    v_cache = torch.randn_like(k_cache)
    # Write the current tokens through the store kernel (mirrors the
    # engine flow: do_kv_cache_update runs before attention).
    k_new = torch.randn(num_tokens, NUM_KV_HEADS, head_dim).half()
    v_new = torch.randn(num_tokens, NUM_KV_HEADS, head_dim).half()
    store_kvcache_paged(k_new, v_new, k_cache, v_cache, slot_mapping, BLOCK_SIZE)

    q = torch.randn(num_tokens, NUM_HEADS, head_dim).half()
    out = torch.zeros_like(q)
    paged_decode_attention(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.arange(0, num_tokens + 1, dtype=torch.int32),
        NUM_HEADS,
        NUM_KV_HEADS,
        head_dim,
        BLOCK_SIZE,
        SCALE,
    )
    ref = _reference(q, k_cache, v_cache, block_table, seq_lens, query_lens)
    assert (out.float() - ref.float()).abs().max().item() < TOL


@pytest.mark.parametrize("head_dim", [64, 128])
def test_varlen_prefill_attention_matches_reference(head_dim):
    torch.manual_seed(1)
    # First prefill, chunked prefill with cached prefix, and a long
    # extend spanning multiple query tiles.
    seq_lens = [8, 40, 97]
    query_lens = [8, 16, 33]
    num_tokens = sum(query_lens)
    block_table, slot_mapping, num_blocks = _build_tables(seq_lens, query_lens)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim).half()
    v_cache = torch.randn_like(k_cache)
    k_new = torch.randn(num_tokens, NUM_KV_HEADS, head_dim).half()
    v_new = torch.randn(num_tokens, NUM_KV_HEADS, head_dim).half()
    store_kvcache_paged(k_new, v_new, k_cache, v_cache, slot_mapping, BLOCK_SIZE)

    q = torch.randn(num_tokens, NUM_HEADS, head_dim).half()
    out = torch.zeros_like(q)
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), 0)), dtype=torch.int32
    )
    varlen_paged_prefill_attention(
        out,
        q,
        k_cache,
        v_cache,
        cu,
        torch.tensor(seq_lens, dtype=torch.int32),
        block_table,
        NUM_HEADS,
        NUM_KV_HEADS,
        head_dim,
        BLOCK_SIZE,
        max(query_lens),
        SCALE,
    )
    ref = _reference(q, k_cache, v_cache, block_table, seq_lens, query_lens)
    assert (out.float() - ref.float()).abs().max().item() < TOL

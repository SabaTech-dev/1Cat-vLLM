# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained Triton paged-attention kernels for the TRITON_PAGED backend.

This module is intentionally dependency-free (only Triton + PyTorch). It is
ported from the nano-vllm / MinivLLM reference implementations and adapted
to the vLLM KV cache layout:

    kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]

After ``kv_cache.unbind(1)`` each half is laid out as

    k_cache / v_cache: [num_blocks, block_size, num_kv_heads, head_size]

which matches the MinivLLM paged cache layout exactly. ``slot_mapping``
uses the vLLM convention of a flat slot index:

    slot = physical_block_idx * block_size + offset_in_block

References:
    - https://github.com/GeeeekExplorer/nano-vllm  (nanovllm/layers/attention.py)
    - https://github.com/Wenyueh/MinivLLM         (src/myvllm/layers/attention.py)
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Consistent softmax trick constants (vLLM Triton convention): fold log2(e)
# into the qk scale so the online softmax can use exp2().
_SM_SCALE_LOG2E_CONSTANT = 1.4426950408889634


@triton.jit
def _store_kvcache_paged_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_block_stride,
    v_block_stride,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_padded: tl.constexpr,
    block_size: tl.constexpr,
):
    """Store keys and values into the paged KV cache.

    Grid layout: (num_tokens, num_kv_heads). Each program copies the
    head_dim elements of one (token, head) pair from the dense projection
    output into its paged cache slot. Tokens with slot == -1 (padding) are
    skipped. head_dim_padded is next_power_of_2(head_dim); lanes beyond
    head_dim are masked off so non-power-of-2 packed slots (int4) store
    correctly.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    offs_d = tl.arange(0, head_dim_padded)
    mask_d = offs_d < head_dim
    # Input: (num_tokens, num_kv_heads, head_dim), assumed contiguous.
    input_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim + offs_d
    k_offset = block_idx * k_block_stride + (
        block_offset * num_kv_heads * head_dim
        + head_idx * head_dim
        + offs_d
    )
    v_offset = block_idx * v_block_stride + (
        block_offset * num_kv_heads * head_dim
        + head_idx * head_dim
        + offs_d
    )
    key = tl.load(key_ptr + input_offset, mask=mask_d, other=0)
    value = tl.load(value_ptr + input_offset, mask=mask_d, other=0)
    tl.store(k_cache_ptr + k_offset, key)
    tl.store(v_cache_ptr + v_offset, value)


def store_kvcache_paged(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter-write ``key``/``value`` into the paged KV cache.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) flat slot indices (-1 = skip)
        block_size: tokens per cache block
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    assert slot_mapping.numel() >= num_tokens

    # The kernel uses slot_mapping's length to determine the number of
    # actual tokens, mirroring reshape_and_cache_flash semantics (slot
    # mapping is not padded even when key/value are).
    num_tokens = slot_mapping.shape[0]

    grid = (num_tokens, num_kv_heads)
    _store_kvcache_paged_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_cache.stride(0),
        v_cache.stride(0),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_padded=triton.next_power_of_2(head_dim),
        block_size=block_size,
    )


@triton.jit
def _paged_decode_attention_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    qo_offset_ptr,
    scale_log2,
    k_block_stride,
    v_block_stride,
    KV_INT4: tl.constexpr,
    PACKED_HS: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Paged attention decode kernel (one query token per sequence).

    Grid layout: (num_seqs, num_heads). Each program computes one
    (sequence, query head) row of the output by streaming the sequence's
    KV blocks through an online (flash-style) softmax in fp32.

    The block table is gathered per token rather than per chunk, so a
    chunk may straddle multiple, non-adjacent physical blocks (no
    relationship between BLOCK_N and block_size is assumed).
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA: map query head onto its KV head.
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    seq_len = tl.load(seq_lens_ptr + batch_idx)

    # Query: (num_tokens, num_heads, head_dim). For decode each sequence
    # contributes one query token; its row offset comes from
    # query_start_loc so padded/scheduler-reordered batches stay correct.
    qo = tl.load(qo_offset_ptr + batch_idx)
    offs_d = tl.arange(0, head_dim)
    offs_half = tl.arange(0, head_dim // 2)
    q_offset = qo * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset).to(tl.float32)

    # Packed int4 KV: even/odd head lanes split so the byte loads stay
    # [head_dim//2, BLOCK_N]; masked/zeroed slots carry a zero fp32 scale
    # which zeroes their dequantized contribution.
    if KV_INT4:
        qe, qo_ = tl.split(tl.reshape(q, (head_dim // 2, 2)))
        acc_e = tl.zeros([head_dim // 2], dtype=tl.float32)
        acc_o = tl.zeros([head_dim // 2], dtype=tl.float32)
    else:
        acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -float("inf")

    num_chunks = tl.cdiv(seq_len, BLOCK_N)
    for chunk_idx in range(num_chunks):
        token_start = chunk_idx * BLOCK_N
        offs_n = token_start + tl.arange(0, BLOCK_N)
        logical_block = offs_n // block_size
        offs_in_block = offs_n % block_size

        in_range = offs_n < seq_len
        valid = in_range & (logical_block < max_num_blocks_per_seq)

        # One block-table entry per token: a chunk is free to span several
        # physical blocks. Masked lanes get physical block 0 so all
        # computed addresses stay inside the cache.
        physical_block = tl.load(
            block_tables_ptr + batch_idx * max_num_blocks_per_seq + logical_block,
            mask=valid,
            other=-1,
        )
        valid = valid & (physical_block != -1)
        physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

        if KV_INT4:
            # Cache head slot: PACKED_HS = head_dim//2 data bytes plus a
            # trailing fp32 scale (4 bytes) per (slot, head).
            slot_base_k = (
                offs_in_block[None, :] * (num_kv_heads * PACKED_HS)
                + kv_head_idx * PACKED_HS
            )
            byte_off_k = (
                physical_block[None, :] * k_block_stride
                + slot_base_k
                + offs_half[:, None]
            )
            raw_k = tl.load(k_cache_ptr + byte_off_k, mask=valid[None, :],
                            other=0)
            k_lo = (raw_k & 0xF).to(tl.float32) - 8.0
            k_hi = ((raw_k >> 4) & 0xF).to(tl.float32) - 8.0
            scale_ptrs = tl.cast(
                k_cache_ptr + physical_block[None, :] * k_block_stride
                + slot_base_k + head_dim // 2,
                tl.pointer_type(tl.uint32),
            )
            k_scale = tl.load(scale_ptrs, mask=valid[None, :],
                              other=0).to(tl.float32, bitcast=True)
            ke = (k_lo * k_scale).to(tl.float16)
            ko = (k_hi * k_scale).to(tl.float16)
            score = (tl.sum(qe[:, None] * ke, axis=0)
                     + tl.sum(qo_[:, None] * ko, axis=0)) * scale_log2
        else:
            # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
            kv_offset_k = (
                physical_block[None, :] * k_block_stride
                + offs_in_block[None, :] * (num_kv_heads * head_dim)
                + kv_head_idx * head_dim
                + offs_d[:, None]
            )
            k = tl.load(k_cache_ptr + kv_offset_k, mask=valid[None, :], other=0.0)
            k = k.to(tl.float32)
            score = tl.sum(q[:, None] * k, axis=0) * scale_log2
        qk = tl.where(valid, score, -float("inf"))

        # Online softmax update (exp2 trick; scale already folds log2(e)).
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)

        if KV_INT4:
            acc_e = acc_e * alpha
            acc_o = acc_o * alpha
        else:
            acc = acc * alpha
        l_i = l_i * alpha

        weight = tl.where(valid, p, 0.0)
        if KV_INT4:
            byte_off_v = (
                physical_block[None, :] * v_block_stride
                + offs_in_block[None, :] * (num_kv_heads * PACKED_HS)
                + kv_head_idx * PACKED_HS
                + offs_half[:, None]
            )
            raw_v = tl.load(v_cache_ptr + byte_off_v, mask=valid[None, :],
                            other=0)
            v_lo = (raw_v & 0xF).to(tl.float32) - 8.0
            v_hi = ((raw_v >> 4) & 0xF).to(tl.float32) - 8.0
            scale_ptrsv = tl.cast(
                v_cache_ptr + physical_block[None, :] * v_block_stride
                + offs_in_block[None, :] * (num_kv_heads * PACKED_HS)
                + kv_head_idx * PACKED_HS
                + head_dim // 2,
                tl.pointer_type(tl.uint32),
            )
            v_scale = tl.load(scale_ptrsv, mask=valid[None, :],
                              other=0).to(tl.float32, bitcast=True)
            ve = (v_lo * v_scale).to(tl.float16)
            vo = (v_hi * v_scale).to(tl.float16)
            acc_e += tl.sum(weight[None, :] * ve, axis=1)
            acc_o += tl.sum(weight[None, :] * vo, axis=1)
        else:
            kv_offset_v = (
                physical_block[None, :] * v_block_stride
                + offs_in_block[None, :] * (num_kv_heads * head_dim)
                + kv_head_idx * head_dim
                + offs_d[:, None]
            )
            v = tl.load(v_cache_ptr + kv_offset_v, mask=valid[None, :], other=0.0)
            v = v.to(tl.float32)
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
        l_i = l_i + tl.sum(weight)

        m_i = m_new

    if KV_INT4:
        output = tl.reshape(tl.join(acc_e / l_i, acc_o / l_i), (head_dim,))
    else:
        output = acc / l_i
    output_offset = qo * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output.to(output_ptr.dtype.element_ty))


def paged_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    scale: float,
    kv_int4: bool = False,
) -> None:
    """Decode attention over the paged KV cache.

    Args:
        output: (num_tokens, num_heads, head_dim), written in place.
            Only rows belonging to sequences with a single query token are
            supported by the kernel (query_len == 1 per sequence).
        query: (num_tokens, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_table: (num_seqs, max_num_blocks_per_seq) int32
        seq_lens: (num_seqs,) context length per sequence (includes the
            token(s) of the current step; the KV cache update runs before
            attention).
        query_start_loc: (num_seqs + 1,) query row offset per sequence.
        scale: softmax scale (1 / sqrt(head_dim))
    """
    num_seqs = seq_lens.shape[0]
    max_num_blocks_per_seq = block_table.shape[1]
    if not query.is_contiguous():
        query = query.contiguous()

    # Tuned on V100 (f1-autotune, 2026-09-02): BN=128 wins for head_dim<=128
    # at every batch/seq-len combo probed (up to -30% at 4k contexts).
    BLOCK_N = 128 if head_dim <= 128 else 32
    grid = (num_seqs, num_heads)
    _paged_decode_attention_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        query_start_loc,
        scale * _SM_SCALE_LOG2E_CONSTANT,
        k_cache.stride(0),
        v_cache.stride(0),
        KV_INT4=kv_int4,
        PACKED_HS=head_dim // 2 + 4,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _varlen_paged_prefill_attention_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    block_tables_ptr,
    scale_log2,
    k_block_stride,
    v_block_stride,
    KV_INT4: tl.constexpr,
    PACKED_HS: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Varlen flash prefill over the paged KV cache.

    Grid layout: (cdiv(max_query_len, BLOCK_M), num_heads, num_seqs).

    All KV (cached prefix + new tokens) is gathered through the block
    table: the KV cache update runs before attention, so every key/value
    needed by a query row is already resident in the cache. This makes
    the kernel uniform across first prefill, chunked prefill, and prefix
    caching. Causality uses global token positions:

        q_pos = context_len + local_query_offset
        kv_pos = kv offset inside [0, seq_len)
        attend iff kv_pos <= q_pos
    """
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # GQA: map query head onto its KV head.
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    query_len = seq_end - seq_start
    context_len = tl.load(seq_lens_ptr + seq_idx) - query_len
    seq_len = context_len + query_len

    if start_m * BLOCK_M >= query_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    mask_m = offs_m < query_len
    # Query: (num_tokens, num_heads, head_dim)
    q_ptrs = (
        query_ptr
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    offs_half = tl.arange(0, head_dim // 2)
    if KV_INT4:
        qe, qo_ = tl.split(tl.reshape(q, (BLOCK_M, head_dim // 2, 2)))
        acc_e = tl.zeros([BLOCK_M, head_dim // 2], dtype=tl.float32)
        acc_o = tl.zeros([BLOCK_M, head_dim // 2], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    num_kv_blocks = tl.cdiv(seq_len, BLOCK_N)
    for block_n in range(num_kv_blocks):
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        logical_block = offs_n // block_size
        offs_in_block = offs_n % block_size
        valid = mask_n & (logical_block < max_num_blocks_per_seq)

        physical_block = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks_per_seq + logical_block,
            mask=valid,
            other=-1,
        )
        valid = valid & (physical_block != -1)
        physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

        if KV_INT4:
            slot_base = (
                offs_in_block[:, None] * (num_kv_heads * PACKED_HS)
                + kv_head_idx * PACKED_HS
            )
            byte_off_t = (
                physical_block[None, :] * k_block_stride
                + slot_base.T
                + offs_half[:, None]
            )
            raw_k = tl.load(k_cache_ptr + byte_off_t, mask=valid[None, :],
                            other=0)
            k_lo = (raw_k & 0xF).to(tl.float32) - 8.0
            k_hi = ((raw_k >> 4) & 0xF).to(tl.float32) - 8.0
            scale_ptrk = tl.cast(
                k_cache_ptr + physical_block[None, :] * k_block_stride
                + slot_base.T + head_dim // 2,
                tl.pointer_type(tl.uint32),
            )
            k_scale = tl.load(scale_ptrk, mask=valid[None, :],
                              other=0).to(tl.float32, bitcast=True)
            ke = (k_lo * k_scale).to(tl.float16)
            ko = (k_hi * k_scale).to(tl.float16)
            byte_off_n = (
                physical_block[:, None] * v_block_stride
                + slot_base
                + offs_half[None, :]
            )
            raw_v = tl.load(v_cache_ptr + byte_off_n, mask=valid[:, None],
                            other=0)
            v_lo = (raw_v & 0xF).to(tl.float32) - 8.0
            v_hi = ((raw_v >> 4) & 0xF).to(tl.float32) - 8.0
            scale_ptrv = tl.cast(
                v_cache_ptr + physical_block[:, None] * v_block_stride
                + slot_base + head_dim // 2,
                tl.pointer_type(tl.uint32),
            )
            v_scale = tl.load(scale_ptrv, mask=valid[:, None],
                              other=0).to(tl.float32, bitcast=True)
            ve = (v_lo * v_scale).to(tl.float16)
            vo = (v_hi * v_scale).to(tl.float16)
        else:
            # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
            # K^T tile: (head_dim, BLOCK_N)
            kv_offset_t = (
                physical_block[None, :] * k_block_stride
                + offs_in_block[None, :] * (num_kv_heads * head_dim)
                + kv_head_idx * head_dim
                + offs_d[:, None]
            )
            # V tile: (BLOCK_N, head_dim)
            kv_offset_n = (
                physical_block[:, None] * v_block_stride
                + offs_in_block[:, None] * (num_kv_heads * head_dim)
                + kv_head_idx * head_dim
                + offs_d[None, :]
            )

        if KV_INT4:
            qk = (tl.dot(qe, ke) + tl.dot(qo_, ko)).to(tl.float32) * scale_log2
        else:
            k = tl.load(k_cache_ptr + kv_offset_t, mask=valid[None, :], other=0.0)
            qk = tl.dot(q, k).to(tl.float32) * scale_log2

        # Causal mask on global positions; masked lanes -> -inf.
        q_pos = context_len + offs_m[:, None]
        causal = (offs_n[None, :] <= q_pos) & valid[None, :]
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        acc = acc * alpha[:, None]

        # Zero-out padded lanes so -inf -> exp2(-inf) == 0 contributes nothing.
        p = tl.where(causal, p, 0.0)
        if KV_INT4:
            p16 = p.to(tl.float16)
            pe = tl.dot(p16, ve)
            po = tl.dot(p16, vo)
            acc_e = acc_e * alpha[:, None] + pe
            acc_o = acc_o * alpha[:, None] + po
        else:
            v = tl.load(v_cache_ptr + kv_offset_n, mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    if KV_INT4:
        acc = tl.reshape(
            tl.join(acc_e / l_i[:, None], acc_o / l_i[:, None]),
            (BLOCK_M, head_dim),
        )
    else:
        acc = acc / l_i[:, None]
    o_ptrs = (
        output_ptr
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask_m[:, None])


def varlen_paged_prefill_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_query_len: int,
    scale: float,
    kv_int4: bool = False,
) -> None:
    """Varlen prefill attention over the paged KV cache.

    Args:
        output: (num_tokens, num_heads, head_dim), written in place.
        query: (num_tokens, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        cu_seqlens_q: (num_seqs + 1,) cumulative query lengths
            (CommonAttentionMetadata.query_start_loc).
        seq_lens: (num_seqs,) total context length per sequence including
            the current query chunk (KV update runs before attention).
        block_table: (num_seqs, max_num_blocks_per_seq) int32
        max_query_len: longest query in the batch (grid sizing only).
        scale: softmax scale (1 / sqrt(head_dim))
    """
    num_seqs = cu_seqlens_q.shape[0] - 1
    max_num_blocks_per_seq = block_table.shape[1]
    if not query.is_contiguous():
        query = query.contiguous()

    # Tuned on V100 (f1-autotune, 2026-09-02): large M tiles spill the
    # 64KB of shared memory and collapse (BM=64 is 18x slower than BM=16
    # at head_dim 128). BM=16/BN=64 is the fastest pairing probed.
    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 16, 64
    elif head_dim <= 128:
        BLOCK_M, BLOCK_N = 16, 64
    else:
        BLOCK_M, BLOCK_N = 16, 32

    grid = (triton.cdiv(max_query_len, BLOCK_M), num_heads, num_seqs)
    _varlen_paged_prefill_attention_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        seq_lens,
        block_table,
        scale * _SM_SCALE_LOG2E_CONSTANT,
        k_cache.stride(0),
        v_cache.stride(0),
        KV_INT4=kv_int4,
        PACKED_HS=head_dim // 2 + 4,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

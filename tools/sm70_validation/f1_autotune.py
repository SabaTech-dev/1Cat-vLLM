"""Sonda de autotuning decode/prefill de TRITON_PAGED en GPU dedicada.
Formas de Qwen3-0.6B: H=16, KVH=8, D=128, block_size 16."""

import torch, triton, json

import sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
from vllm.v1.attention.ops.triton_paged_attn import (
    _paged_decode_attention_kernel,
    _varlen_paged_prefill_attention_kernel,
)

torch.manual_seed(0)
dev = "cuda:0"
H, KVH, D, BS = 16, 8, 128, 16
SCALE_LOG2 = (D**-0.5) * 1.4426950408889634


def time_fn(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


res = {"decode": [], "prefill": []}

# ---------------- decode sweep ----------------
for batch in (1, 8, 32):
    for seqlen in (1024, 4096):
        nblocks = batch * (seqlen // BS + 2) + 8
        k_cache = (
            torch.randn(nblocks, BS, KVH, D, dtype=torch.float16, device=dev) * 0.3
        )
        v_cache = (
            torch.randn(nblocks, BS, KVH, D, dtype=torch.float16, device=dev) * 0.3
        )
        pool = torch.stack(
            [
                k_cache.reshape(nblocks, BS, KVH, D),
                v_cache.reshape(nblocks, BS, KVH, D),
            ],
            dim=1,
        )
        kc, vc = pool.unbind(1)  # vistas no contiguas como el engine
        bt_rows = seqlen // BS + 1
        block_table = (
            torch.arange(batch * bt_rows, dtype=torch.int32, device=dev).reshape(
                batch, bt_rows
            )
            % nblocks
        )
        seq_lens = torch.full((batch,), seqlen, dtype=torch.int32, device=dev)
        q = torch.randn(batch, H, D, dtype=torch.float16, device=dev) * 0.3
        out = torch.zeros(batch, H, D, dtype=torch.float16, device=dev)
        qsl = torch.arange(batch + 1, dtype=torch.int32, device=dev)
        for bn in (32, 64, 128, 256):

            def run():
                grid = (batch, H)
                _paged_decode_attention_kernel[grid](
                    out,
                    q,
                    kc,
                    vc,
                    block_table,
                    seq_lens,
                    qsl,
                    SCALE_LOG2,
                    kc.stride(0),
                    vc.stride(0),
                    num_heads=H,
                    num_kv_heads=KVH,
                    head_dim=D,
                    block_size=BS,
                    max_num_blocks_per_seq=bt_rows,
                    BLOCK_N=bn,
                )

            ms = time_fn(run)
            res["decode"].append(
                {"batch": batch, "seqlen": seqlen, "BLOCK_N": bn, "ms": round(ms, 3)}
            )
            print(f"decode b{batch} s{seqlen} BN={bn}: {ms:.3f} ms")

# ---------------- prefill sweep ----------------
for qlen in (512, 2048):
    nblocks = (qlen // BS + 2) + 8
    k_cache = torch.randn(nblocks, BS, KVH, D, dtype=torch.float16, device=dev) * 0.3
    v_cache = torch.randn(nblocks, BS, KVH, D, dtype=torch.float16, device=dev) * 0.3
    pool = torch.stack(
        [k_cache.reshape(nblocks, BS, KVH, D), v_cache.reshape(nblocks, BS, KVH, D)],
        dim=1,
    )
    kc, vc = pool.unbind(1)
    bt_rows = qlen // BS + 1
    block_table = (
        torch.arange(bt_rows, dtype=torch.int32, device=dev).reshape(1, bt_rows)
        % nblocks
    )
    seq_lens = torch.full((1,), qlen, dtype=torch.int32, device=dev)
    q = torch.randn(qlen, H, D, dtype=torch.float16, device=dev) * 0.3
    out = torch.zeros(qlen, H, D, dtype=torch.float16, device=dev)
    cu = torch.tensor([0, qlen], dtype=torch.int32, device=dev)
    for bm, bn in ((16, 64), (32, 64), (64, 64), (32, 128), (64, 128), (128, 64)):

        def run():
            grid = (triton.cdiv(qlen, bm), H, 1)
            _varlen_paged_prefill_attention_kernel[grid](
                out,
                q,
                kc,
                vc,
                cu,
                seq_lens,
                block_table,
                SCALE_LOG2,
                kc.stride(0),
                vc.stride(0),
                num_heads=H,
                num_kv_heads=KVH,
                head_dim=D,
                block_size=BS,
                max_num_blocks_per_seq=bt_rows,
                BLOCK_M=bm,
                BLOCK_N=bn,
            )

        ms = time_fn(run)
        res["prefill"].append({"qlen": qlen, "BM": bm, "BN": bn, "ms": round(ms, 3)})
        print(f"prefill q{qlen} BM={bm} BN={bn}: {ms:.3f} ms")

json.dump(res, open("/tmp/opencode/f1-autotune.json", "w"))
print("AUTOTUNE DONE")

"""Verificacion propia del claim #441/#469: tiles narrow (16/1/4) vs GB300
(64/1/2) en QSA prefill, geometria H12/KV1/D256 TOPK2048, chunk 2048.
Peuqui claima 19.3x en V100. Medimos nosotros."""

import sys, time, torch

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
import vllm.models.qwen4_exp.nvidia.ops.qsa as qsa_ops

torch.manual_seed(0)
dev = "cuda:0"
H, KVH, D, TOPK = 12, 1, 256, 2048
NTOK = 2048
PAGE = 16
NPAGES = (NTOK * 2) // PAGE + 8

q = torch.randn(NTOK, H, D, dtype=torch.float16, device=dev) * 0.3
k_cache = torch.randn(NPAGES, PAGE, KVH, D, dtype=torch.float16, device=dev) * 0.3
v_cache = torch.randn(NPAGES, PAGE, KVH, D, dtype=torch.float16, device=dev) * 0.3
# indices logicos validos: para el token t, TOPK indices dentro de [0, t+PAGE)
rows = torch.arange(NTOK, device=dev).unsqueeze(1)
lims = torch.clamp(rows + PAGE, max=NTOK)
base = torch.arange(TOPK, device=dev).unsqueeze(0) % lims.clamp(min=1)
logical_indices = torch.minimum(base, lims - 1).to(torch.int32)
block_table = (logical_indices // PAGE).to(torch.int32)
token_to_req = torch.zeros(NTOK, dtype=torch.int32, device=dev)
out = torch.empty(NTOK, H, D, dtype=torch.float16, device=dev)


def time_run(n=20, warm=5):
    for _ in range(warm):
        qsa_ops.qsa_sparse_paged_attention(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
        )
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        qsa_ops.qsa_sparse_paged_attention(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
        )
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


orig = qsa_ops._qsa_sparse_launch_profile
# narrow (post-#469, default en pre-Ampere)
t_narrow = time_run()
# GB300 forzado (el perfil pre-#469 para prefill: 64/1/2)
qsa_ops._qsa_sparse_launch_profile = lambda bp, bm, pa: (64, 1, 2)
t_gb300 = time_run()
qsa_ops._qsa_sparse_launch_profile = orig

ratio = t_gb300 / t_narrow
print(
    f"narrow(16/1/4): {t_narrow:.2f} ms | GB300(64/1/2): {t_gb300:.2f} ms "
    f"| ratio GB300/narrow: {ratio:.2f}x"
)
print(
    f"VEREDICTO: {'CLAIM CONFIRMADO (narrow gana)' if ratio > 1.2 else 'CLAIM NO REPRODUCIDO'} "
    f"(claim #441: 19.3x)"
)

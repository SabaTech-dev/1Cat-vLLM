"""Repro aislado: KKT fwd con config UNICA forzada (sin benchmarking de
autotune). Si BK=64 cuelga aqui tambien, el bug es del kernel; si pasa,
el bug es la interaccion autotune x init-profiling.

Uso: python f9_kkt_probe.py <BK>
"""

import sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")

import torch  # noqa: E402


def main() -> None:
    bk = int(sys.argv[1])
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )

    torch.manual_seed(0)
    dev = "cuda:0"
    B, T, H, K = 1, 512, 8, 128  # forma tipica de GDN (head_k_dim=128)
    k = torch.randn(B, T, H, K, device=dev, dtype=torch.float16)
    beta = torch.rand(B, T, H, device=dev, dtype=torch.float32)

    # parchear la lista de configs del autotune a UNA sola (BK forzado)
    import vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt as m

    single = [m.triton.Config({"BK": bk}, num_warps=4, num_stages=2)]
    m.chunk_scaled_dot_kkt_fwd_kernel.configs = single

    out = chunk_scaled_dot_kkt_fwd(k, beta=beta, chunk_size=64)
    torch.cuda.synchronize()
    print(
        f"KKT probe BK={bk}: OK, out {tuple(out.shape)} "
        f"mean={out.float().mean().item():.4f}"
    )


if __name__ == "__main__":
    main()

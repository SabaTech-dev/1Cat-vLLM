# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained Triton paged-attention backend (TRITON_PAGED) for
Volta SM70.

Sprint F1 skeleton. The backend ports the nano-vllm / MinivLLM paged
attention design into the vLLM V1 attention backend interface:

* KV cache store:      Triton scatter kernel (ops.triton_paged_attn)
* Decode attention:    Triton paged kernel with block tables, online
                       softmax in fp32, one query token per sequence
* Prefill attention:   Triton varlen flash kernel; all KV (cached prefix
                       + new tokens) is gathered through the block table,
                       so chunked prefill and prefix caching follow the
                       same uniform code path

No flash_attn / FlashInfer dependency: the whole backend is Triton +
PyTorch, which makes it usable on GPUs without vendor attention kernels
(V100 / SM70 in particular).

References:
    - https://github.com/GeeeekExplorer/nano-vllm
    - https://github.com/Wenyueh/MinivLLM
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import functools
import os

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_num_attention_heads_from_layers
from vllm.v1.kv_cache_interface import kv_cache_uses_per_token_head_scales
from vllm.v1.attention.ops.triton_paged_attn import (
    paged_decode_attention,
    store_kvcache_paged,
    varlen_paged_prefill_attention,
)


@functools.lru_cache(maxsize=1)
def _int8_clip_frac() -> float:
    """Fraction of channels per row allowed to saturate in int8 K/V
    quantization (env VLLM_SM70_INT8_CLIP, default 0.01; 0 disables
    clipping and restores the pure amax scale)."""
    raw = os.getenv("VLLM_SM70_INT8_CLIP")
    if raw is None:
        return 0.01
    try:
        return max(0.0, min(float(raw), 0.5))
    except ValueError:
        return 0.01


def _row_clip_threshold(xf: torch.Tensor, clip_frac: float) -> torch.Tensor:
    """Per-row magnitude threshold allowing the top clip_frac channels to
    saturate. Returns a (…, 1) tensor broadcastable over the last dim.
    clip_frac <= 0 falls back to the row amax (no clipping).
    """
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if clip_frac <= 0:
        return amax
    d = xf.shape[-1]
    k = max(2, int(d * clip_frac))
    # (k+1)-th largest magnitude becomes the scale reference; the k
    # largest channels saturate at +-127.
    kth = torch.topk(xf.abs(), k + 1, dim=-1).values[..., -1:].float()
    # Never let clipping shrink the scale below 5% of amax: protects rows
    # whose mass is a single dominant channel.
    return torch.maximum(kth, amax * 0.05)


def _quantize_int8_pth(x: torch.Tensor, clip_frac: float | None = None) -> torch.Tensor:
    """Per-(token, head) symmetric int8 quantization (K side).

    x: (num_tokens, num_kv_heads, head_dim) -> int8
    (num_tokens, num_kv_heads, head_dim + 4): full-resolution bytes plus a
    trailing fp32 scale. The int8 range (127) tolerates the post-RoPE
    key outliers that break symmetric int4.

    clip_frac (env VLLM_SM70_INT8_CLIP, default 0.01) lets the largest
    channels of each row saturate instead of stretching the scale to the
    row amax. Fine-tuned checkpoints can carry single-channel outliers
    100x the row median; an amax scale then leaves ~1% resolution for the
    mass channels and collapses perplexity.
    """
    if clip_frac is None:
        clip_frac = _int8_clip_frac()
    xf = x.float()
    scale = (
        (_row_clip_threshold(xf, clip_frac) / 127.0).clamp_(min=1e-8).to(torch.float32)
    )
    q = torch.clamp(torch.round(xf / scale), -127, 127).to(torch.int8)
    scale_bytes = scale.contiguous().view(torch.uint8)
    return torch.cat([q.view(torch.uint8), scale_bytes], dim=-1)


def _quantize_int4_pth(x: torch.Tensor) -> torch.Tensor:
    """Per-(token, head) symmetric int4 quantization (V side).

    x: (num_tokens, num_kv_heads, head_dim) -> uint8
    (num_tokens, num_kv_heads, head_dim//2 + 4): two nibbles per byte
    (low nibble = even lane) plus a trailing fp32 scale such that
    data = round(x / scale) + 8 fits [0, 15]. Values (no RoPE) have
    tight distributions and survive int4.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / 7.0).clamp_(min=1e-8).to(torch.float32)
    q = torch.clamp(torch.round(x.float() / scale), -7, 7).to(torch.uint8) + 8
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    scale_bytes = scale.contiguous().view(torch.uint8)
    return torch.cat([packed, scale_bytes], dim=-1)


from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


@dataclass
class TritonPagedMetadata:
    """Per-batch metadata for the TRITON_PAGED backend.

    All tensors come straight from ``CommonAttentionMetadata``; the
    backend keeps a plain pass-through dataclass so the kernels receive
    exactly what the scheduler produced.

    NOTE(sang, ported): definition of context_len / query_len / seq_len.

    |---------- N-1 iteration --------|
    |---------------- N iteration ---------------------|
    |- tokenA -|......................|-- newTokens ---|
    |---------- context_len ----------|
    |-------------------- seq_len ---------------------|
                                   |-- query_len ---|
    """

    num_actual_tokens: int
    """Number of tokens excluding padding."""

    max_query_len: int
    query_start_loc: torch.Tensor
    """(num_seqs + 1,), dtype int32"""

    max_seq_len: int
    seq_lens: torch.Tensor
    """(num_seqs,), total context length including the current query."""

    block_table: torch.Tensor
    """(num_seqs, max_num_blocks_per_seq), dtype int32"""

    slot_mapping: torch.Tensor
    """(num_actual_tokens,) flat slot indices for the KV cache store."""


class TritonPagedMetadataBuilder(AttentionMetadataBuilder[TritonPagedMetadata]):
    # Graph-safe for pure decode batches (shapes derive from the runner's
    # persistent padded buffers, no host syncs, store skips slot == -1).
    # Re-evaluated 2026-09-03 at util 0.85: greedy outputs are identical to
    # eager across piecewise and decode-only capture, decode b1 goes from
    # 22.9 to 316 tok/s (13.8x). The earlier "corrupted first request" was
    # a misdiagnosis: Qwen3-0.6B is a BASE model and that is its legitimate
    # raw-completion output (chat-templated serving says "Paris"; raw
    # completion says otherwise). Scope: decode-only is the measured-fastest
    # and most conservative mode.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> TritonPagedMetadata:
        # This backend does not implement cascade attention; a non-zero
        # common prefix is simply ignored (each sequence attends to its
        # own pages through the block table).
        return TritonPagedMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: "CommonAttentionMetadata"
    ) -> TritonPagedMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Capture speed: real seq_lens make the decode kernel walk every KV
        # block of max_model_len for every capture size. Content is refreshed
        # from the runner's persistent buffers before each replay.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata


class TritonPagedBackend(AttentionBackend):
    supported_dtypes: list[torch.dtype] = [
        # Volta (SM70) has no BF16 support.
        torch.float16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: list[CacheDType] = [
        "auto",
        "float16",
        "int8_per_token_head",
        "int8k_int4v_per_token_head",
    ]

    # The KV cache update runs through do_kv_cache_update (the fork's
    # convention for every V1 backend), not inside forward().
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_PAGED"

    @staticmethod
    def get_impl_cls() -> type["TritonPagedImpl"]:
        return TritonPagedImpl

    @staticmethod
    def get_builder_cls() -> type["TritonPagedMetadataBuilder"]:
        return TritonPagedMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Same allocation shape as FlashAttention / TRITON_ATTN backends:
        # (num_blocks, 2, block_size, num_kv_heads, head_size) with the
        # K and V halves split via unbind(1). The kernels in
        # ops.triton_paged_attn then operate on the
        # (num_blocks, block_size, num_kv_heads, head_size) views.
        # Per-token-head modes carry the fp32 scale inline in the last 4
        # bytes of each head slot (mirrors triton_attn), so pad the head
        # to keep the allocation and this view on the same slot width.
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            from vllm.utils.torch_utils import (
                STR_DTYPE_TO_TORCH_DTYPE,
                get_dtype_size,
            )

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            head_size += get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # The kernels index the NHD layout directly.
        if include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def get_required_kv_cache_layout(cls) -> str | None:
        return "NHD"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # tl.arange requires a power of 2; keep the range generous.
        return (
            head_size >= 16 and head_size <= 256 and (head_size & (head_size - 1)) == 0
        )

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Sprint F1 scope: Volta only. The kernels contain nothing
        # SM70-specific, so lifting this gate (e.g. to also cover Turing)
        # is trivial once validated.
        return capability.major == 7 and capability.minor == 0

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class TritonPagedImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype == "int8k_int4v_per_token_head":
            # The K and V packed slots need different widths (D+4 vs
            # D//2+4) while the engine carves a uniform slot per side;
            # store-side input/cache width separation is pending.
            raise NotImplementedError(
                "int8k_int4v_per_token_head is not wired yet; use "
                "int8_per_token_head (2x context) on TRITON_PAGED"
            )
        self.kv_quant = kv_cache_dtype == "int8_per_token_head"
        self.kv_int8_both = kv_cache_dtype == "int8_per_token_head"
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if alibi_slopes is not None:
            raise NotImplementedError("TRITON_PAGED does not support alibi slopes yet.")
        if sliding_window is not None:
            raise NotImplementedError(
                "TRITON_PAGED does not support sliding windows yet."
            )
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise NotImplementedError(
                "TRITON_PAGED does not support logits soft capping yet."
            )
        if sinks is not None:
            raise NotImplementedError("TRITON_PAGED does not support sinks.")
        if use_alibi_sqrt:
            raise NotImplementedError(
                "TRITON_PAGED does not support alibi sqrt attention."
            )
        if chunk_lookback != -1:
            raise NotImplementedError("TRITON_PAGED does not support chunk lookback.")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonPagedMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention forward.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: metadata for attention.
        Returns:
            shape = [num_tokens, num_heads, head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported by TRITON_PAGED"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert kv_cache is not None and kv_cache.numel() > 0, (
            "TRITON_PAGED requires an allocated KV cache (decoder attention)."
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(1)

        if attn_metadata.max_query_len == 1:
            # Decode: one query token per sequence. The KV cache update has
            # already run (do_kv_cache_update), so seq_lens includes the
            # tokens of this step.
            paged_decode_attention(
                output=output[:num_actual_tokens],
                query=query[:num_actual_tokens],
                k_cache=key_cache,
                v_cache=value_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                query_start_loc=attn_metadata.query_start_loc,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                block_size=kv_cache.shape[2],
                scale=self.scale,
                k_int8=self.kv_quant,
                v_int8=self.kv_int8_both,
                v_int4=self.kv_quant and not self.kv_int8_both,
            )
        else:
            # Prefill / extend (also used for multi-token decode steps,
            # e.g. speculative decoding: the kernel is position-based so
            # any query length works). All KV — cached prefix and new
            # tokens — is gathered through the block table.
            varlen_paged_prefill_attention(
                output=output[:num_actual_tokens],
                query=query[:num_actual_tokens],
                k_cache=key_cache,
                v_cache=value_cache,
                cu_seqlens_q=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                block_size=kv_cache.shape[2],
                max_query_len=attn_metadata.max_query_len,
                scale=self.scale,
                k_int8=self.kv_quant,
                v_int8=self.kv_int8_both,
                v_int4=self.kv_quant and not self.kv_int8_both,
            )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store the projected keys/values into the paged KV cache."""
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            # Encoder attention uses direct K/V tensors without caching.
            return
        if kv_cache.numel() == 0:
            return
        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_quant:
            # Quantized slots with in-tail fp32 scales: K = int8 rows
            # (head_dim+4); V = int4 nibble rows (head_dim//2+4) for the
            # hybrid dtype, or int8 rows for int8_per_token_head. The
            # store kernel skips slot == -1 natively (capture-safe).
            key = _quantize_int8_pth(key)
            if self.kv_int8_both:
                value = _quantize_int8_pth(value)
            else:
                value = _quantize_int4_pth(value)
        store_kvcache_paged(
            key=key,
            value=value,
            k_cache=key_cache,
            v_cache=value_cache,
            slot_mapping=slot_mapping,
            block_size=kv_cache.shape[2],
        )

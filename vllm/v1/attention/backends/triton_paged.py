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
from vllm.v1.attention.ops.triton_paged_attn import (
    paged_decode_attention,
    store_kvcache_paged,
    varlen_paged_prefill_attention,
)
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
    # CUDA graphs: attempted 2026-09-02 and reverted. Both piecewise and
    # decode-only capture deterministically corrupted the first replayed
    # request on V100 (Qwen3-0.6B, "Francia" prompt), while later requests
    # stayed coherent. Root cause not yet isolated (capture-time kernel
    # constexprs vs serving-time buffer layout are the prime suspects).
    # The backend is fully functional in eager mode; keep graphs off until
    # the capture path is instrumented properly.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

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
        # TODO(f1): fp8 KV cache (dequant in kernel), fp8 per-token-head.
        "auto",
        "float16",
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
        store_kvcache_paged(
            key=key,
            value=value,
            k_cache=key_cache,
            v_cache=value_cache,
            slot_mapping=slot_mapping,
            block_size=kv_cache.shape[2],
        )

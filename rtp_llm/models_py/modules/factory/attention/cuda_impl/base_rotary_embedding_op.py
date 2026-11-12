import os
from pathlib import Path
import rtp_llm
_RQ_LEGACY = os.environ.get("RQ_LEGACY_NUMERICS") == "1"
"""Base class for Rotary Positional Embedding operations.

This module provides the abstract base class for implementing RoPE (Rotary Positional Embedding)
operations with FlashInfer. It includes common functionality for both MHA and MLA implementations.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import flashinfer
import flashinfer.rope as rope
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.ops import RopeConfig, get_rope_cache_once


class BaseRotaryEmbeddingOp(ABC):
    """Base class for rotary positional embedding with FlashInfer.

    This class provides common functionality for both MHA and MLA implementations,
    including RoPE application and warmup cache preparation.
    """

    def __init__(
        self,
        head_size: int,
        cos_sin_cache: torch.Tensor | None,
        token_per_block: int,
        is_neox_style: bool,
        rope_config: Optional[RopeConfig] = None,
        max_position_embeddings: int = 32768,
    ) -> None:
        """
        Args:
            head_size: Dimension of each attention head
            cos_sin_cache: Precomputed cos/sin cache for RoPE [max_seq_len, rope_dim].
                          If None and rope_config is provided, will auto-generate using get_rope_cache_once.
            token_per_block: Number of tokens per KV cache block (page size)
            is_neox_style: RoPE interleave style:
                          - True (GPT-NeoX/interleave): Rotate adjacent pairs of dimensions together,
                            i.e., (x[0], x[1]), (x[2], x[3]), ..., (x[d-2], x[d-1])
                          - False (LLaMA/non-interleave): Rotate first and second halves separately,
                            i.e., (x[0], x[d/2]), (x[1], x[d/2+1]), ..., (x[d/2-1], x[d-1])
                          Most modern models (LLaMA, Qwen, DeepSeek, Mistral, etc.) use False.
                          Only specific models like GPT-NeoX use True.
            rope_config: RoPE configuration for auto-generating cos_sin_cache if not provided (optional)
            max_position_embeddings: Maximum position embeddings for auto-generating cache
        """
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.token_per_block = token_per_block
        self.rope_config = rope_config

        # Try to get cos_sin_cache from C++ RopeCache if not provided
        if cos_sin_cache is None and rope_config is not None:
            # Save original interleave value

            try:
                # FlashInfer uses non-interleaved format (False)
                rope_cache = get_rope_cache_once(
                    rope_config, max_position_embeddings, is_cuda=True, interleave=False
                )
                self.cos_sin_cache = rope_cache.data
                if _RQ_LEGACY and float(rope_config.base) == 1000000.0 and int(rope_config.dim) == 128 and float(rope_config.scale) == 1.0:
                    inv_freq = 1.0 / torch.pow(int(rope_config.base), torch.arange(0, 128, 2, dtype=torch.int64).float() / 128)
                    positions = torch.arange(max_position_embeddings, dtype=torch.int64).float()
                    freqs = torch.outer(positions, inv_freq)
                    self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=1).contiguous().cuda()
            except Exception:
                # If get_rope_cache_once fails, fallback to dynamic computation in _apply_rope
                self.cos_sin_cache = None
        else:
            self.cos_sin_cache = cos_sin_cache

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope_params: Any,
    ) -> None:
        """Apply RoPE to query and key tensors in-place.

        Args:
            query: Query tensor to apply RoPE to
            key: Key tensor to apply RoPE to
            rope_params: Parameters containing position IDs
        """
        # narrow() to query.size(0) — no-op when fillParams already shrank
        # positions_d to nnz, correct slice when fill_params_mha_device left
        # the buffer at its alloc-time size.
        nnz = query.size(0)
        pos_ids = rope_params.positions_d.narrow(0, 0, nnz)

        if _RQ_LEGACY and query.dtype == torch.float16 and query.shape[-1] == 128 and self.cos_sin_cache is not None:
            import ctypes
            if not hasattr(self, '_rq_cache_fn'):
                self._rq_cache_lib = ctypes.CDLL(str(Path(rtp_llm.__file__).parent / 'rq_compat/lib/rope-legacy.so'))
                self._rq_cache_fn = self._rq_cache_lib.rq_cache_rope
                self._rq_cache_fn.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
                self._rq_cache_fn.restype = ctypes.c_int
            positions = pos_ids.int().contiguous()
            for tensor in (query, key):
                src = tensor.contiguous()
                dst = torch.empty_like(src)
                err = self._rq_cache_fn(src.data_ptr(), dst.data_ptr(), positions.data_ptr(), self.cos_sin_cache.data_ptr(), src.shape[0], src.shape[1], torch.cuda.current_stream().cuda_stream)
                if err: raise RuntimeError('cache rope diagnostic failed')
                tensor.copy_(dst)
            return
        if self.cos_sin_cache is not None:
            rope._apply_rope_pos_ids_cos_sin_cache(  # type: ignore
                q=query,
                k=key,
                q_rope=query,
                k_rope=key,
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=pos_ids,
                interleave=self.is_neox_style,
            )
        else:
            rope_theta = (
                self.rope_config.base if self.rope_config is not None else 10000
            )
            flashinfer.apply_rope_pos_ids_inplace(
                query, key, pos_ids, rope_theta=rope_theta
            )

    def _prepare_warmup_cache_indices(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """Prepare indices and metadata for warmup KV cache creation.

        This creates dummy batch indices, positions, page indices, etc. required
        for JIT compilation warmup when no real KV cache is provided.

        Args:
            num_tokens: Number of tokens to process
            device: Device to create tensors on

        Returns:
            Tuple of (batch_indices, positions, kv_page_indices, kv_page_indptr,
                     kv_last_page_len, max_num_pages)
        """
        kv_len = [num_tokens]
        num_pages_per_req = torch.tensor(
            [math.ceil(length / self.token_per_block) for length in kv_len],
            dtype=torch.int32,
            device=device,
        )
        kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device=device)
        kv_append_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(kv_append_length, dim=0),
            ]
        )

        max_num_pages = int(sum(num_pages_per_req))
        kv_page_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(num_pages_per_req, dim=0),
            ]
        )
        kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=device)

        kv_last_page_len = torch.tensor(
            [
                (
                    length % self.token_per_block
                    if length % self.token_per_block != 0
                    else self.token_per_block
                )
                for length in kv_len
            ],
            dtype=torch.int32,
            device=device,
        )

        batch_indices, positions = get_batch_indices_positions(
            kv_append_indptr,
            get_seq_lens(kv_page_indptr, kv_last_page_len, self.token_per_block),
            num_tokens,
        )

        return (
            batch_indices,
            positions,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            max_num_pages,
        )

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> None:
        """Forward pass - must be implemented by subclasses.

        Subclasses should:
        1. Call self._apply_rope() to apply RoPE to Q and K
        2. Implement their specific KV cache append logic
        3. Use self._prepare_warmup_cache_indices() for warmup cache preparation
        """
        pass

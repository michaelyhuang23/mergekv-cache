from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from einops import rearrange, einsum



class KNormCache(Cache):
    """
    An implementation of KNorm filtering in transformers' KV cache framework.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the short context window.
        max_length (`int`):
            The maximum cache length.
    """

    is_sliding = False

    def __init__(self, compression_ratio:float, window_length: int) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.compression_ratio = compression_ratio
        self.window_length = window_length
        self.actual_len = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        assert False
        return -1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # [bsz, num_heads, seq_len, head_dim]
        #if layer_idx == 0:
        #    print('update called', max([cache.shape[-2] for cache in self.key_cache] + [0]), len(self.key_cache), key_states.shape)
        #print(key_states.shape)
        self.actual_len += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        if key_cache.shape[2] < round(self.compression_ratio * self.actual_len):
            return key_cache, value_cache

        max_length = round(self.compression_ratio * self.actual_len)
        key_length = key_cache.shape[-2]
        
        comp_key, window_key = key_cache[..., :key_length-self.window_length, :], key_cache[..., key_length-self.window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-self.window_length, :], value_cache[..., key_length-self.window_length:,:]

        proj_key = comp_key.norm(dim=-1)
        key_idx = (-proj_key).topk(max(0, max_length - self.window_length), -1).indices.sort().values
        key_idx = key_idx[..., None].repeat(1, 1, 1, key_cache.shape[-1])

        comp_key = torch.gather(comp_key, -2, key_idx)
        comp_value = torch.gather(comp_value, -2, key_idx)

        self.key_cache[layer_idx] = torch.cat((comp_key, window_key), -2)
        self.value_cache[layer_idx] = torch.cat((comp_value, window_value), -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]



class MergeKV(Cache):
    def __init__(self, compression_ratio:int, window_length: int, top_k: int) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.attn_cache: List[torch.Tensor] = []
        self.attn_count: List[torch.Tensor] = []
        self.compression_ratio = compression_ratio
        self.window_length = window_length
        self.top_k = top_k
        self.actual_len = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        return -1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        self.actual_len += key_states.shape[-2]

        assert "query" in cache_kwargs
        query_states = cache_kwargs["query"]
        triangle_attn_scores = torch.einsum("bhid,bhjd->bhij", query_states, key_states) / (query_states.shape[-1] ** 0.5)
        seq_len = query_states.size(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query_states.device)).bool()
        triangle_attn_scores = triangle_attn_scores.masked_fill(~causal_mask, float("-inf"))
        triangle_attn_scores = torch.softmax(triangle_attn_scores, dim=-1)
        triangle_attn_scores = triangle_attn_scores.sum(dim=-2)
        triangle_attn_count = causal_mask.sum(dim=-2)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.attn_cache.append(triangle_attn_scores)
            self.attn_count.append(triangle_attn_count)
        else:
            # Growing cache
            attn_scores = torch.einsum("bhid,bhjd->bhij", query_states, self.key_cache[layer_idx]) / (query_states.shape[-1] ** 0.5)
            attn_scores = torch.softmax(attn_scores, dim=-1)
            self.attn_cache[layer_idx] += attn_scores.sum(dim=-2) # the rectangle
            self.attn_cache[layer_idx] = torch.cat([self.attn_cache[layer_idx], triangle_attn_scores], dim=-1) # pieces the triangle with the rectangle
            self.attn_count[layer_idx] += self.query_states.size(-2)
            self.attn_count[layer_idx] = torch.cat([self.attn_count[layer_idx], triangle_attn_count], dim=-1)

            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)


        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        max_length = self.compression_ratio * self.actual_len
        self.top_k = max(0, round(max_length - self.window_length))

        if key_cache.shape[2] < max_length:
            return key_cache, value_cache

        key_length = key_cache.shape[-2]

        comp_key, window_key = key_cache[..., :key_length-self.window_length, :], key_cache[..., key_length-self.window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-self.window_length, :], value_cache[..., key_length-self.window_length:,:]

        avg_attn_scores = self.attn_cache[layer_idx] / self.attn_count[layer_idx]
        top_k_idx = avg_attn_scores.topk(self.top_k, -1).indices.sort().values
        top_k_idx = top_k_idx[..., None].repeat(1, 1, 1, key_cache.shape[-1])
        top_k_key = torch.gather(comp_key, -2, top_k_idx)
        top_k_value = torch.gather(comp_value, -2, top_k_idx)

        self.key_cache[layer_idx] = torch.cat((top_k_key, window_key), -2)
        self.value_cache[layer_idx] = torch.cat((top_k_value, window_value), -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

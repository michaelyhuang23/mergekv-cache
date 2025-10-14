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

    def __init__(self, window_length: int, compression_ratio:float, tolerance:float = 0.05) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.compression_ratio = compression_ratio
        self.tolerance = tolerance
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
        self.actual_len += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def clip(self, layer_idx, *args, **kwargs):
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        if key_cache.shape[2] < round((self.compression_ratio + self.tolerance) * self.actual_len):
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



class TopKKV(Cache):
    def __init__(self, compression_ratio: float, window_length: int, top_k: int, tolerance: float = 0.05) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.attn_cache: List[torch.Tensor] = []
        self.attn_count: List[torch.Tensor] = []
        self.compression_ratio = compression_ratio
        self.window_length = window_length
        self.tolerance = tolerance
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

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def clip(self, layer_idx: int, attn_weights: torch.Tensor, num_heads: int):
        attn_weights = attn_weights.detach()
        attn_weights = rearrange(attn_weights, "b (h gs) s1 s2 -> b h gs s1 s2", h=num_heads)
        aggr_attn_scores = attn_weights.sum(dim=(-2, -3)) # sum over s1 and gs
        if len(self.attn_cache) <= layer_idx:
            self.attn_cache.append(aggr_attn_scores)
            self.attn_count.append(torch.ones_like(aggr_attn_scores) * attn_weights.shape[-2])
        else:
            cur_len = self.attn_cache[layer_idx].shape[-1]
            self.attn_cache[layer_idx] += aggr_attn_scores[..., :cur_len]
            self.attn_count[layer_idx] += torch.ones_like(aggr_attn_scores[..., :cur_len]) * attn_weights.shape[-2]
            self.attn_cache[layer_idx] = torch.cat([self.attn_cache[layer_idx], aggr_attn_scores[..., cur_len:]], dim=-1)
            aggr_attn_counts = (attn_weights[..., cur_len:] > 1e-6).sum(dim=(-2, -3))
            self.attn_count[layer_idx] = torch.cat([self.attn_count[layer_idx], aggr_attn_counts], dim=-1)

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        attn_cache = self.attn_cache[layer_idx]
        attn_count = self.attn_count[layer_idx]

        max_length = self.compression_ratio * self.actual_len
        self.top_k = max(0, round(max_length - self.window_length))

        if key_cache.shape[2] < self.actual_len * (self.compression_ratio + self.tolerance):
            return key_cache, value_cache

        key_length = key_cache.shape[-2]

        comp_key, window_key = key_cache[..., :key_length-self.window_length, :], key_cache[..., key_length-self.window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-self.window_length, :], value_cache[..., key_length-self.window_length:,:]
        comp_attn_score, window_attn_score = attn_cache[..., :key_length-self.window_length], attn_cache[..., key_length-self.window_length:]
        comp_attn_count, window_attn_count = attn_count[..., :key_length-self.window_length], attn_count[..., key_length-self.window_length:]

        avg_attn_score = comp_attn_score / comp_attn_count
        top_k_idx = avg_attn_score.topk(self.top_k, -1).indices.sort().values
        top_k_idx_kv = rearrange(top_k_idx, 'b h l -> b h l 1')
        top_k_idx_kv = top_k_idx_kv.repeat(1, 1, 1, key_cache.shape[-1])
        top_k_key = torch.gather(comp_key, -2, top_k_idx_kv)
        top_k_value = torch.gather(comp_value, -2, top_k_idx_kv)
        top_k_attn_score = torch.gather(comp_attn_score, -1, top_k_idx)
        top_k_attn_count = torch.gather(comp_attn_count, -1, top_k_idx)

        self.key_cache[layer_idx] = torch.cat((top_k_key, window_key), -2)
        self.value_cache[layer_idx] = torch.cat((top_k_value, window_value), -2)
        self.attn_cache[layer_idx] = torch.cat((top_k_attn_score, window_attn_score), -1)
        self.attn_count[layer_idx] = torch.cat((top_k_attn_count, window_attn_count), -1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


@torch.compile
def gaussian_kernel(a, b, sigma):
    return torch.exp(- (a-b).norm(dim=-1)**2 / (2 * sigma**2))

class MergeKV(Cache):
    def __init__(self, compression_ratio: float, window_ratio: float, top_k_ratio: float, tolerance: float = 0.05) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.attn_cache: List[torch.Tensor] = []
        self.attn_count: List[torch.Tensor] = []
        self.compression_ratio = compression_ratio
        self.window_ratio = window_ratio
        self.top_k_ratio = top_k_ratio
        self.tolerance = tolerance
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
        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def clip(self, layer_idx: int, attn_weights: torch.Tensor, num_heads: int):
        attn_weights = attn_weights.detach()
        attn_weights = rearrange(attn_weights, "b (h gs) s1 s2 -> b h gs s1 s2", h=num_heads)
        aggr_attn_scores = attn_weights.sum(dim=(-2, -3)) # sum over s1 and gs
        if len(self.attn_cache) <= layer_idx:
            self.attn_cache.append(aggr_attn_scores)
            self.attn_count.append(torch.ones_like(aggr_attn_scores) * attn_weights.shape[-2])
        else:
            cur_len = self.attn_cache[layer_idx].shape[-1]
            self.attn_cache[layer_idx] += aggr_attn_scores[..., :cur_len]
            self.attn_count[layer_idx] += torch.ones_like(aggr_attn_scores[..., :cur_len]) * attn_weights.shape[-2]
            self.attn_cache[layer_idx] = torch.cat([self.attn_cache[layer_idx], aggr_attn_scores[..., cur_len:]], dim=-1)
            aggr_attn_counts = (attn_weights[..., cur_len:] > 1e-6).sum(dim=(-2, -3))
            self.attn_count[layer_idx] = torch.cat([self.attn_count[layer_idx], aggr_attn_counts], dim=-1)

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        attn_cache = self.attn_cache[layer_idx]
        attn_count = self.attn_count[layer_idx]

        max_length = round(self.compression_ratio * self.actual_len)
        top_k = round(self.top_k_ratio * self.actual_len)
        window_length = round(self.window_ratio * self.actual_len)

        if key_cache.shape[2] <= self.actual_len * (self.compression_ratio + self.tolerance):
            return key_cache, value_cache

        key_length = key_cache.shape[-2]

        comp_key, window_key = key_cache[..., :key_length-window_length, :], key_cache[..., key_length-window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-window_length, :], value_cache[..., key_length-window_length:,:]
        comp_attn_score, window_attn_score = attn_cache[..., :key_length-window_length], attn_cache[..., key_length-window_length:]
        comp_attn_count, window_attn_count = attn_count[..., :key_length-window_length], attn_count[..., key_length-window_length:]

        avg_attn_score = comp_attn_score / comp_attn_count

        connections = (comp_key[..., :-1, :] * comp_key[..., 1: , :]).mean(dim=-1)
        separators = connections.sort(dim=-1).indices[..., : max_length - top_k - window_length - 2].sort().values
        merged_key = torch.zeros((comp_key.shape[0], comp_key.shape[1], separators.shape[-1]+1, comp_key.shape[-1]), device=comp_key.device, dtype=comp_key.dtype)
        merged_value = torch.zeros((comp_key.shape[0], comp_key.shape[1], separators.shape[-1]+1, comp_key.shape[-1]), device=comp_key.device, dtype=comp_key.dtype)
        merged_id = torch.zeros((comp_key.shape[0], comp_key.shape[1], separators.shape[-1]+1), device=comp_key.device, dtype=comp_key.dtype)
        merged_attn_score = torch.zeros((comp_key.shape[0], comp_key.shape[1], separators.shape[-1]+1), device=comp_key.device, dtype=comp_key.dtype)
        merged_attn_count = torch.zeros((comp_key.shape[0], comp_key.shape[1], separators.shape[-1]+1), device=comp_key.device, dtype=comp_key.dtype)
        for batch in range(comp_key.shape[0]):
            for head in range(comp_key.shape[1]):
                for i in range(separators.shape[-1] + 1):
                    if i == 0:
                        start = 0
                    else:
                        start = separators[batch, head, i-1] + 1
                    if i == separators.shape[-1]:
                        end = comp_key.shape[-2]
                    else:
                        end = separators[batch, head, i] + 1
                    pivot_id = avg_attn_score[batch, head, start : end].argmax() + start
                    initial_g_vals = gaussian_kernel(comp_key[batch, head, pivot_id:pivot_id+1, :], comp_key[batch, head, start : end, :], 1)
                    sigma = initial_g_vals.sum(dim=-1) / (2**0.5 * (end-start))
                    g_vals = gaussian_kernel(comp_key[batch, head, pivot_id:pivot_id+1, :], comp_key[batch, head, start : end, :], sigma)
                    w = g_vals / g_vals.sum(dim=-1)
                    merged_attn_score[batch, head, i] = comp_attn_score[batch, head, start:end].sum(dim=-1) 
                    merged_attn_count[batch, head, i] = comp_attn_count[batch, head, start:end].sum(dim=-1) 
                    merged_key[batch, head, i, :] = w @ comp_key[batch, head, start:end, :] 
                    merged_value[batch, head, i, :] = w @ comp_value[batch, head, start:end, :] * (end-start)
                    merged_id[batch, head, i] = pivot_id

        top_k_idx = avg_attn_score.topk(top_k, -1).indices.sort().values
        top_k_idx_kv = rearrange(top_k_idx, 'b h l -> b h l 1')
        top_k_idx_kv = top_k_idx_kv.repeat(1, 1, 1, key_cache.shape[-1])
        top_k_key = torch.gather(comp_key, -2, top_k_idx_kv)
        top_k_value = torch.gather(comp_value, -2, top_k_idx_kv)
        top_k_attn_score = torch.gather(comp_attn_score, -1, top_k_idx)
        top_k_attn_count = torch.gather(comp_attn_count, -1, top_k_idx)

        tot_idx = torch.cat([top_k_idx, merged_id], dim=-1)
        sort_idx = torch.argsort(tot_idx, stable=True, dim=-1)
        sort_idx_kv = sort_idx[..., None].repeat(1,1,1,key_cache.shape[-1])
        tot_key = torch.cat([top_k_key, merged_key], dim=-2)
        tot_key = torch.gather(tot_key, -2, sort_idx_kv)
        tot_value = torch.cat([top_k_value, merged_value], dim=-2)
        tot_value = torch.gather(tot_value, -2, sort_idx_kv)
        tot_attn_score = torch.cat([top_k_attn_score, merged_attn_score], dim=-1)
        tot_attn_score = torch.gather(tot_attn_score, -1, sort_idx)
        tot_attn_count = torch.cat([top_k_attn_count, merged_attn_count], dim=-1)
        tot_attn_count = torch.gather(tot_attn_count, -1, sort_idx)


        self.key_cache[layer_idx] = torch.cat((tot_key, window_key), -2)
        self.value_cache[layer_idx] = torch.cat((tot_value, window_value), -2)
        self.attn_cache[layer_idx] = torch.cat((tot_attn_score, window_attn_score), -1)
        self.attn_count[layer_idx] = torch.cat((tot_attn_count, window_attn_count), -1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


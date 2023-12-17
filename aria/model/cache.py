from typing import Optional

import torch


class KVCache(torch.nn.Module):
    def __init__(
        self,
        max_batch_size,
        n_head,
        d_head,
        dtype=torch.float16,
        max_size=8192,
        rolling=True,
    ):
        """
        Cache for key-value pairs used in self-attention.
        Args:
            max_batch_size: the maximum batch size
            n_head: the number of heads
            d_head: the dimension of each head
            dtype: the dtype of the cache
            max_size: the maximum number of positions to cache
            rolling: whether to roll when it is full
        """
        super().__init__()
        self.rolling = rolling
        self.shape = (max_batch_size, max_size, n_head, d_head)
        self.register_buffer(
            "k_cache", torch.empty(self.shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.empty(self.shape, dtype=dtype), persistent=False
        )
        self.next_pos = 0

    def _get_tensor(self, cache, start_pos, next_pos):
        if self.rolling and next_pos > self.shape[1]:
            return torch.cat(
                [
                    cache[:, next_pos % self.shape[1] :],
                    cache[:, : next_pos % self.shape[1]],
                ],
                dim=1,
            )
        else:
            return cache[:, start_pos:next_pos]

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        max_pos: Optional[int] = None,
    ):
        """
        Update the kv cache and return the new k, v sequences of vectors

        Args:
            k: key to update. Shape: (batch_size, num_positions, n_head, d_head)
            v: value to update. Shape: (batch_size, num_positions, n_head, d_head)
            pos: positions to update. Shape: (num_positions,).
                 Example: None to append to the end of the cache.
                          [0, 1, 2, 3, 4] to update the first 5 positions.
                          [5] to only update the 6th position.
            start_pos: the starting position of the cache. Default to 0
            max_pos: the maximum position to update. Default to None.
                     Only used when pos is *NOT* None. Can be inferred from pos.max(),
                     but such an operation causes a sync with massive overhead
                     due to dynamic shape.
        """
        if pos is None:
            k_pos = torch.arange(
                self.next_pos, self.next_pos + k.size(1), device=k.device
            )
            v_pos = torch.arange(
                self.next_pos, self.next_pos + v.size(1), device=v.device
            )
            if self.rolling:
                k_pos = k_pos % self.shape[1]
                v_pos = v_pos % self.shape[1]
            self.k_cache[: k.size(0), k_pos] = k
            self.v_cache[: v.size(0), v_pos] = v
            self.next_pos += k.size(1)
        else:
            assert pos.size(0) == k.size(1)
            assert max_pos is not None, (
                "Need to pass in `pos.max()` explicitly. "
                "Doing `pos.max()` creates massive overhead."
            )
            if self.rolling:
                pos = pos % self.shape[1]
            self.k_cache[: k.size(0), pos] = k
            self.v_cache[: v.size(0), pos] = v
            # Update next_pos using the max entry.
            # Note: `self.next_pos = pos.max() + 1` could have worked, but it
            #       causes the shape to be dynamic and creates a massive overhead.
            self.next_pos = max_pos + 1

        return self._get_tensor(
            self.k_cache, start_pos, self.next_pos
        ), self._get_tensor(self.v_cache, start_pos, self.next_pos)

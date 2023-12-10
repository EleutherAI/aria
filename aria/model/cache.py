import torch


class KVCache(torch.nn.Module):
    def __init__(
        self, max_batch_size, n_head, d_head, dtype=torch.float16, max_size=8192
    ):
        super().__init__()
        self.max_size = max_size
        self.shape = (max_batch_size, max_size, n_head, d_head)
        self.register_buffer(
            "k_cache", torch.empty(self.shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.empty(self.shape, dtype=dtype), persistent=False
        )
        self.next_pos = 0

    def update(self, pos, k, v):
        """
        Update the kv cache and return the new k, v sequences of vectors
        Note: `self.next_pos` is always updated to the max entry of `pos` plus one.
              This means that one can rewind the cache by passing in a smaller `pos`.
        Args:
            pos: positions to update. Shape: (num_positions,).
                 Example: [0, 1, 2, 3, 4] to update the first 5 positions.
                          [5] to only update the 6th position.
            k: key to update. Shape: (batch_size, num_positions, n_head, d_head)
            v: value to update. Shape: (batch_size, num_positions, n_head, d_head)
        """
        self.k_cache[: k.size(0), pos] = k
        self.v_cache[: v.size(0), pos] = v
        self.next_pos = pos.max() + 1
        return (
            self.k_cache[: k.size(0), : self.next_pos],
            self.v_cache[: v.size(0), : self.next_pos],
        )

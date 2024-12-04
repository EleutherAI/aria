"""Contains miscellaneous utilities"""


def _load_weight(ckpt_path: str, device="cpu"):
    if ckpt_path.endswith("safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                f"Please install safetensors in order to read from the checkpoint: {ckpt_path}"
            ) from e
        return load_file(ckpt_path, device=device)
    else:
        import torch

        return torch.load(ckpt_path, map_location=device)

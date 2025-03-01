"""GPU utilities for TAMP improvisational policies."""

from typing import Any, Union

import numpy as np
import torch


def __init__(self, device_name: str = "cuda"):
    self.device = get_device(device_name)
    print(f"DeviceContext initialized with device: {self.device}")
    if self.device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(self.device)}")
        print(
            f"  CUDA memory allocated: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"  # pylint: disable=line-too-long
        )
        print(
            f"  CUDA memory cached: {torch.cuda.memory_reserved(self.device) / 1e9:.2f} GB"  # pylint: disable=line-too-long
        )


def get_device(device_name: str = "cuda") -> torch.device:
    """Get PyTorch device."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_torch(
    data: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert data to torch tensor on specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, (list, float, int)):
        data = np.array(data)
    return torch.tensor(data, device=device, dtype=dtype)


def to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


class DeviceContext:
    """Context for managing device placement of data."""

    def __init__(self, device_name: str = "cuda"):
        self.device = get_device(device_name)

    def __call__(self, data: Any) -> Any:
        """Convert data to torch tensor on device."""
        return to_torch(data, self.device)

    def numpy(self, data: Any) -> np.ndarray:
        """Convert data back to numpy."""
        return to_numpy(data)

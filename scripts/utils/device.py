import torch
import platform

def get_best_device():
    """
    Returns the best available device:
    - CUDA (Windows/Linux with NVIDIA)
    - MPS (Mac Silicon)
    - CPU fallback
    """

    # Windows / Linux with NVIDIA
    if torch.cuda.is_available():
        return "cuda"

    # macOS M1/M2/M3
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"

    # Default
    return "cpu"

DEVICE = get_best_device()

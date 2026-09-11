"""Detect CUDA / ROCm and validate real GPU kernels before enabling acceleration."""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

def gpu_backend() -> str:
    """ROCm exposes its devices through torch.cuda, too."""
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def canary_kernel_ok() -> bool:
    """Check matrix, convolution and FFT support on the installed GPU runtime."""
    try:
        a = torch.ones(8, 8, device="cuda")
        # Force a real kernel launch + a device->host sync so a lazy failure surfaces here.
        if (a @ a).sum().item() != 512:
            return False
        conv = F.conv2d(a[None, None], torch.ones(1, 1, 3, 3, device="cuda"))
        if conv.sum().item() != 324:
            return False
        restored = torch.fft.irfft2(torch.fft.rfft2(a), s=a.shape)
        return bool(torch.allclose(restored, a))
    except Exception as exc:
        logging.getLogger(__name__).warning("GPU probe failed: %s", exc)
        return False


def resolve_device(pref: str = "auto") -> torch.device:
    """Resolve a usable torch device.

    Args:
        pref: ``"auto"`` (CUDA if usable else CPU), ``"cuda"`` (try CUDA, still
            fall back to CPU if the canary fails), ``"rocm"`` (require HIP),
            or ``"cpu"`` (force CPU). ROCm uses torch.device("cuda") internally.
            The legacy cuda preference accepts either GPU runtime.
    """
    pref = str(pref).lower()
    if pref not in ("auto", "cuda", "rocm", "cpu"):
        raise ValueError(f"Unknown device preference: {pref}")
    if pref == "cpu" or (pref == "rocm" and gpu_backend() != "rocm"):
        return torch.device("cpu")
    try:
        if torch.cuda.is_available() and canary_kernel_ok():
            return torch.device("cuda")
    except Exception as exc:
        logging.getLogger(__name__).warning("GPU detection failed: %s", exc)
    return torch.device("cpu")


def device_banner_text(dev: torch.device) -> str:
    """A short human-readable status banner for the resolved device."""
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(dev)
        except Exception:
            name = "CUDA"
        backend = "ROCm / HIP" if gpu_backend() == "rocm" else "CUDA"
        return f"Engine: {backend} ({name})"
    return "Engine: CPU (slow) - install a compatible AMD ROCm or NVIDIA CUDA PyTorch build for GPU acceleration"

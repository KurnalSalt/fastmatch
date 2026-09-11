"""Backend routing is tested without requiring AMD/NVIDIA hardware."""
import pytest
import torch
from fastmatch import device
from fastmatch.__main__ import _build_parser

@pytest.mark.parametrize("hip", [None, "7.2.1"])
def test_auto_gpu(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: True)
    assert device.resolve_device("auto").type == "cuda"
    assert device.resolve_device("rocm").type == ("cuda" if hip else "cpu")
    assert device.resolve_device("cpu").type == "cpu"

@pytest.mark.parametrize("pref", ["auto", "cuda", "rocm"])
def test_failed_canary_falls_back(monkeypatch, pref):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: False)
    assert device.resolve_device(pref).type == "cpu"

def test_rocm_banner_and_cli(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "AMD Radeon RX 7900 XT")
    assert device.device_banner_text(torch.device("cuda")) == "Engine: ROCm / HIP (AMD Radeon RX 7900 XT)"
    assert _build_parser().parse_args(["--device", "rocm"]).device == "rocm"

def test_invalid_preference():
    with pytest.raises(ValueError):
        device.resolve_device("typo")

def test_detection_exception_falls_back(monkeypatch):
    def fail():
        raise RuntimeError("driver init failed")
    monkeypatch.setattr(torch.cuda, "is_available", fail)
    assert device.resolve_device("auto").type == "cpu"

@pytest.mark.parametrize("generation", ["RDNA1", "RDNA2", "RDNA3", "RDNA4"])
def test_no_generation_whitelist(monkeypatch, generation):
    monkeypatch.setattr(torch.version, "hip", "compatible-build")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: generation)
    monkeypatch.setattr(device, "canary_kernel_ok", lambda: True)
    assert device.resolve_device("rocm").type == "cuda"

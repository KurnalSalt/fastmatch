# Desktop release / 桌面发布版

This fork adds AMD ROCm detection, English / Simplified Chinese switching,
configurable multicore CPU computation, and unlimited results by default.
It retains FastMatch's GPL-2.0 license and original author attribution.

## Using the app / 使用方法

- Windows: extract the entire archive and open `FastMatch/FastMatch.exe`.
  Keep the `_internal` directory beside the executable. Python installation is not required.
- macOS: extract the archive and open `FastMatch.app`. Choose the arm64 build for
  Apple Silicon or the x64 build for Intel. These community builds are not Apple-notarized.
- Switch language using **Language → English / 简体中文**. The choice persists.
- **Engine → CPU threads** selects the PyTorch and OpenCV thread count.
  Changing this setting is disabled while a search is running.
- **Max results / 结果上限 = Unlimited / 不限 (0)** retains all results after
  overlap removal; a positive value applies an optional cap.
- NCC / SSD / CCORR use ROCm when available. OpenCV feature matching uses CPU.
- macOS packages currently use CPU computation. ROCm is not a macOS backend.

Windows ROCm builds include AMD's runtime. A compatible graphics driver is still
required. RDNA generation alone does not establish driver/runtime support; see
[Windows ROCm notes](WINDOWS_ROCM.md). Unsupported GPUs fall back to CPU.

The small `FastMatch.exe` in the source directory is a development launcher and
requires the local `.venv`; the executable inside a release archive is standalone.

## Build

Use Python 3.12 and build on the target operating system:

```sh
python -m pip install -r requirements-app.txt pyinstaller
# CPU / Apple Silicon:
python -m pip install torch==2.9.1
# Windows ROCm instead:
python -m pip install -r requirements-rocm-windows.txt
python -m PyInstaller --noconfirm packaging/fastmatch.spec
```

Intel Macs need the final upstream x86 PyTorch line:

```sh
python -m pip install -r requirements-app.txt -c packaging/constraints-macos-intel.txt torch pyinstaller
python -m PyInstaller --noconfirm packaging/fastmatch.spec
```

The **Desktop release builds** GitHub Actions workflow builds Windows x64 CPU,
macOS arm64 and macOS x64 packages. Each package runs a GUI + matcher smoke test
before being uploaded as an artifact. Builds use macOS 15 runners.

## Local validation

Windows / RX 7900 XT (gfx1100, 20 GB), PyTorch 2.9.1 + ROCm 7.2.1:
matrix, convolution and FFT startup probes passed; all six NCC / SSD / CCORR
GPU integration checks passed. No real hardware testing was performed on
RDNA 1, 2 or 4 GPUs.

A synthetic 768×768 CPU search measured approximately 0.59 s with one thread,
0.22 s with 12 threads and 0.21 s with 24 threads on this machine. This is a small
benchmark, not a guarantee for large images; memory traffic and workload affect scaling.

Frozen application logs: Windows `%LOCALAPPDATA%/FastMatch/logs/fastmatch.log`;
macOS `~/Library/Logs/FastMatch/fastmatch.log`.

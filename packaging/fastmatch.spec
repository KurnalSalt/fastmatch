# PyInstaller: build natively on Windows or macOS with the chosen torch runtime.
from pathlib import Path
import importlib.util
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = [(str(root / 'LICENSE'), '.')]
hidden = []
for module in ('rocm_sdk', 'rocm_sdk_core', 'rocm_sdk_libraries_custom',
               '_rocm_sdk_core', '_rocm_sdk_libraries_custom'):
    spec = importlib.util.find_spec(module)
    if spec is not None:
        hidden.extend(collect_submodules(module, filter=lambda name: '.tests' not in name))
        # Preserve runtime DLL, kernel database and compiler paths used by HIP.
        folder = Path(spec.origin).parent
        datas.append((str(folder), module))
for distribution in ('torch', 'rocm', 'rocm-sdk-core', 'rocm-sdk-libraries-custom'):
    try:
        datas.extend(copy_metadata(distribution))
    except Exception:
        pass

a = Analysis([str(root / 'scripts' / 'desktop_entry.py')], pathex=[str(root)],
             binaries=[], datas=datas, hiddenimports=hidden,
             excludes=['pytest', 'torchvision', 'torchaudio', 'matplotlib',
                       'IPython', 'notebook', 'tensorboard'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='FastMatch',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, argv_emulation=False)
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='FastMatch')
if sys.platform == 'darwin':
    app = BUNDLE(collection, name='FastMatch.app', bundle_identifier='org.fastmatch.desktop',
                 info_plist={'CFBundleShortVersionString': '0.2.1',
                             'NSHighResolutionCapable': True})

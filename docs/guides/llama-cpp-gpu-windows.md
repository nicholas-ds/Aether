# llama-cpp-python GPU Setup (Windows, Blackwell)

## Working configuration

- GPU: RTX 5070 Ti (sm_120, Blackwell consumer)
- CPU: Ryzen 5 5600 (AVX2, no AVX-512)
- Python: 3.13
- Wheel: `llama_cpp_python-0.3.34+cu130.basic` from [JamePeng](https://github.com/JamePeng/llama-cpp-python/releases)
- CUDA Toolkit: 13.0 (required for the DLL runtime — toolkit, not just driver)

## The DLL fix

Python 3.8+ does not use PATH to find DLL dependencies. CUDA DLLs must be registered explicitly before importing llama_cpp:

```python
import os, sys
if sys.platform == "win32":
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    for ver in os.listdir(cuda_base):
        for sub in ("bin\\x64", "bin"):
            p = os.path.join(cuda_base, ver, sub)
            if os.path.isdir(p):
                os.add_dll_directory(p)
```

This is already in `src/models/backends/llama_cpp.py`.

## Wheels that did NOT work

- Official PyPI `cu124` — no Blackwell support
- dougeeai `sm100.blackwell` — targets datacenter Blackwell (B100/B200), compiled with AVX-512 which crashes on Ryzen 5 5600
- JamePeng `cu130.basic` without CUDA Toolkit 13.0 installed — CUDA DLLs not found

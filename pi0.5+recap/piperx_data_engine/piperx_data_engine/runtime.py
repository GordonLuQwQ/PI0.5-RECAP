"""Process-local compatibility setup for the existing Conda/Genesis installation."""

import ctypes
import os
import sys
from pathlib import Path


def init_genesis(backend="cpu", seed=0):
    # Do this before importing OpenGL/Genesis; do not alter the user's environment.
    if sys.platform == "linux" and (Path(sys.prefix) / "conda-meta").is_dir():
        library = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    cache = Path(os.environ.get("PIPERX_CACHE_DIR", Path.home() / ".cache" / "piperx_data_engine")).expanduser()
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(cache / "genesis"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import genesis as gs

    if backend == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to this process. Use --backend cpu or run with GPU device access.")
    gs.init(backend=gs.cuda if backend == "gpu" else gs.cpu, seed=seed, logging_level="warning")
    return gs

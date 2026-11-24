# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import ctypes
import subprocess
import sys
from pathlib import Path

import torch

_LIB_INSTANCE: ctypes.CDLL | None = None
_KERNEL_AVAILABLE: bool | None = None


def compile_and_load_ternary_kernel() -> ctypes.CDLL | None:
    global _LIB_INSTANCE, _KERNEL_AVAILABLE
    if _LIB_INSTANCE is not None:
        return _LIB_INSTANCE
    if _KERNEL_AVAILABLE is False:
        return None

    curr_dir = Path(__file__).parent.resolve()
    root_so = curr_dir.parent.parent / "libpaged_gemm.so"
    if root_so.exists():
        try:
            lib = ctypes.CDLL(str(root_so))
            if hasattr(lib, "ternary_addsub_cpu_kernel"):
                lib.ternary_addsub_cpu_kernel.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
                ]
                lib.ternary_addsub_cpu_kernel.restype = None
                _LIB_INSTANCE = lib
                _KERNEL_AVAILABLE = True
                return lib
        except Exception:
            pass

    proj_dir = curr_dir.parent.parent
    so_path = curr_dir / "libternary.so"

    try:
        if not so_path.exists():
            cmd = ["make", "-C", str(proj_dir), "trace/ops/libternary.so"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[Aviso] No se pudo compilar libternary.so: {res.stderr}", file=sys.stderr)
                _KERNEL_AVAILABLE = False
                return None

        lib = ctypes.CDLL(str(so_path))
        lib.ternary_addsub_cpu_kernel.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]
        lib.ternary_addsub_cpu_kernel.restype = None
        _LIB_INSTANCE = lib
        _KERNEL_AVAILABLE = True
        return lib
    except Exception as e:
        print(f"[Aviso] Error cargando kernel C ternario: {e}", file=sys.stderr)
        _KERNEL_AVAILABLE = False
        return None


def is_kernel_available() -> bool:
    global _KERNEL_AVAILABLE
    if _KERNEL_AVAILABLE is None:
        compile_and_load_ternary_kernel()
    return bool(_KERNEL_AVAILABLE)


def ternary_addsub_matmul(x_int8: torch.Tensor, w_codes: torch.Tensor) -> torch.Tensor:
    """
    Multiplicación de matrices ternarias sin multiplicaciones:
    x_int8: [..., K] tensor int8
    w_codes: [N, K] tensor int8 con valores {-1, 0, +1}
    Retorna: [..., N] tensor int32 calculado exclusivamente por sumas y restas con OpenMP/AVX2.
    Si el kernel nativo no está disponible, realiza fallback equivalente en PyTorch.
    """
    orig_shape = x_int8.shape
    x_2d = x_int8.view(-1, orig_shape[-1]).contiguous()
    w_2d = w_codes.contiguous()

    M, K = x_2d.shape
    N, K_w = w_2d.shape
    assert K == K_w, f"Dimensión interna K incompatible: {K} vs {K_w}"

    lib = compile_and_load_ternary_kernel()
    if lib is not None:
        out_2d = torch.empty((M, N), dtype=torch.int32, device=x_int8.device)
        lib.ternary_addsub_cpu_kernel(
            ctypes.cast(x_2d.data_ptr(), ctypes.c_char_p),
            ctypes.cast(w_2d.data_ptr(), ctypes.c_char_p),
            ctypes.c_void_p(out_2d.data_ptr()),
            M, K, N
        )
        return out_2d.view(*orig_shape[:-1], N)

    # Fallback matemático transparente si no hay gcc/libternary.so
    return torch.matmul(x_2d.to(torch.int32), w_2d.t().to(torch.int32)).view(*orig_shape[:-1], N)

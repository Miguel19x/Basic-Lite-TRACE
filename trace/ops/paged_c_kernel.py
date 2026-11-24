# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import ctypes
import struct
import subprocess
from pathlib import Path

import torch

_LIB_PAGED_INSTANCE = None
_PAGE_HEADER_FMT = "<IIHHfII"
_PAGE_HEADER_SIZE = struct.calcsize(_PAGE_HEADER_FMT)


def get_paged_c_lib() -> ctypes.CDLL:
    global _LIB_PAGED_INSTANCE
    if _LIB_PAGED_INSTANCE is not None:
        return _LIB_PAGED_INSTANCE

    curr_dir = Path(__file__).resolve().parent.parent.parent
    so_path = curr_dir / "libpaged_gemm.so"

    if not so_path.exists():
        cmd = ["make", "-C", str(curr_dir), "libpaged_gemm.so"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Error compilando libpaged_gemm.so:\n{res.stderr}")

    lib = ctypes.CDLL(str(so_path))
    lib.paged_direct_matmul_fast_c.argtypes = [
        ctypes.c_char_p,  # x [M, K]
        ctypes.c_char_p,  # page_blob
        ctypes.c_void_p,  # out [M, rows]
        ctypes.c_int64,   # M
        ctypes.c_int64,   # K
    ]
    lib.paged_direct_matmul_fast_c.restype = None

    lib.paged_direct_matmul_fast_c_multi.argtypes = [
        ctypes.c_char_p,                            # x [M, K]
        ctypes.POINTER(ctypes.c_char_p),           # array of page_blobs
        ctypes.c_int64,                            # num_pages
        ctypes.c_void_p,                            # out [M, total_out_features]
        ctypes.c_int64,                            # M
        ctypes.c_int64,                            # K
        ctypes.c_int64,                            # total_out_features
    ]
    lib.paged_direct_matmul_fast_c_multi.restype = None

    lib.paged_lowrank_matmul_row_c.argtypes = [
        ctypes.c_char_p,  # x [K]
        ctypes.c_char_p,  # w_packed [rows * (K/4)]
        ctypes.c_void_p,  # out [rows]
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # K
    ]
    lib.paged_lowrank_matmul_row_c.restype = None

    _LIB_PAGED_INSTANCE = lib
    return lib


def paged_direct_block_matmul_fast_c(
    q_x_int8: torch.Tensor,
    page_bytes: bytes,
) -> tuple[int, int, float, torch.Tensor]:
    """
    Ejecución fisionada C de multiplicación de una página individual (formato 'fast').
    """
    if q_x_int8.device.type != "cpu" or q_x_int8.dtype != torch.int8:
        raise ValueError(
            f"paged_direct_block_matmul_fast_c requiere tensor int8 en CPU; "
            f"recibido device={q_x_int8.device}, dtype={q_x_int8.dtype}"
        )
    if len(page_bytes) < _PAGE_HEADER_SIZE:
        raise ValueError(f"Página truncada: {len(page_bytes)} < {_PAGE_HEADER_SIZE} bytes")

    row_start, rows, cols, fmt_code, scale_w, _len_p1, _len_p2 = struct.unpack(
        _PAGE_HEADER_FMT, page_bytes[:_PAGE_HEADER_SIZE]
    )
    if fmt_code != 1:
        raise ValueError(f"El kernel C fast solo admite fmt_code=1, recibido: {fmt_code}")

    orig_shape = q_x_int8.shape
    q_2d = q_x_int8.reshape(-1, orig_shape[-1]).contiguous()
    M, K = q_2d.shape
    if K != cols:
        raise ValueError(f"Dimensión K ({K}) no coincide con cols de página ({cols})")

    required_bytes = _PAGE_HEADER_SIZE + rows * (((cols + 31) // 32) * 8)
    if len(page_bytes) < required_bytes:
        raise ValueError(f"Payload de página insuficiente: {len(page_bytes)} < {required_bytes} bytes")

    out_int32 = torch.empty((M, rows), dtype=torch.int32, device=q_x_int8.device)

    lib = get_paged_c_lib()
    lib.paged_direct_matmul_fast_c(
        ctypes.cast(q_2d.data_ptr(), ctypes.c_char_p),
        page_bytes,
        ctypes.c_void_p(out_int32.data_ptr()),
        M,
        K,
    )

    final_shape = list(orig_shape[:-1]) + [rows]
    return row_start, rows, scale_w, out_int32.reshape(*final_shape)


def paged_direct_matmul_fast_c_multi(
    q_x_int8: torch.Tensor,
    pages_list: list[bytes],
    total_out_features: int,
) -> torch.Tensor:
    """
    Ejecución Fisionada Multi-Página de alto rendimiento:
    Procesa todas las páginas del tensor en un único pase C con OpenMP dinámico,
    escribiendo directamente en un único búfer de salida [..., total_out_features].
    Cero copias, cero concatenaciones intermedias en Python.
    """
    if q_x_int8.device.type != "cpu" or q_x_int8.dtype != torch.int8:
        raise ValueError(
            f"paged_direct_matmul_fast_c_multi requiere tensor int8 en CPU; "
            f"recibido device={q_x_int8.device}, dtype={q_x_int8.dtype}"
        )

    orig_shape = q_x_int8.shape
    q_2d = q_x_int8.reshape(-1, orig_shape[-1]).contiguous()
    M, K = q_2d.shape

    num_pages = len(pages_list)
    final_shape = list(orig_shape[:-1]) + [total_out_features]
    if num_pages == 0:
        out_int32 = torch.zeros((M, total_out_features), dtype=torch.int32, device=q_x_int8.device)
        return out_int32.reshape(*final_shape)

    for idx, page in enumerate(pages_list):
        if len(page) < _PAGE_HEADER_SIZE:
            raise ValueError(f"Página {idx} truncada: {len(page)} < {_PAGE_HEADER_SIZE} bytes")
        row_start, rows, cols, fmt_code, _scale_w, _len_p1, _len_p2 = struct.unpack(
            _PAGE_HEADER_FMT, page[:_PAGE_HEADER_SIZE]
        )
        if fmt_code != 1:
            raise ValueError(f"Página {idx}: el kernel C fast solo admite fmt_code=1, recibido: {fmt_code}")
        if cols != K:
            raise ValueError(f"Página {idx}: cols ({cols}) no coincide con K ({K})")
        if row_start + rows > total_out_features:
            raise ValueError(
                f"Página {idx}: row_start({row_start}) + rows({rows}) > total_out_features({total_out_features})"
            )
        required_bytes = _PAGE_HEADER_SIZE + rows * (((cols + 31) // 32) * 8)
        if len(page) < required_bytes:
            raise ValueError(f"Página {idx}: payload insuficiente ({len(page)} < {required_bytes} bytes)")

    out_int32 = torch.empty((M, total_out_features), dtype=torch.int32, device=q_x_int8.device)
    page_ptrs = (ctypes.c_char_p * num_pages)(*pages_list)

    lib = get_paged_c_lib()
    lib.paged_direct_matmul_fast_c_multi(
        ctypes.cast(q_2d.data_ptr(), ctypes.c_char_p),
        page_ptrs,
        num_pages,
        ctypes.c_void_p(out_int32.data_ptr()),
        M,
        K,
        total_out_features,
    )

    return out_int32.reshape(*final_shape)


def paged_lowrank_matmul_c(
    x_int8: torch.Tensor,
    packed_weights: bytes,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """
    Multiplica vector 1D x_int8 [cols] por pesos ternarios 2-bit [rows, cols]
    devolviendo out [rows] en int32.
    """
    if x_int8.device.type != "cpu" or x_int8.dtype != torch.int8:
        raise ValueError(
            f"paged_lowrank_matmul_c requiere tensor int8 en CPU; "
            f"recibido device={x_int8.device}, dtype={x_int8.dtype}"
        )
    x_contig = x_int8.contiguous()
    K = x_contig.numel()
    if K != cols:
        raise ValueError(f"Dimensión K ({K}) != cols ({cols})")
    required_bytes = rows * (((cols + 31) // 32) * 8)
    if len(packed_weights) < required_bytes:
        raise ValueError(f"packed_weights insuficiente: {len(packed_weights)} < {required_bytes} bytes")

    out_int32 = torch.empty((rows,), dtype=torch.int32, device=x_contig.device)
    lib = get_paged_c_lib()
    lib.paged_lowrank_matmul_row_c(
        ctypes.cast(x_contig.data_ptr(), ctypes.c_char_p),
        packed_weights,
        ctypes.c_void_p(out_int32.data_ptr()),
        rows,
        cols,
    )
    return out_int32

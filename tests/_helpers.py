# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Utilidades compartidas para suites de tests (sin código de producción)."""

import ctypes
import os
import struct
from pathlib import Path

from trace.lm.config import TRACELMConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGED_HEADER_FMT = "<IIHHfII"
PAGED_HEADER_SIZE = struct.calcsize(PAGED_HEADER_FMT)

REAL_CAUSAL_CKPT = str(REPO_ROOT / "checkpoints" / "trace_s456_fase1_bitnet_last.pt")
REAL_CAUSAL_BIN = str(REPO_ROOT / "checkpoints" / "trace_s456_fase1_paged_bitcos.bin")
REAL_LM_BIN = str(REPO_ROOT / "checkpoints" / "trace_lm_trained_paged_bitcos.bin")
REAL_LM_TOK = str(REPO_ROOT / "checkpoints" / "trace_tokenizer_8k_bi.bin")


def has_real_causal_checkpoints() -> bool:
    return os.path.exists(REAL_CAUSAL_CKPT) and os.path.exists(REAL_CAUSAL_BIN)


def has_real_lm_checkpoints() -> bool:
    return os.path.exists(REAL_LM_BIN) and os.path.exists(REAL_LM_TOK)


def flip_ternary_payload_inplace(buf: bytearray, start: int, end: int) -> None:
    """Invierte signos ternarios válidos (0 <-> 2), dejando 1 intacto y sin inyectar código 3."""
    for idx in range(start, end):
        b = buf[idx]
        c0 = b & 3
        c1 = (b >> 2) & 3
        c2 = (b >> 4) & 3
        c3 = (b >> 6) & 3
        inv_c0 = (2 - c0) if c0 <= 2 else c0
        inv_c1 = (2 - c1) if c1 <= 2 else c1
        inv_c2 = (2 - c2) if c2 <= 2 else c2
        inv_c3 = (2 - c3) if c3 <= 2 else c3
        buf[idx] = inv_c0 | (inv_c1 << 2) | (inv_c2 << 4) | (inv_c3 << 6)


def make_small_lm_config(vocab_size: int = 512, max_seq_len: int = 256) -> TRACELMConfig:
    """Configuración compacta y determinista de 2 capas para tests rápidos."""
    return TRACELMConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=16,
        v_head_dim=24,
        tie_word_embeddings=True,
    )


def load_paged_gemm_lib() -> ctypes.CDLL:
    so_path = str(REPO_ROOT / "libpaged_gemm.so")
    assert os.path.exists(so_path), f"El binario {so_path} no existe. Compila con 'make all'."
    lib = ctypes.CDLL(so_path)

    lib.paged_direct_matmul_fast_c.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.paged_direct_matmul_fast_c.restype = None

    lib.paged_direct_matmul_fast_c_multi.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.paged_direct_matmul_fast_c_multi.restype = None

    lib.paged_lowrank_matmul_row_c.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.paged_lowrank_matmul_row_c.restype = None

    lib.ternary_addsub_cpu_kernel.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.ternary_addsub_cpu_kernel.restype = None

    lib.trace_dot_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.trace_dot_f32.restype = ctypes.c_float

    lib.trace_axpy_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.trace_axpy_f32.restype = None

    for sym in ("paged_absorb_uk_f32", "paged_absorb_uv_f32"):
        fn = getattr(lib, sym)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = None

    lib.trace_cpu_active_isa.argtypes = []
    lib.trace_cpu_active_isa.restype = ctypes.c_char_p
    lib.trace_cpu_set_isa.argtypes = [ctypes.c_char_p]
    lib.trace_cpu_set_isa.restype = None
    return lib


def load_trace_infer_lib() -> ctypes.CDLL:
    so_path = str(REPO_ROOT / "libtrace_infer.so")
    assert os.path.exists(so_path), f"No existe {so_path}. Compila con 'make all'."
    lib = ctypes.CDLL(so_path)

    lib.trace_model_sizeof.argtypes = []
    lib.trace_model_sizeof.restype = ctypes.c_size_t

    lib.trace_model_load_odirect.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.trace_model_load_odirect.restype = ctypes.c_int

    lib.trace_model_free.argtypes = [ctypes.c_void_p]
    lib.trace_model_free.restype = None

    lib.trace_model_get_used_odirect.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_used_odirect.restype = ctypes.c_int

    lib.trace_model_get_pread_calls.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_pread_calls.restype = ctypes.c_uint64

    lib.trace_model_get_pages_offset.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_pages_offset.restype = ctypes.c_size_t

    lib.trace_forward_backbone.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trace_forward_backbone.restype = None

    lib.trace_forward_backbone_lazy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trace_forward_backbone_lazy.restype = None

    lib.trace_generate_causal_two_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_generate_causal_two_step.restype = ctypes.c_int
    return lib


def load_trace_lm_lib() -> ctypes.CDLL:
    so_path = str(REPO_ROOT / "libtrace_lm.so")
    assert os.path.exists(so_path), f"No se encontró {so_path}. Ejecuta 'make all'."
    lib = ctypes.CDLL(so_path)

    lib.trace_lm_load.argtypes = [ctypes.c_char_p]
    lib.trace_lm_load.restype = ctypes.c_void_p

    lib.trace_lm_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_free.restype = None

    lib.trace_lm_get_config.argtypes = [ctypes.c_void_p]
    lib.trace_lm_get_config.restype = ctypes.c_void_p

    lib.trace_lm_workspace_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.trace_lm_workspace_create.restype = ctypes.c_void_p

    lib.trace_lm_workspace_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_workspace_free.restype = None

    lib.trace_lm_kv_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.trace_lm_kv_create.restype = ctypes.c_void_p

    lib.trace_lm_kv_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_free.restype = None

    lib.trace_lm_kv_reset.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_reset.restype = None

    lib.trace_lm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.trace_lm_forward.restype = ctypes.c_int

    lib.trace_lm_sample_greedy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.trace_lm_sample_greedy.restype = ctypes.c_int

    lib.trace_lm_get_pread_calls.argtypes = [ctypes.c_void_p]
    lib.trace_lm_get_pread_calls.restype = ctypes.c_int

    lib.trace_lm_get_used_odirect.argtypes = [ctypes.c_void_p]
    lib.trace_lm_get_used_odirect.restype = ctypes.c_int

    lib.trace_lm_kv_get_last_common_prefix.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_get_last_common_prefix.restype = ctypes.c_int

    lib.trace_cpu_active_isa.argtypes = []
    lib.trace_cpu_active_isa.restype = ctypes.c_char_p

    lib.trace_cpu_set_isa.argtypes = [ctypes.c_char_p]
    lib.trace_cpu_set_isa.restype = None
    return lib


def load_trace_tokenizer_lib() -> ctypes.CDLL:
    so_path = str(REPO_ROOT / "libtrace_tokenizer.so")
    assert os.path.exists(so_path), f"El binario {so_path} no existe. Compila con 'make all'."
    lib = ctypes.CDLL(so_path)

    lib.trace_tokenizer_sizeof.restype = ctypes.c_size_t
    lib.trace_tokenizer_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.trace_tokenizer_load.restype = ctypes.c_int
    lib.trace_tokenizer_free.argtypes = [ctypes.c_void_p]
    lib.trace_tokenizer_free.restype = None

    lib.trace_tokenizer_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.trace_tokenizer_encode.restype = ctypes.c_int

    lib.trace_tokenizer_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_tokenizer_decode.restype = ctypes.c_int
    return lib

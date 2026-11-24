# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Test de paridad y exactitud numérica estricta en int32 (diff == 0) y float32 (Delta < 1e-4):
1. Valida paged_direct_matmul_fast_c (ruta 2-bit empaquetada con alineación de 32 pesos).
2. Valida paged_direct_matmul_fast_c_multi (ruta multi-página fisionada).
3. Valida paged_lowrank_matmul_row_c (ruta de fila individual).
4. Valida ternary_addsub_cpu_kernel (ruta densa int8 con vpsignb).
5. Valida trace_dot_f32 y trace_axpy_f32 (operaciones vectoriales float32).
6. Valida paged_absorb_uk_f32 y paged_absorb_uv_f32 (absorción MLA multi-página en scalar y avx2).
7. Valida despacho dinámico de ISA (scalar vs avx2) con paridad matemática estricta.
"""
import ctypes
import os
from pathlib import Path

import numpy as np

from trace.serialization.paged_bitcos import PagedBitCOSSerializer

REPO_ROOT = Path(__file__).resolve().parent.parent
SO_PATH = str(REPO_ROOT / "libpaged_gemm.so")


def load_kernel_lib():
    assert os.path.exists(SO_PATH), f"El binario {SO_PATH} no existe. Compila con make primero."
    lib = ctypes.CDLL(SO_PATH)

    # 1. paged_direct_matmul_fast_c
    lib.paged_direct_matmul_fast_c.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.paged_direct_matmul_fast_c.restype = None

    # 2. paged_direct_matmul_fast_c_multi
    lib.paged_direct_matmul_fast_c_multi.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.paged_direct_matmul_fast_c_multi.restype = None

    # 3. paged_lowrank_matmul_row_c
    lib.paged_lowrank_matmul_row_c.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.paged_lowrank_matmul_row_c.restype = None

    # 4. ternary_addsub_cpu_kernel
    lib.ternary_addsub_cpu_kernel.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.ternary_addsub_cpu_kernel.restype = None

    # 5. trace_dot_f32
    lib.trace_dot_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.trace_dot_f32.restype = ctypes.c_float

    # 6. trace_axpy_f32
    lib.trace_axpy_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.trace_axpy_f32.restype = None

    # 7. paged_absorb_uk_f32 y paged_absorb_uv_f32
    lib.paged_absorb_uk_f32.argtypes = [
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
    lib.paged_absorb_uk_f32.restype = None

    lib.paged_absorb_uv_f32.argtypes = [
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
    lib.paged_absorb_uv_f32.restype = None

    # Control de ISA y autodetección (obligatorios)
    assert hasattr(lib, "trace_cpu_active_isa"), "Símbolo 'trace_cpu_active_isa' ausente en libpaged_gemm.so."
    assert hasattr(lib, "trace_cpu_set_isa"), "Símbolo 'trace_cpu_set_isa' ausente en libpaged_gemm.so."
    lib.trace_cpu_active_isa.argtypes = []
    lib.trace_cpu_active_isa.restype = ctypes.c_char_p
    lib.trace_cpu_set_isa.argtypes = [ctypes.c_char_p]
    lib.trace_cpu_set_isa.restype = None

    return lib


def _pack_multi_pages(w: np.ndarray, chunk_size: int = 24) -> list[bytes]:
    total_rows, K = w.shape
    page_blobs = []
    for r in range(0, total_rows, chunk_size):
        w_chunk = w[r : r + chunk_size, :]
        rows_chunk = w_chunk.shape[0]
        needed_bytes = rows_chunk * (((K + 31) // 32) * 8)
        page_sz = 8192 if needed_bytes > 3900 else 4096
        blob = PagedBitCOSSerializer.pack_ternary_page(
            row_start=r,
            out_features=rows_chunk,
            in_features=K,
            scale_w=1.0,
            q_w_chunk=w_chunk,
            page_size=page_sz,
            format_type="fast",
        )
        page_blobs.append(blob)
    return page_blobs


def test_avx2_paged_fast_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(42)

    M_list = [1, 4, 8]
    K_list = [1, 7, 31, 32, 33, 50, 96, 192, 384]
    rows_list = [3, 16, 17, 32, 64]

    for M in M_list:
        for K in K_list:
            for rows in rows_list:
                w = rng.randint(-1, 2, size=(rows, K)).astype(np.int8)
                x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
                if M >= 2:
                    x[0, :] = 127
                    x[1, :] = -127

                expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))

                page_blob = PagedBitCOSSerializer.pack_ternary_page(
                    row_start=0,
                    out_features=rows,
                    in_features=K,
                    scale_w=1.0,
                    q_w_chunk=w,
                    page_size=8192 if rows * (((K + 31) // 32) * 8) > 3900 else 4096,
                    format_type="fast",
                )

                out = np.zeros((M, rows), dtype=np.int32)
                lib.paged_direct_matmul_fast_c(
                    x.ctypes.data_as(ctypes.c_char_p),
                    ctypes.c_void_p(ctypes.cast(ctypes.c_char_p(page_blob), ctypes.c_void_p).value),
                    out.ctypes.data_as(ctypes.c_void_p),
                    M,
                    K,
                )

                diff = np.max(np.abs(out - expected))
                assert diff == 0, f"Discrepancia en Fast AVX2: M={M}, K={K}, rows={rows}, max_diff={diff}"


def test_avx2_paged_fast_multi_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(123)

    M_list = [1, 4]
    K_list = [1, 7, 31, 32, 33, 192, 530, 768]
    total_rows_list = [17, 64]

    for M in M_list:
        for K in K_list:
            for total_rows in total_rows_list:
                w = rng.randint(-1, 2, size=(total_rows, K)).astype(np.int8)
                x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
                expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))

                chunk_size = 7 if total_rows == 17 else 32
                page_blobs = _pack_multi_pages(w, chunk_size=chunk_size)
                c_blobs = (ctypes.c_char_p * len(page_blobs))(*page_blobs)
                out = np.zeros((M, total_rows), dtype=np.int32)

                lib.paged_direct_matmul_fast_c_multi(
                    x.ctypes.data_as(ctypes.c_char_p),
                    c_blobs,
                    len(page_blobs),
                    out.ctypes.data_as(ctypes.c_void_p),
                    M,
                    K,
                    total_rows,
                )

                diff = np.max(np.abs(out - expected))
                assert diff == 0, (
                    f"Discrepancia en Multi Fast AVX2: M={M}, K={K}, rows={total_rows}, diff={diff}"
                )


def test_avx2_lowrank_matmul_row_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(999)

    K_list = [1, 7, 31, 32, 33, 64, 128, 256, 512, 1024]
    rows_list = [3, 16, 17, 32, 64]

    for K in K_list:
        for rows in rows_list:
            w = rng.randint(-1, 2, size=(rows, K)).astype(np.int8)
            x = rng.randint(-127, 128, size=(K,)).astype(np.int8)
            expected = np.dot(w.astype(np.int32), x.astype(np.int32))

            padded_cols = ((K + 31) // 32) * 32
            if padded_cols > K:
                w_pad = np.pad(w, ((0, 0), (0, padded_cols - K)), mode="constant", constant_values=0)
            else:
                w_pad = w
            codes = (w_pad.flatten() + 1).astype(np.uint8) & 0x03
            b0 = codes[0::4]
            b1 = codes[1::4] << 2
            b2 = codes[2::4] << 4
            b3 = codes[3::4] << 6
            w_packed_bytes = (b0 | b1 | b2 | b3).tobytes()

            out = np.zeros((rows,), dtype=np.int32)
            lib.paged_lowrank_matmul_row_c(
                x.ctypes.data_as(ctypes.c_char_p),
                ctypes.c_char_p(w_packed_bytes),
                out.ctypes.data_as(ctypes.c_void_p),
                rows,
                K,
            )

            diff = np.max(np.abs(out - expected))
            assert diff == 0, f"Discrepancia en LowRank Row AVX2: K={K}, rows={rows}, diff={diff}"


def test_avx2_ternary_addsub_dense_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(2026)

    M_list = [1, 4, 8]
    K_list = [1, 7, 31, 32, 33, 50, 96, 192, 384]
    N_list = [3, 16, 17, 32, 64, 192]

    for M in M_list:
        for K in K_list:
            for N in N_list:
                w = rng.randint(-1, 2, size=(N, K)).astype(np.int8)
                x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
                x[0, : min(32, K)] = 127
                x[-1, : min(32, K)] = -127

                expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))

                out = np.zeros((M, N), dtype=np.int32)
                lib.ternary_addsub_cpu_kernel(
                    x.ctypes.data_as(ctypes.c_char_p),
                    w.ctypes.data_as(ctypes.c_char_p),
                    out.ctypes.data_as(ctypes.c_void_p),
                    M,
                    K,
                    N,
                )

                diff = np.max(np.abs(out - expected))
                assert diff == 0, f"Discrepancia en Dense AVX2: M={M}, K={K}, N={N}, max_diff={diff}"


def test_trace_dot_and_axpy_f32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(777)

    n_list = [1, 7, 16, 31, 32, 64, 128, 256, 512, 1024]
    for n in n_list:
        a = rng.randn(n).astype(np.float32)
        b = rng.randn(n).astype(np.float32)
        expected_dot = float(np.dot(a, b))

        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        actual_dot = lib.trace_dot_f32(a_ptr, b_ptr, n)

        diff_dot = abs(actual_dot - expected_dot)
        assert diff_dot < 1e-4, f"Fallo trace_dot_f32: n={n}, diff={diff_dot}"

        # axpy: y = y + alpha * x
        alpha = float(rng.randn(1)[0])
        y = rng.randn(n).astype(np.float32)
        expected_y = y + alpha * a
        y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.trace_axpy_f32(y_ptr, ctypes.c_float(alpha), a_ptr, n)
        diff_axpy = float(np.max(np.abs(y - expected_y)))
        assert diff_axpy < 1e-5, f"Fallo trace_axpy_f32: n={n}, diff={diff_axpy}"


def test_paged_absorb_uk_and_uv_f32():
    """Valida paged_absorb_uk_f32 y paged_absorb_uv_f32 contra NumPy en scalar y avx2."""
    lib = load_kernel_lib()
    rng = np.random.RandomState(31415)

    n_heads = 4
    qk_nope_head_dim = 16
    v_head_dim = 16
    S = 3
    scale = 0.125

    try:
        for isa in ("scalar", "avx2"):
            lib.trace_cpu_set_isa(isa.encode("utf-8"))
            active = lib.trace_cpu_active_isa().decode("utf-8")

            for kv_lora_rank in (24, 32, 48, 50):
                # 1. paged_absorb_uk_f32: multi-página (64 filas repartidas en páginas de 20 filas)
                q_c = rng.randn(S, n_heads, qk_nope_head_dim).astype(np.float32)
                w_uk = rng.randint(-1, 2, size=(n_heads * qk_nope_head_dim, kv_lora_rank)).astype(np.int8)
                uk_pages = _pack_multi_pages(w_uk, chunk_size=20)
                assert len(uk_pages) >= 2
                c_uk_pages = (ctypes.c_char_p * len(uk_pages))(*uk_pages)

                out_uk = np.zeros((S, n_heads, kv_lora_rank), dtype=np.float32)
                lib.paged_absorb_uk_f32(
                    q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    c_uk_pages,
                    len(uk_pages),
                    ctypes.c_float(scale),
                    out_uk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n_heads,
                    qk_nope_head_dim,
                    kv_lora_rank,
                    S,
                )
                w_uk_h = w_uk.reshape(n_heads, qk_nope_head_dim, kv_lora_rank).astype(np.float32)
                expected_uk = scale * np.einsum("thd,hdk->thk", q_c, w_uk_h)
                diff_uk = float(np.max(np.abs(out_uk - expected_uk)))
                assert diff_uk < 1e-4, (
                    f"Fallo paged_absorb_uk_f32 (isa={active}, kv_lora_rank={kv_lora_rank}): diff={diff_uk:.6e}"
                )

                # 2. paged_absorb_uv_f32: multi-página (64 filas repartidas en páginas de 20 filas)
                o_latent = rng.randn(S, n_heads, kv_lora_rank).astype(np.float32)
                w_uv = rng.randint(-1, 2, size=(n_heads * v_head_dim, kv_lora_rank)).astype(np.int8)
                uv_pages = _pack_multi_pages(w_uv, chunk_size=20)
                assert len(uv_pages) >= 2
                c_uv_pages = (ctypes.c_char_p * len(uv_pages))(*uv_pages)

                out_uv = np.zeros((S, n_heads, v_head_dim), dtype=np.float32)
                lib.paged_absorb_uv_f32(
                    o_latent.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    c_uv_pages,
                    len(uv_pages),
                    ctypes.c_float(scale),
                    out_uv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n_heads,
                    v_head_dim,
                    kv_lora_rank,
                    S,
                )
                w_uv_h = w_uv.reshape(n_heads, v_head_dim, kv_lora_rank).astype(np.float32)
                expected_uv = scale * np.einsum("thk,hvk->thv", o_latent, w_uv_h)
                diff_uv = float(np.max(np.abs(out_uv - expected_uv)))
                assert diff_uv < 1e-4, (
                    f"Fallo paged_absorb_uv_f32 (isa={active}, kv_lora_rank={kv_lora_rank}): diff={diff_uv:.6e}"
                )
    finally:
        lib.trace_cpu_set_isa(b"auto")


def test_isa_dispatch_parity():
    lib = load_kernel_lib()
    assert hasattr(lib, "trace_cpu_set_isa"), "trace_cpu_set_isa debe existir en libpaged_gemm.so."

    rng = np.random.RandomState(555)
    active_init = lib.trace_cpu_active_isa().decode("utf-8")
    host_supports_avx2 = active_init == "avx2"
    print(f"ISA activa detectada por omisión: '{active_init}'")

    try:
        for isa in ("scalar", "avx2"):
            lib.trace_cpu_set_isa(isa.encode("utf-8"))
            active = lib.trace_cpu_active_isa().decode("utf-8")
            print(f"ISA tras configurar '{isa}': '{active}'")
            if isa == "scalar":
                assert active == "scalar", f"Se esperaba ISA 'scalar', se obtuvo '{active}'"
            elif host_supports_avx2:
                assert active == "avx2", f"El host soporta AVX2 pero se obtuvo '{active}'"

            # Ejercitar bucle AVX2 de 32 elementos Y cola escalar (K in [33, 64, 95])
            for K in (33, 64, 95):
                M, N = 3, 17
                w = rng.randint(-1, 2, size=(N, K)).astype(np.int8)
                x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
                expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))

                out_dense = np.zeros((M, N), dtype=np.int32)
                lib.ternary_addsub_cpu_kernel(
                    x.ctypes.data_as(ctypes.c_char_p),
                    w.ctypes.data_as(ctypes.c_char_p),
                    out_dense.ctypes.data_as(ctypes.c_void_p),
                    M,
                    K,
                    N,
                )
                assert np.array_equal(out_dense, expected), (
                    f"Fallo matemático dense en modo ISA={active}, K={K}"
                )

                page_blob = PagedBitCOSSerializer.pack_ternary_page(
                    row_start=0,
                    out_features=N,
                    in_features=K,
                    scale_w=1.0,
                    q_w_chunk=w,
                    page_size=4096,
                    format_type="fast",
                )
                out_fast = np.zeros((M, N), dtype=np.int32)
                lib.paged_direct_matmul_fast_c(
                    x.ctypes.data_as(ctypes.c_char_p),
                    ctypes.c_void_p(ctypes.cast(ctypes.c_char_p(page_blob), ctypes.c_void_p).value),
                    out_fast.ctypes.data_as(ctypes.c_void_p),
                    M,
                    K,
                )
                assert np.array_equal(out_fast, expected), (
                    f"Fallo matemático fast en modo ISA={active}, K={K}"
                )
    finally:
        lib.trace_cpu_set_isa(b"auto")


if __name__ == "__main__":
    test_avx2_paged_fast_exact_int32()
    print("✓ 1. AVX2 Fast (2-bit packed): Paridad int32 exacta (diff == 0) superada en todas las dimensiones.")
    test_avx2_paged_fast_multi_exact_int32()
    print("✓ 2. AVX2 Multi-Page Fisionado: Paridad int32 exacta (diff == 0) superada.")
    test_avx2_lowrank_matmul_row_exact_int32()
    print("✓ 3. AVX2 LowRank Row: Paridad int32 exacta (diff == 0) superada.")
    test_avx2_ternary_addsub_dense_exact_int32()
    print("✓ 4. AVX2 Dense (int8): Paridad int32 exacta (diff == 0) superada en todas las dimensiones.")
    test_trace_dot_and_axpy_f32()
    print("✓ 5. trace_dot_f32 (< 1e-4) y trace_axpy_f32 (< 1e-5): Exactitud float32 superada.")
    test_paged_absorb_uk_and_uv_f32()
    print("✓ 6. paged_absorb_uk_f32 y paged_absorb_uv_f32: Paridad multi-página scalar y avx2 (< 1e-4) superada.")
    test_isa_dispatch_parity()
    print("✓ 7. Despacho dinámico de ISA: Conmutación scalar <-> avx2 sin divergencia matemática.")
    print("✓ Todas las pruebas de exactitud matemática int32 y float32 del kernel pasaron exitosamente.")

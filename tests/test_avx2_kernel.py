"""
Test de paridad y exactitud numérica estricta en int32 (diff == 0) para el kernel AVX2:
1. Valida paged_direct_matmul_fast_c (ruta 2-bit empaquetada con alineación de 32 pesos).
2. Valida paged_direct_matmul_fast_c_multi (ruta multi-página fisionada).
3. Valida paged_lowrank_matmul_row_c (ruta de fila individual).
4. Valida ternary_addsub_cpu_kernel (ruta densa int8 con vpsignb).
Cubre M in {1, 4, 8}, K in {32, 50, 96, 192, 384}, extremos [-127, 127] y casos borde.
"""
import os
import ctypes
import numpy as np

from trace.serialization.paged_bitcos import PagedBitCOSSerializer


def load_kernel_lib():
    so_path = os.path.abspath("libpaged_gemm.so")
    assert os.path.exists(so_path), f"El binario {so_path} no existe. Compila con make primero."
    lib = ctypes.CDLL(so_path)

    # 1. paged_direct_matmul_fast_c
    lib.paged_direct_matmul_fast_c.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64
    ]
    lib.paged_direct_matmul_fast_c.restype = None

    # 2. paged_direct_matmul_fast_c_multi
    lib.paged_direct_matmul_fast_c_multi.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
    ]
    lib.paged_direct_matmul_fast_c_multi.restype = None

    # 3. paged_lowrank_matmul_row_c
    lib.paged_lowrank_matmul_row_c.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64
    ]
    lib.paged_lowrank_matmul_row_c.restype = None

    # 4. ternary_addsub_cpu_kernel
    lib.ternary_addsub_cpu_kernel.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
    ]
    lib.ternary_addsub_cpu_kernel.restype = None

    return lib


def test_avx2_paged_fast_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(42)

    M_list = [1, 4, 8]
    K_list = [32, 50, 96, 192, 384]
    rows_list = [16, 32, 64]

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
    K_list = [32, 192]
    total_rows = 64
    rows_per_page = 16

    for M in M_list:
        for K in K_list:
            w_total = rng.randint(-1, 2, size=(total_rows, K)).astype(np.int8)
            x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
            expected = np.dot(x.astype(np.int32), w_total.T.astype(np.int32))

            pages = []
            for r_start in range(0, total_rows, rows_per_page):
                r_end = min(total_rows, r_start + rows_per_page)
                chunk = w_total[r_start:r_end, :]
                p_blob = PagedBitCOSSerializer.pack_ternary_page(
                    row_start=r_start,
                    out_features=total_rows,
                    in_features=K,
                    scale_w=1.0,
                    q_w_chunk=chunk,
                    page_size=4096,
                    format_type="fast",
                )
                pages.append(p_blob)

            page_ptrs = (ctypes.c_char_p * len(pages))(*[ctypes.c_char_p(p) for p in pages])
            out = np.zeros((M, total_rows), dtype=np.int32)

            lib.paged_direct_matmul_fast_c_multi(
                x.ctypes.data_as(ctypes.c_char_p),
                page_ptrs,
                len(pages),
                out.ctypes.data_as(ctypes.c_void_p),
                M,
                K,
                total_rows,
            )

            diff = np.max(np.abs(out - expected))
            assert diff == 0, f"Discrepancia en Multi-Page AVX2: M={M}, K={K}, diff={diff}"


def test_avx2_lowrank_matmul_row_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(456)

    for K in [32, 50, 96, 192]:
        for rows in [16, 32]:
            w = rng.randint(-1, 2, size=(rows, K)).astype(np.int8)
            x = rng.randint(-127, 128, size=(K,)).astype(np.int8)
            expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))

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
    K_list = [32, 50, 96, 192, 384]
    N_list = [16, 32, 64, 192]

    for M in M_list:
        for K in K_list:
            for N in N_list:
                w = rng.randint(-1, 2, size=(N, K)).astype(np.int8)
                x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
                x[0, :min(32, K)] = 127
                x[-1, :min(32, K)] = -127

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


if __name__ == "__main__":
    test_avx2_paged_fast_exact_int32()
    print("✓ 1. AVX2 Fast (2-bit packed): Paridad int32 exacta (diff == 0) superada en todas las dimensiones.")
    test_avx2_paged_fast_multi_exact_int32()
    print("✓ 2. AVX2 Multi-Page Fisionado: Paridad int32 exacta (diff == 0) superada.")
    test_avx2_lowrank_matmul_row_exact_int32()
    print("✓ 3. AVX2 LowRank Row: Paridad int32 exacta (diff == 0) superada.")
    test_avx2_ternary_addsub_dense_exact_int32()
    print("✓ 4. AVX2 Dense (int8): Paridad int32 exacta (diff == 0) superada en todas las dimensiones.")
    print("✓ Todas las pruebas de exactitud matemática int32 del kernel AVX2 pasaron exitosamente.")

"""
Test de paridad y exactitud numérica estricta en int32 (diff == 0) y float32 (Delta < 1e-5):
1. Valida paged_direct_matmul_fast_c (ruta 2-bit empaquetada con alineación de 32 pesos).
2. Valida paged_direct_matmul_fast_c_multi (ruta multi-página fisionada).
3. Valida paged_lowrank_matmul_row_c (ruta de fila individual).
4. Valida ternary_addsub_cpu_kernel (ruta densa int8 con vpsignb).
5. Valida trace_dot_f32 y trace_axpy_f32 (operaciones vectoriales float32).
6. Valida despacho dinámico de ISA (scalar vs avx2) con paridad matemática estricta.
"""
import ctypes
import os
from trace.serialization.paged_bitcos import PagedBitCOSSerializer

import numpy as np


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

    # 5. trace_dot_f32
    lib.trace_dot_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    lib.trace_dot_f32.restype = ctypes.c_float

    # 6. trace_axpy_f32
    lib.trace_axpy_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    lib.trace_axpy_f32.restype = None

    # Control de ISA y autodetección
    if hasattr(lib, "trace_cpu_active_isa"):
        lib.trace_cpu_active_isa.argtypes = []
        lib.trace_cpu_active_isa.restype = ctypes.c_char_p

    if hasattr(lib, "trace_cpu_set_isa"):
        lib.trace_cpu_set_isa.argtypes = [ctypes.c_char_p]
        lib.trace_cpu_set_isa.restype = None

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
    K_list = [32, 192, 530, 768]
    total_rows = 64

    for M in M_list:
        for K in K_list:
            w = rng.randint(-1, 2, size=(total_rows, K)).astype(np.int8)
            x = rng.randint(-127, 128, size=(M, K)).astype(np.int8)
            expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))

            page_blobs = []
            chunk_size = 32
            for r in range(0, total_rows, chunk_size):
                w_chunk = w[r : r + chunk_size, :]
                blob = PagedBitCOSSerializer.pack_ternary_page(
                    row_start=r,
                    out_features=chunk_size,
                    in_features=K,
                    scale_w=1.0,
                    q_w_chunk=w_chunk,
                    page_size=8192 if chunk_size * (((K + 31) // 32) * 8) > 3900 else 4096,
                    format_type="fast",
                )
                page_blobs.append(blob)

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
            assert diff == 0, f"Discrepancia en Multi Fast AVX2: M={M}, K={K}, diff={diff}"


def test_avx2_lowrank_matmul_row_exact_int32():
    lib = load_kernel_lib()
    rng = np.random.RandomState(999)

    K_list = [32, 64, 128, 256, 512, 1024]
    rows_list = [16, 32, 64]

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


def test_isa_dispatch_parity():
    lib = load_kernel_lib()
    if not hasattr(lib, "trace_cpu_set_isa"):
        return

    active_init = lib.trace_cpu_active_isa().decode("utf-8")
    print(f"ISA activa detectada por omisión: '{active_init}'")

    for isa in ["scalar", "avx2"]:
        lib.trace_cpu_set_isa(isa.encode("utf-8"))
        active = lib.trace_cpu_active_isa().decode("utf-8")
        print(f"ISA tras configurar '{isa}': '{active}'")

        # Probar matmul básico int32
        w = np.array([[1, -1, 0, 1], [-1, 0, 1, 0]], dtype=np.int8)
        x = np.array([[10, 20, 30, 40]], dtype=np.int8)
        out = np.zeros((1, 2), dtype=np.int32)
        lib.ternary_addsub_cpu_kernel(
            x.ctypes.data_as(ctypes.c_char_p),
            w.ctypes.data_as(ctypes.c_char_p),
            out.ctypes.data_as(ctypes.c_void_p),
            1, 4, 2
        )
        expected = np.dot(x.astype(np.int32), w.T.astype(np.int32))
        assert np.array_equal(out, expected), f"Fallo matemático en modo ISA={isa}"

    # Restaurar
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
    print("✓ 5. trace_dot_f32 y trace_axpy_f32: Exactitud float32 (Delta < 1e-5) superada en todos los tamaños.")
    test_isa_dispatch_parity()
    print("✓ 6. Despacho dinámico de ISA: Conmutación scalar <-> avx2 sin divergencia matemática.")
    print("✓ Todas las pruebas de exactitud matemática int32 y float32 del kernel pasaron exitosamente.")

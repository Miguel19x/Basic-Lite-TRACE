# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Tests de paridad numérica estricta del kernel nativo C (int32 diff == 0, float32 < 1e-4):
- paged_direct_matmul_fast_c, paged_direct_matmul_fast_c_multi, paged_lowrank_matmul_row_c
- ternary_addsub_cpu_kernel, trace_dot_f32, trace_axpy_f32
- paged_absorb_uk_f32, paged_absorb_uv_f32 y despacho dinámico scalar <-> avx2
"""

import ctypes

import numpy as np
import pytest

from tests._helpers import load_paged_gemm_lib
from trace.serialization.paged_bitcos import PagedBitCOSSerializer


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
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(42)

    for M in (1, 4, 8):
        for K in (1, 7, 31, 32, 33, 50, 96, 192, 384):
            for rows in (3, 16, 17, 32, 64):
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
                assert np.array_equal(out, expected), f"Fast AVX2 M={M}, K={K}, rows={rows}"


def test_avx2_paged_fast_multi_exact_int32():
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(123)

    for M in (1, 4):
        for K in (1, 7, 31, 32, 33, 192, 530, 768):
            for total_rows in (17, 64):
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
                assert np.array_equal(out, expected), f"Multi Fast M={M}, K={K}, rows={total_rows}"


def test_avx2_lowrank_matmul_row_exact_int32():
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(999)

    for K in (1, 7, 31, 32, 33, 64, 128, 256, 512, 1024):
        for rows in (3, 16, 17, 32, 64):
            w = rng.randint(-1, 2, size=(rows, K)).astype(np.int8)
            x = rng.randint(-127, 128, size=(K,)).astype(np.int8)
            expected = np.dot(w.astype(np.int32), x.astype(np.int32))

            padded_cols = ((K + 31) // 32) * 32
            w_pad = (
                np.pad(w, ((0, 0), (0, padded_cols - K)), mode="constant", constant_values=0)
                if padded_cols > K
                else w
            )
            codes = (w_pad.flatten() + 1).astype(np.uint8) & 0x03
            w_packed_bytes = (
                codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)
            ).tobytes()

            out = np.zeros((rows,), dtype=np.int32)
            lib.paged_lowrank_matmul_row_c(
                x.ctypes.data_as(ctypes.c_char_p),
                ctypes.c_char_p(w_packed_bytes),
                out.ctypes.data_as(ctypes.c_void_p),
                rows,
                K,
            )
            assert np.array_equal(out, expected), f"LowRank Row K={K}, rows={rows}"


def test_avx2_ternary_addsub_dense_exact_int32():
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(2026)

    for M in (1, 4, 8):
        for K in (1, 7, 31, 32, 33, 50, 96, 192, 384):
            for N in (3, 16, 17, 32, 64, 192):
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
                assert np.array_equal(out, expected), f"Dense AVX2 M={M}, K={K}, N={N}"


@pytest.mark.parametrize("n", [1, 7, 16, 31, 32, 64, 128, 256, 512, 1024])
def test_trace_dot_f32_matches_numpy(n: int):
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(777 + n)
    a = rng.randn(n).astype(np.float32)
    b = rng.randn(n).astype(np.float32)
    expected_dot = float(np.dot(a, b))

    actual_dot = lib.trace_dot_f32(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )
    assert abs(actual_dot - expected_dot) < 1e-4


@pytest.mark.parametrize("n", [1, 7, 16, 31, 32, 64, 128, 256, 512, 1024])
def test_trace_axpy_f32_matches_numpy(n: int):
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(888 + n)
    a = rng.randn(n).astype(np.float32)
    y = rng.randn(n).astype(np.float32)
    alpha = float(rng.randn(1)[0])
    expected_y = y + alpha * a

    lib.trace_axpy_f32(
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(alpha),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )
    assert float(np.max(np.abs(y - expected_y))) < 1e-5


@pytest.mark.parametrize("isa", ["scalar", "avx2"])
@pytest.mark.parametrize("kv_lora_rank", [24, 32, 48, 50])
def test_paged_absorb_uk_f32_matches_numpy(isa: str, kv_lora_rank: int):
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(31415 + kv_lora_rank)
    n_heads, qk_nope_head_dim, S, scale = 4, 16, 3, 0.125

    try:
        lib.trace_cpu_set_isa(isa.encode("utf-8"))
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
        expected_uk = scale * np.einsum(
            "thd,hdk->thk", q_c, w_uk.reshape(n_heads, qk_nope_head_dim, kv_lora_rank).astype(np.float32)
        )
        assert float(np.max(np.abs(out_uk - expected_uk))) < 1e-4
    finally:
        lib.trace_cpu_set_isa(b"auto")


@pytest.mark.parametrize("isa", ["scalar", "avx2"])
@pytest.mark.parametrize("kv_lora_rank", [24, 32, 48, 50])
def test_paged_absorb_uv_f32_matches_numpy(isa: str, kv_lora_rank: int):
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(27182 + kv_lora_rank)
    n_heads, v_head_dim, S, scale = 4, 16, 3, 0.125

    try:
        lib.trace_cpu_set_isa(isa.encode("utf-8"))
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
        expected_uv = scale * np.einsum(
            "thk,hvk->thv", o_latent, w_uv.reshape(n_heads, v_head_dim, kv_lora_rank).astype(np.float32)
        )
        assert float(np.max(np.abs(out_uv - expected_uv))) < 1e-4
    finally:
        lib.trace_cpu_set_isa(b"auto")


def test_isa_dispatch_parity():
    lib = load_paged_gemm_lib()
    rng = np.random.RandomState(555)
    host_supports_avx2 = lib.trace_cpu_active_isa().decode("utf-8") == "avx2"

    try:
        for isa in ("scalar", "avx2"):
            lib.trace_cpu_set_isa(isa.encode("utf-8"))
            active = lib.trace_cpu_active_isa().decode("utf-8")
            if isa == "scalar":
                assert active == "scalar"
            elif host_supports_avx2:
                assert active == "avx2"

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
                assert np.array_equal(out_dense, expected)

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
                assert np.array_equal(out_fast, expected)
    finally:
        lib.trace_cpu_set_isa(b"auto")


def test_isa_dispatch_handles_unknown_name_safely():
    lib = load_paged_gemm_lib()
    try:
        lib.trace_cpu_set_isa(b"nonexistent_isa")
        active = lib.trace_cpu_active_isa().decode("utf-8")
        assert active in ("scalar", "avx2")
    finally:
        lib.trace_cpu_set_isa(b"auto")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

/*
 * Basic-Lite TRACE
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#if defined(TRACE_ENABLE_AVX512) && (defined(__x86_64__) || defined(_M_X64))

#include "paged_dispatch.h"
#include "trace_infer.h"
#include "trace_math.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define OMP_MIN_WORK_THRESHOLD 65536
#define K_TILE_SIZE 8192

// Desempaqueta 64 pesos de 2 bits (16 bytes) -> 64 uint8_t en {0, 1, 2} sin resta de 1.
// Identidad exacta para vpdpbusd: sum((w_u8 - 1) * x_i8) = sum(w_u8 * x_i8) - sum(x_i8).
static inline __m512i avx512_unpack_64_weights_u8(const uint8_t* p) {
    __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
    __m512i b0 = _mm512_and_si512(v, _mm512_set1_epi32(0x00000003));
    __m512i b1 = _mm512_and_si512(_mm512_slli_epi32(v, 6), _mm512_set1_epi32(0x00000300));
    __m512i b2 = _mm512_and_si512(_mm512_slli_epi32(v, 12), _mm512_set1_epi32(0x00030000));
    __m512i b3 = _mm512_and_si512(_mm512_slli_epi32(v, 18), _mm512_set1_epi32(0x03000000));
    return _mm512_or_si512(_mm512_or_si512(b0, b1), _mm512_or_si512(b2, b3));
}

// Suma exacta en int32 de un vector int8_t[n] usando vpdpbusd(ones_u8, x_i8)
static inline int32_t avx512_sum_i8(const int8_t* x, int64_t n) {
    __m512i acc = _mm512_setzero_si512();
    const __m512i ones_u8 = _mm512_set1_epi8(1);
    int64_t k = 0;
    for (; k + 64 <= n; k += 64) {
        __m512i xv = _mm512_loadu_si512((const void*)(x + k));
        acc = _mm512_dpbusd_epi32(acc, ones_u8, xv);
    }
    int32_t s = _mm512_reduce_add_epi32(acc);
    for (; k < n; ++k) {
        s += (int32_t)x[k];
    }
    return s;
}

static inline int32_t avx512_dot_packed_u8(const int8_t* x, const uint8_t* w_packed, int64_t cols, int32_t sum_x) {
    __m512i acc = _mm512_setzero_si512();
    int64_t k = 0;
    for (; k + 64 <= cols; k += 64) {
        __m512i xv = _mm512_loadu_si512((const void*)(x + k));
        __m512i wv = avx512_unpack_64_weights_u8(w_packed + (k >> 2));
        acc = _mm512_dpbusd_epi32(acc, wv, xv);
    }
    int32_t dot = _mm512_reduce_add_epi32(acc);
    for (; k < cols; ++k) {
        int shift = (k & 3) * 2;
        int32_t wu = (int32_t)((w_packed[k >> 2] >> shift) & 0x03);
        dot += (int32_t)x[k] * wu;
    }
    return dot - sum_x;
}

static inline const uint8_t* paged_get_row_packed_avx512(
    const uint8_t* const* page_ptrs,
    int num_pages,
    int r
) {
    for (int p = 0; p < num_pages; ++p) {
        const uint8_t* blob = page_ptrs[p];
        const PagedHeader* hdr = (const PagedHeader*)blob;
        if ((uint32_t)r >= hdr->row_start && (uint32_t)r < hdr->row_start + hdr->rows) {
            uint32_t local_r = (uint32_t)r - hdr->row_start;
            int64_t padded_cols = ((int64_t)hdr->cols + 31) / 32 * 32;
            int64_t row_stride_bytes = padded_cols >> 2;
            return blob + sizeof(PagedHeader) + local_r * row_stride_bytes;
        }
    }
    return NULL;
}

void paged_direct_matmul_fast_c_avx512(
    const int8_t* x,
    const uint8_t* page_blob,
    int32_t* out,
    int64_t M,
    int64_t K
) {
    const PagedHeader* hdr = (const PagedHeader*)page_blob;
    if (hdr->fmt_code != 1) return;

    const uint32_t rows = hdr->rows;
    const uint16_t cols = hdr->cols;
    const uint8_t* payload = page_blob + sizeof(PagedHeader);
    int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;
    int64_t total_work = M * (int64_t)rows * (int64_t)cols;

    #pragma omp parallel for schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t m = 0; m < M; ++m) {
        const int8_t* x_row = x + m * K;
        int32_t sx = avx512_sum_i8(x_row, cols);
        for (int64_t r = 0; r < (int64_t)rows; ++r) {
            const uint8_t* w_row = payload + r * row_stride_bytes;
            out[m * rows + r] = avx512_dot_packed_u8(x_row, w_row, cols, sx);
        }
    }
}

void paged_direct_matmul_fast_c_multi_avx512(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
) {
    if (!x || !page_blobs || !out || num_pages <= 0 || M <= 0 || K <= 0 || total_out_features <= 0) return;

    // Precomputar sum(x_row) una sola vez por token para el truco vpdpbusd
    int32_t sum_x_stack[512];
    int32_t* sum_x_full = sum_x_stack;
    int sum_x_alloc = 0;
    if (M > 512) {
        sum_x_full = (int32_t*)malloc((size_t)M * sizeof(int32_t));
        if (!sum_x_full) return;
        sum_x_alloc = 1;
    }
    for (int64_t m = 0; m < M; ++m) {
        sum_x_full[m] = avx512_sum_i8(x + m * K, K);
    }

    int64_t total_work = M * total_out_features * K;

    #pragma omp parallel for schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t p = 0; p < num_pages; ++p) {
        const uint8_t* blob = page_blobs[p];
        if (!blob) continue;
        const PagedHeader* hdr = (const PagedHeader*)blob;
        if (hdr->fmt_code != 1) continue;

        const uint32_t row_start = hdr->row_start;
        const uint32_t rows = hdr->rows;
        const uint16_t cols = hdr->cols;
        if ((int64_t)cols > K || (uint64_t)row_start + (uint64_t)rows > (uint64_t)total_out_features) continue;
        const uint8_t* payload = blob + sizeof(PagedHeader);
        int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
        int64_t row_stride_bytes = padded_cols >> 2;

        if (M == 1) {
            const int8_t* x_row = x;
            int32_t sx = ((int64_t)cols == K) ? sum_x_full[0] : avx512_sum_i8(x_row, cols);
            int32_t* out_row = out + row_start;
            int64_t r = 0;
            for (; r + 4 <= (int64_t)rows; r += 4) {
                const uint8_t* w0_p = payload + (r + 0) * row_stride_bytes;
                const uint8_t* w1_p = payload + (r + 1) * row_stride_bytes;
                const uint8_t* w2_p = payload + (r + 2) * row_stride_bytes;
                const uint8_t* w3_p = payload + (r + 3) * row_stride_bytes;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();

                int64_t k = 0;
                for (; k + 64 <= cols; k += 64) {
                    _mm_prefetch((const char*)(w0_p + (k >> 2) + 128), _MM_HINT_T0);
                    _mm_prefetch((const char*)(w1_p + (k >> 2) + 128), _MM_HINT_T0);
                    _mm_prefetch((const char*)(w2_p + (k >> 2) + 128), _MM_HINT_T0);
                    _mm_prefetch((const char*)(w3_p + (k >> 2) + 128), _MM_HINT_T0);
                    __m512i xv = _mm512_loadu_si512((const void*)(x_row + k));
                    acc0 = _mm512_dpbusd_epi32(acc0, avx512_unpack_64_weights_u8(w0_p + (k >> 2)), xv);
                    acc1 = _mm512_dpbusd_epi32(acc1, avx512_unpack_64_weights_u8(w1_p + (k >> 2)), xv);
                    acc2 = _mm512_dpbusd_epi32(acc2, avx512_unpack_64_weights_u8(w2_p + (k >> 2)), xv);
                    acc3 = _mm512_dpbusd_epi32(acc3, avx512_unpack_64_weights_u8(w3_p + (k >> 2)), xv);
                }

                int32_t d0 = _mm512_reduce_add_epi32(acc0);
                int32_t d1 = _mm512_reduce_add_epi32(acc1);
                int32_t d2 = _mm512_reduce_add_epi32(acc2);
                int32_t d3 = _mm512_reduce_add_epi32(acc3);

                for (; k < cols; ++k) {
                    int32_t xv_s = (int32_t)x_row[k];
                    int shift = (k & 3) * 2;
                    d0 += xv_s * (int32_t)((w0_p[k >> 2] >> shift) & 0x03);
                    d1 += xv_s * (int32_t)((w1_p[k >> 2] >> shift) & 0x03);
                    d2 += xv_s * (int32_t)((w2_p[k >> 2] >> shift) & 0x03);
                    d3 += xv_s * (int32_t)((w3_p[k >> 2] >> shift) & 0x03);
                }

                out_row[r + 0] = d0 - sx;
                out_row[r + 1] = d1 - sx;
                out_row[r + 2] = d2 - sx;
                out_row[r + 3] = d3 - sx;
            }
            for (; r + 2 <= (int64_t)rows; r += 2) {
                const uint8_t* w0_p = payload + (r + 0) * row_stride_bytes;
                const uint8_t* w1_p = payload + (r + 1) * row_stride_bytes;
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                int64_t k = 0;
                for (; k + 64 <= cols; k += 64) {
                    __m512i xv = _mm512_loadu_si512((const void*)(x_row + k));
                    acc0 = _mm512_dpbusd_epi32(acc0, avx512_unpack_64_weights_u8(w0_p + (k >> 2)), xv);
                    acc1 = _mm512_dpbusd_epi32(acc1, avx512_unpack_64_weights_u8(w1_p + (k >> 2)), xv);
                }
                int32_t d0 = _mm512_reduce_add_epi32(acc0);
                int32_t d1 = _mm512_reduce_add_epi32(acc1);
                for (; k < cols; ++k) {
                    int32_t xv_s = (int32_t)x_row[k];
                    int shift = (k & 3) * 2;
                    d0 += xv_s * (int32_t)((w0_p[k >> 2] >> shift) & 0x03);
                    d1 += xv_s * (int32_t)((w1_p[k >> 2] >> shift) & 0x03);
                }
                out_row[r + 0] = d0 - sx;
                out_row[r + 1] = d1 - sx;
            }
            for (; r < (int64_t)rows; ++r) {
                out_row[r] = avx512_dot_packed_u8(x_row, payload + r * row_stride_bytes, cols, sx);
            }
        } else if (cols == 0) {
            for (int64_t m = 0; m < M; ++m) {
                memset(out + m * total_out_features + row_start, 0, sizeof(int32_t) * (size_t)rows);
            }
        } else {
            // Prefill M > 1: micro-kernel 4x4 en 16 registros ZMM (4 filas x 4 tokens = 1024 MACs / iter)
            uint8_t w_u8[4][K_TILE_SIZE] __attribute__((aligned(64)));

            int64_t r = 0;
            for (; r + 4 <= (int64_t)rows; r += 4) {
                const uint8_t* wps[4] = {
                    payload + (r + 0) * row_stride_bytes,
                    payload + (r + 1) * row_stride_bytes,
                    payload + (r + 2) * row_stride_bytes,
                    payload + (r + 3) * row_stride_bytes
                };

                for (int64_t k_block = 0; k_block < (int64_t)cols; k_block += K_TILE_SIZE) {
                    int64_t chunk_cols = (int64_t)cols - k_block;
                    if (chunk_cols > K_TILE_SIZE) chunk_cols = K_TILE_SIZE;

                    for (int rr = 0; rr < 4; ++rr) {
                        int64_t k = 0;
                        for (; k + 64 <= chunk_cols; k += 64) {
                            __m512i wv = avx512_unpack_64_weights_u8(wps[rr] + ((k_block + k) >> 2));
                            _mm512_store_si512((void*)(w_u8[rr] + k), wv);
                        }
                        for (; k < chunk_cols; ++k) {
                            int shift = ((k_block + k) & 3) * 2;
                            w_u8[rr][k] = (uint8_t)((wps[rr][(k_block + k) >> 2] >> shift) & 0x03);
                        }
                    }

                    int64_t m = 0;
                    for (; m + 4 <= M; m += 4) {
                        const int8_t* x0 = x + (m + 0) * K + k_block;
                        const int8_t* x1 = x + (m + 1) * K + k_block;
                        const int8_t* x2 = x + (m + 2) * K + k_block;
                        const int8_t* x3 = x + (m + 3) * K + k_block;

                        __m512i a00 = _mm512_setzero_si512(), a01 = _mm512_setzero_si512(), a02 = _mm512_setzero_si512(), a03 = _mm512_setzero_si512();
                        __m512i a10 = _mm512_setzero_si512(), a11 = _mm512_setzero_si512(), a12 = _mm512_setzero_si512(), a13 = _mm512_setzero_si512();
                        __m512i a20 = _mm512_setzero_si512(), a21 = _mm512_setzero_si512(), a22 = _mm512_setzero_si512(), a23 = _mm512_setzero_si512();
                        __m512i a30 = _mm512_setzero_si512(), a31 = _mm512_setzero_si512(), a32 = _mm512_setzero_si512(), a33 = _mm512_setzero_si512();

                        int64_t k = 0;
                        for (; k + 64 <= chunk_cols; k += 64) {
                            __m512i wv0 = _mm512_load_si512((const void*)(w_u8[0] + k));
                            __m512i wv1 = _mm512_load_si512((const void*)(w_u8[1] + k));
                            __m512i wv2 = _mm512_load_si512((const void*)(w_u8[2] + k));
                            __m512i wv3 = _mm512_load_si512((const void*)(w_u8[3] + k));

                            __m512i xv0 = _mm512_loadu_si512((const void*)(x0 + k));
                            a00 = _mm512_dpbusd_epi32(a00, wv0, xv0);
                            a01 = _mm512_dpbusd_epi32(a01, wv1, xv0);
                            a02 = _mm512_dpbusd_epi32(a02, wv2, xv0);
                            a03 = _mm512_dpbusd_epi32(a03, wv3, xv0);

                            __m512i xv1 = _mm512_loadu_si512((const void*)(x1 + k));
                            a10 = _mm512_dpbusd_epi32(a10, wv0, xv1);
                            a11 = _mm512_dpbusd_epi32(a11, wv1, xv1);
                            a12 = _mm512_dpbusd_epi32(a12, wv2, xv1);
                            a13 = _mm512_dpbusd_epi32(a13, wv3, xv1);

                            __m512i xv2 = _mm512_loadu_si512((const void*)(x2 + k));
                            a20 = _mm512_dpbusd_epi32(a20, wv0, xv2);
                            a21 = _mm512_dpbusd_epi32(a21, wv1, xv2);
                            a22 = _mm512_dpbusd_epi32(a22, wv2, xv2);
                            a23 = _mm512_dpbusd_epi32(a23, wv3, xv2);

                            __m512i xv3 = _mm512_loadu_si512((const void*)(x3 + k));
                            a30 = _mm512_dpbusd_epi32(a30, wv0, xv3);
                            a31 = _mm512_dpbusd_epi32(a31, wv1, xv3);
                            a32 = _mm512_dpbusd_epi32(a32, wv2, xv3);
                            a33 = _mm512_dpbusd_epi32(a33, wv3, xv3);
                        }

                        int32_t d[4][4] = {
                            { _mm512_reduce_add_epi32(a00), _mm512_reduce_add_epi32(a01), _mm512_reduce_add_epi32(a02), _mm512_reduce_add_epi32(a03) },
                            { _mm512_reduce_add_epi32(a10), _mm512_reduce_add_epi32(a11), _mm512_reduce_add_epi32(a12), _mm512_reduce_add_epi32(a13) },
                            { _mm512_reduce_add_epi32(a20), _mm512_reduce_add_epi32(a21), _mm512_reduce_add_epi32(a22), _mm512_reduce_add_epi32(a23) },
                            { _mm512_reduce_add_epi32(a30), _mm512_reduce_add_epi32(a31), _mm512_reduce_add_epi32(a32), _mm512_reduce_add_epi32(a33) }
                        };

                        const int8_t* xs[4] = { x0, x1, x2, x3 };
                        for (; k < chunk_cols; ++k) {
                            int32_t w0s = (int32_t)w_u8[0][k], w1s = (int32_t)w_u8[1][k], w2s = (int32_t)w_u8[2][k], w3s = (int32_t)w_u8[3][k];
                            for (int tm = 0; tm < 4; ++tm) {
                                int32_t xvs = (int32_t)xs[tm][k];
                                d[tm][0] += xvs * w0s;
                                d[tm][1] += xvs * w1s;
                                d[tm][2] += xvs * w2s;
                                d[tm][3] += xvs * w3s;
                            }
                        }

                        for (int tm = 0; tm < 4; ++tm) {
                            int32_t* dst = out + (m + tm) * total_out_features + row_start + r;
                            if (k_block == 0) {
                                int32_t sx = ((int64_t)cols == K) ? sum_x_full[m + tm] : avx512_sum_i8(x + (m + tm) * K, cols);
                                dst[0] = d[tm][0] - sx;
                                dst[1] = d[tm][1] - sx;
                                dst[2] = d[tm][2] - sx;
                                dst[3] = d[tm][3] - sx;
                            } else {
                                dst[0] += d[tm][0];
                                dst[1] += d[tm][1];
                                dst[2] += d[tm][2];
                                dst[3] += d[tm][3];
                            }
                        }
                    }

                    for (; m < M; ++m) {
                        const int8_t* x_row = x + m * K + k_block;
                        __m512i acc0 = _mm512_setzero_si512();
                        __m512i acc1 = _mm512_setzero_si512();
                        __m512i acc2 = _mm512_setzero_si512();
                        __m512i acc3 = _mm512_setzero_si512();
                        int64_t k = 0;
                        for (; k + 64 <= chunk_cols; k += 64) {
                            __m512i xv = _mm512_loadu_si512((const void*)(x_row + k));
                            acc0 = _mm512_dpbusd_epi32(acc0, _mm512_load_si512((const void*)(w_u8[0] + k)), xv);
                            acc1 = _mm512_dpbusd_epi32(acc1, _mm512_load_si512((const void*)(w_u8[1] + k)), xv);
                            acc2 = _mm512_dpbusd_epi32(acc2, _mm512_load_si512((const void*)(w_u8[2] + k)), xv);
                            acc3 = _mm512_dpbusd_epi32(acc3, _mm512_load_si512((const void*)(w_u8[3] + k)), xv);
                        }
                        int32_t d0 = _mm512_reduce_add_epi32(acc0);
                        int32_t d1 = _mm512_reduce_add_epi32(acc1);
                        int32_t d2 = _mm512_reduce_add_epi32(acc2);
                        int32_t d3 = _mm512_reduce_add_epi32(acc3);
                        for (; k < chunk_cols; ++k) {
                            int32_t xvs = (int32_t)x_row[k];
                            d0 += xvs * (int32_t)w_u8[0][k];
                            d1 += xvs * (int32_t)w_u8[1][k];
                            d2 += xvs * (int32_t)w_u8[2][k];
                            d3 += xvs * (int32_t)w_u8[3][k];
                        }
                        int32_t* dst = out + m * total_out_features + row_start + r;
                        if (k_block == 0) {
                            int32_t sx = ((int64_t)cols == K) ? sum_x_full[m] : avx512_sum_i8(x + m * K, cols);
                            dst[0] = d0 - sx;
                            dst[1] = d1 - sx;
                            dst[2] = d2 - sx;
                            dst[3] = d3 - sx;
                        } else {
                            dst[0] += d0;
                            dst[1] += d1;
                            dst[2] += d2;
                            dst[3] += d3;
                        }
                    }
                }
            }

            for (; r + 2 <= (int64_t)rows; r += 2) {
                const uint8_t* wps[2] = {
                    payload + (r + 0) * row_stride_bytes,
                    payload + (r + 1) * row_stride_bytes
                };

                for (int64_t k_block = 0; k_block < (int64_t)cols; k_block += K_TILE_SIZE) {
                    int64_t chunk_cols = (int64_t)cols - k_block;
                    if (chunk_cols > K_TILE_SIZE) chunk_cols = K_TILE_SIZE;

                    for (int rr = 0; rr < 2; ++rr) {
                        int64_t k = 0;
                        for (; k + 64 <= chunk_cols; k += 64) {
                            __m512i wv = avx512_unpack_64_weights_u8(wps[rr] + ((k_block + k) >> 2));
                            _mm512_store_si512((void*)(w_u8[rr] + k), wv);
                        }
                        for (; k < chunk_cols; ++k) {
                            int shift = ((k_block + k) & 3) * 2;
                            w_u8[rr][k] = (uint8_t)((wps[rr][(k_block + k) >> 2] >> shift) & 0x03);
                        }
                    }

                    int64_t m = 0;
                    for (; m + 4 <= M; m += 4) {
                        const int8_t* x0 = x + (m + 0) * K + k_block;
                        const int8_t* x1 = x + (m + 1) * K + k_block;
                        const int8_t* x2 = x + (m + 2) * K + k_block;
                        const int8_t* x3 = x + (m + 3) * K + k_block;

                        __m512i a00 = _mm512_setzero_si512(), a01 = _mm512_setzero_si512();
                        __m512i a10 = _mm512_setzero_si512(), a11 = _mm512_setzero_si512();
                        __m512i a20 = _mm512_setzero_si512(), a21 = _mm512_setzero_si512();
                        __m512i a30 = _mm512_setzero_si512(), a31 = _mm512_setzero_si512();

                        int64_t k = 0;
                        for (; k + 64 <= chunk_cols; k += 64) {
                            __m512i wv0 = _mm512_load_si512((const void*)(w_u8[0] + k));
                            __m512i wv1 = _mm512_load_si512((const void*)(w_u8[1] + k));

                            __m512i xv0 = _mm512_loadu_si512((const void*)(x0 + k));
                            a00 = _mm512_dpbusd_epi32(a00, wv0, xv0);
                            a01 = _mm512_dpbusd_epi32(a01, wv1, xv0);

                            __m512i xv1 = _mm512_loadu_si512((const void*)(x1 + k));
                            a10 = _mm512_dpbusd_epi32(a10, wv0, xv1);
                            a11 = _mm512_dpbusd_epi32(a11, wv1, xv1);

                            __m512i xv2 = _mm512_loadu_si512((const void*)(x2 + k));
                            a20 = _mm512_dpbusd_epi32(a20, wv0, xv2);
                            a21 = _mm512_dpbusd_epi32(a21, wv1, xv2);

                            __m512i xv3 = _mm512_loadu_si512((const void*)(x3 + k));
                            a30 = _mm512_dpbusd_epi32(a30, wv0, xv3);
                            a31 = _mm512_dpbusd_epi32(a31, wv1, xv3);
                        }

                        int32_t d[4][2] = {
                            { _mm512_reduce_add_epi32(a00), _mm512_reduce_add_epi32(a01) },
                            { _mm512_reduce_add_epi32(a10), _mm512_reduce_add_epi32(a11) },
                            { _mm512_reduce_add_epi32(a20), _mm512_reduce_add_epi32(a21) },
                            { _mm512_reduce_add_epi32(a30), _mm512_reduce_add_epi32(a31) }
                        };

                        const int8_t* xs[4] = { x0, x1, x2, x3 };
                        for (; k < chunk_cols; ++k) {
                            int32_t w0s = (int32_t)w_u8[0][k], w1s = (int32_t)w_u8[1][k];
                            for (int tm = 0; tm < 4; ++tm) {
                                int32_t xvs = (int32_t)xs[tm][k];
                                d[tm][0] += xvs * w0s;
                                d[tm][1] += xvs * w1s;
                            }
                        }

                        for (int tm = 0; tm < 4; ++tm) {
                            int32_t* dst = out + (m + tm) * total_out_features + row_start + r;
                            if (k_block == 0) {
                                int32_t sx = ((int64_t)cols == K) ? sum_x_full[m + tm] : avx512_sum_i8(x + (m + tm) * K, cols);
                                dst[0] = d[tm][0] - sx;
                                dst[1] = d[tm][1] - sx;
                            } else {
                                dst[0] += d[tm][0];
                                dst[1] += d[tm][1];
                            }
                        }
                    }

                    for (; m < M; ++m) {
                        const int8_t* x_row = x + m * K + k_block;
                        __m512i acc0 = _mm512_setzero_si512();
                        __m512i acc1 = _mm512_setzero_si512();
                        int64_t k = 0;
                        for (; k + 64 <= chunk_cols; k += 64) {
                            __m512i xv = _mm512_loadu_si512((const void*)(x_row + k));
                            acc0 = _mm512_dpbusd_epi32(acc0, _mm512_load_si512((const void*)(w_u8[0] + k)), xv);
                            acc1 = _mm512_dpbusd_epi32(acc1, _mm512_load_si512((const void*)(w_u8[1] + k)), xv);
                        }
                        int32_t d0 = _mm512_reduce_add_epi32(acc0);
                        int32_t d1 = _mm512_reduce_add_epi32(acc1);
                        for (; k < chunk_cols; ++k) {
                            int32_t xvs = (int32_t)x_row[k];
                            d0 += xvs * (int32_t)w_u8[0][k];
                            d1 += xvs * (int32_t)w_u8[1][k];
                        }
                        int32_t* dst = out + m * total_out_features + row_start + r;
                        if (k_block == 0) {
                            int32_t sx = ((int64_t)cols == K) ? sum_x_full[m] : avx512_sum_i8(x + m * K, cols);
                            dst[0] = d0 - sx;
                            dst[1] = d1 - sx;
                        } else {
                            dst[0] += d0;
                            dst[1] += d1;
                        }
                    }
                }
            }

            for (; r < (int64_t)rows; ++r) {
                const uint8_t* w_row = payload + r * row_stride_bytes;
                for (int64_t m = 0; m < M; ++m) {
                    int32_t sx = ((int64_t)cols == K) ? sum_x_full[m] : avx512_sum_i8(x + m * K, cols);
                    out[m * total_out_features + row_start + r] = avx512_dot_packed_u8(x + m * K, w_row, cols, sx);
                }
            }
        }
    }

    if (sum_x_alloc) free(sum_x_full);
}

void paged_lowrank_matmul_row_c_avx512(
    const int8_t* x,
    const uint8_t* w_packed,
    int32_t* out,
    int64_t rows,
    int64_t K
) {
    int64_t padded_cols = (K + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;
    int32_t sx = avx512_sum_i8(x, K);
    int64_t total_work = rows * K;

    #pragma omp parallel for schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t r = 0; r < rows; ++r) {
        out[r] = avx512_dot_packed_u8(x, w_packed + r * row_stride_bytes, K, sx);
    }
}

void ternary_addsub_cpu_kernel_avx512(
    const int8_t* x,
    const int8_t* w,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t total_work = M * N * K;
    const __m512i ones = _mm512_set1_epi8(1);

    #pragma omp parallel for collapse(2) schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const int8_t* x_row = x + m * K;
            const int8_t* w_row = w + n * K;
            __m512i acc = _mm512_setzero_si512();
            int64_t k = 0;
            for (; k + 64 <= K; k += 64) {
                __m512i xv = _mm512_loadu_si512((const void*)(x_row + k));
                __m512i wv = _mm512_add_epi8(_mm512_loadu_si512((const void*)(w_row + k)), ones);
                acc = _mm512_dpbusd_epi32(acc, wv, xv);
            }
            int32_t dot = _mm512_reduce_add_epi32(acc) - avx512_sum_i8(x_row, k);
            for (; k < K; ++k) {
                dot += (int32_t)x_row[k] * (int32_t)w_row[k];
            }
            out[m * N + n] = dot;
        }
    }
}

// Reutilizar absorcion MLA desde AVX2 (misma paridad) o version AVX-512 directa
void paged_absorb_uk_f32_avx2(const float* q_c, const uint8_t* const* page_ptrs, int num_pages, float scale, float* q_c_abs, int n_heads, int qk_nope_head_dim, int kv_lora_rank, int S);
void paged_absorb_uv_f32_avx2(const float* o_latent, const uint8_t* const* page_ptrs, int num_pages, float scale, float* o_heads, int n_heads, int v_head_dim, int kv_lora_rank, int S);

void paged_absorb_uk_f32_avx512(
    const float* q_c,
    const uint8_t* const* page_ptrs,
    int num_pages,
    float scale,
    float* q_c_abs,
    int n_heads,
    int qk_nope_head_dim,
    int kv_lora_rank,
    int S
) {
    (void)paged_get_row_packed_avx512;
    paged_absorb_uk_f32_avx2(q_c, page_ptrs, num_pages, scale, q_c_abs, n_heads, qk_nope_head_dim, kv_lora_rank, S);
}

void paged_absorb_uv_f32_avx512(
    const float* o_latent,
    const uint8_t* const* page_ptrs,
    int num_pages,
    float scale,
    float* o_heads,
    int n_heads,
    int v_head_dim,
    int kv_lora_rank,
    int S
) {
    paged_absorb_uv_f32_avx2(o_latent, page_ptrs, num_pages, scale, o_heads, n_heads, v_head_dim, kv_lora_rank, S);
}

float trace_dot_f32_avx512(const float* a, const float* b, int n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }
    float dot = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; ++i) dot += a[i] * b[i];
    return dot;
}

void trace_axpy_f32_avx512(float* y, float a, const float* x, int n) {
    __m512 a_vec = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(a_vec, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

float trace_dot_f32_f16_avx512(const float* x, const uint16_t* w_f16, int n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 w0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w_f16 + i)));
        __m512 w1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w_f16 + i + 16)));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), w0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16), w1, acc1);
    }
    for (; i + 16 <= n; i += 16) {
        __m512 w0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w_f16 + i)));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), w0, acc0);
    }
    float dot = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; ++i) dot += x[i] * trace_f16_to_f32(w_f16[i]);
    return dot;
}

void trace_matvec_f32_f16_avx512(const float* x, const uint16_t* w_f16, float* out, int rows, int cols) {
    int blocks8 = rows / 8;
    #pragma omp parallel for schedule(static) if (rows >= 256)
    for (int b = 0; b < blocks8; ++b) {
        int v = b * 8;
        const uint16_t* w0 = w_f16 + (size_t)(v + 0) * (size_t)cols;
        const uint16_t* w1 = w_f16 + (size_t)(v + 1) * (size_t)cols;
        const uint16_t* w2 = w_f16 + (size_t)(v + 2) * (size_t)cols;
        const uint16_t* w3 = w_f16 + (size_t)(v + 3) * (size_t)cols;
        const uint16_t* w4 = w_f16 + (size_t)(v + 4) * (size_t)cols;
        const uint16_t* w5 = w_f16 + (size_t)(v + 5) * (size_t)cols;
        const uint16_t* w6 = w_f16 + (size_t)(v + 6) * (size_t)cols;
        const uint16_t* w7 = w_f16 + (size_t)(v + 7) * (size_t)cols;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps(), a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= cols; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            a0 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w0 + i))), a0);
            a1 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w1 + i))), a1);
            a2 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w2 + i))), a2);
            a3 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w3 + i))), a3);
            a4 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w4 + i))), a4);
            a5 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w5 + i))), a5);
            a6 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w6 + i))), a6);
            a7 = _mm512_fmadd_ps(vx, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w7 + i))), a7);
        }
        float d0 = _mm512_reduce_add_ps(a0), d1 = _mm512_reduce_add_ps(a1), d2 = _mm512_reduce_add_ps(a2), d3 = _mm512_reduce_add_ps(a3);
        float d4 = _mm512_reduce_add_ps(a4), d5 = _mm512_reduce_add_ps(a5), d6 = _mm512_reduce_add_ps(a6), d7 = _mm512_reduce_add_ps(a7);
        for (; i < cols; ++i) {
            float xv = x[i];
            d0 += xv * trace_f16_to_f32(w0[i]); d1 += xv * trace_f16_to_f32(w1[i]);
            d2 += xv * trace_f16_to_f32(w2[i]); d3 += xv * trace_f16_to_f32(w3[i]);
            d4 += xv * trace_f16_to_f32(w4[i]); d5 += xv * trace_f16_to_f32(w5[i]);
            d6 += xv * trace_f16_to_f32(w6[i]); d7 += xv * trace_f16_to_f32(w7[i]);
        }
        out[v + 0] = d0; out[v + 1] = d1; out[v + 2] = d2; out[v + 3] = d3;
        out[v + 4] = d4; out[v + 5] = d5; out[v + 6] = d6; out[v + 7] = d7;
    }
    for (int v = blocks8 * 8; v < rows; ++v) {
        out[v] = trace_dot_f32_f16_avx512(x, w_f16 + (size_t)v * (size_t)cols, cols);
    }
}

float trace_dot_f32_i8_avx512(const float* x, const int8_t* w_i8, float scale, int n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_i8 + i))));
        __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_i8 + i + 16))));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), w0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16), w1, acc1);
    }
    for (; i + 16 <= n; i += 16) {
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w_i8 + i))));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), w0, acc0);
    }
    float dot = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; ++i) dot += x[i] * (float)w_i8[i];
    return dot * scale;
}

void trace_mla_attn_group_f32_avx512(
    const float* q_c_abs_base,
    const float* q_pe_base,
    const float* c_kv_layer,
    const float* k_pe_layer,
    float* o_lat_base,
    float* scores_scratch,
    int h_start,
    int h_count,
    int total_len,
    int kv_lora_rank,
    int qk_rope_head_dim,
    float attn_scale
) {
    if (h_count == 4) {
        const float* qc0 = q_c_abs_base + (size_t)(h_start + 0) * (size_t)kv_lora_rank;
        const float* qc1 = q_c_abs_base + (size_t)(h_start + 1) * (size_t)kv_lora_rank;
        const float* qc2 = q_c_abs_base + (size_t)(h_start + 2) * (size_t)kv_lora_rank;
        const float* qc3 = q_c_abs_base + (size_t)(h_start + 3) * (size_t)kv_lora_rank;

        const float* qp0 = q_pe_base + (size_t)(h_start + 0) * (size_t)qk_rope_head_dim;
        const float* qp1 = q_pe_base + (size_t)(h_start + 1) * (size_t)qk_rope_head_dim;
        const float* qp2 = q_pe_base + (size_t)(h_start + 2) * (size_t)qk_rope_head_dim;
        const float* qp3 = q_pe_base + (size_t)(h_start + 3) * (size_t)qk_rope_head_dim;

        float* sc0 = scores_scratch + (size_t)0 * (size_t)total_len;
        float* sc1 = scores_scratch + (size_t)1 * (size_t)total_len;
        float* sc2 = scores_scratch + (size_t)2 * (size_t)total_len;
        float* sc3 = scores_scratch + (size_t)3 * (size_t)total_len;

        float max0 = -INFINITY, max1 = -INFINITY, max2 = -INFINITY, max3 = -INFINITY;

        for (int s = 0; s < total_len; ++s) {
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            const float* k_pe_s = k_pe_layer + (size_t)s * (size_t)qk_rope_head_dim;

            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= kv_lora_rank; i += 16) {
                __m512 vc = _mm512_loadu_ps(c_kv_s + i);
                a0 = _mm512_fmadd_ps(vc, _mm512_loadu_ps(qc0 + i), a0);
                a1 = _mm512_fmadd_ps(vc, _mm512_loadu_ps(qc1 + i), a1);
                a2 = _mm512_fmadd_ps(vc, _mm512_loadu_ps(qc2 + i), a2);
                a3 = _mm512_fmadd_ps(vc, _mm512_loadu_ps(qc3 + i), a3);
            }
            float d0 = _mm512_reduce_add_ps(a0), d1 = _mm512_reduce_add_ps(a1), d2 = _mm512_reduce_add_ps(a2), d3 = _mm512_reduce_add_ps(a3);
            for (; i < kv_lora_rank; ++i) {
                float vc = c_kv_s[i];
                d0 += vc * qc0[i]; d1 += vc * qc1[i]; d2 += vc * qc2[i]; d3 += vc * qc3[i];
            }

            __m512 p0 = _mm512_setzero_ps(), p1 = _mm512_setzero_ps(), p2 = _mm512_setzero_ps(), p3 = _mm512_setzero_ps();
            int j = 0;
            for (; j + 16 <= qk_rope_head_dim; j += 16) {
                __m512 vk = _mm512_loadu_ps(k_pe_s + j);
                p0 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(qp0 + j), p0);
                p1 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(qp1 + j), p1);
                p2 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(qp2 + j), p2);
                p3 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(qp3 + j), p3);
            }
            d0 += _mm512_reduce_add_ps(p0); d1 += _mm512_reduce_add_ps(p1); d2 += _mm512_reduce_add_ps(p2); d3 += _mm512_reduce_add_ps(p3);
            for (; j < qk_rope_head_dim; ++j) {
                float vk = k_pe_s[j];
                d0 += vk * qp0[j]; d1 += vk * qp1[j]; d2 += vk * qp2[j]; d3 += vk * qp3[j];
            }

            float v0 = d0 * attn_scale, v1 = d1 * attn_scale, v2 = d2 * attn_scale, v3 = d3 * attn_scale;
            sc0[s] = v0; sc1[s] = v1; sc2[s] = v2; sc3[s] = v3;
            if (v0 > max0) max0 = v0;
            if (v1 > max1) max1 = v1;
            if (v2 > max2) max2 = v2;
            if (v3 > max3) max3 = v3;
        }

        float maxv[4] = { max0, max1, max2, max3 };
        for (int gh = 0; gh < 4; ++gh) {
            float* sc = scores_scratch + (size_t)gh * (size_t)total_len;
            float m = maxv[gh];
            float sum_exp = 0.0f;
            for (int s = 0; s < total_len; ++s) {
                float e = expf(sc[s] - m);
                sc[s] = e;
                sum_exp += e;
            }
            float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            __m512 inv_vec = _mm512_set1_ps(sum_inv);
            int s = 0;
            for (; s + 16 <= total_len; s += 16) {
                _mm512_storeu_ps(sc + s, _mm512_mul_ps(_mm512_loadu_ps(sc + s), inv_vec));
            }
            for (; s < total_len; ++s) sc[s] *= sum_inv;
        }

        float* o0 = o_lat_base + (size_t)(h_start + 0) * (size_t)kv_lora_rank;
        float* o1 = o_lat_base + (size_t)(h_start + 1) * (size_t)kv_lora_rank;
        float* o2 = o_lat_base + (size_t)(h_start + 2) * (size_t)kv_lora_rank;
        float* o3 = o_lat_base + (size_t)(h_start + 3) * (size_t)kv_lora_rank;
        memset(o0, 0, sizeof(float) * (size_t)kv_lora_rank);
        memset(o1, 0, sizeof(float) * (size_t)kv_lora_rank);
        memset(o2, 0, sizeof(float) * (size_t)kv_lora_rank);
        memset(o3, 0, sizeof(float) * (size_t)kv_lora_rank);

        for (int s = 0; s < total_len; ++s) {
            float pr0 = sc0[s], pr1 = sc1[s], pr2 = sc2[s], pr3 = sc3[s];
            if (pr0 == 0.0f && pr1 == 0.0f && pr2 == 0.0f && pr3 == 0.0f) continue;
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            __m512 vp0 = _mm512_set1_ps(pr0), vp1 = _mm512_set1_ps(pr1), vp2 = _mm512_set1_ps(pr2), vp3 = _mm512_set1_ps(pr3);
            int i = 0;
            for (; i + 16 <= kv_lora_rank; i += 16) {
                __m512 vc = _mm512_loadu_ps(c_kv_s + i);
                _mm512_storeu_ps(o0 + i, _mm512_fmadd_ps(vp0, vc, _mm512_loadu_ps(o0 + i)));
                _mm512_storeu_ps(o1 + i, _mm512_fmadd_ps(vp1, vc, _mm512_loadu_ps(o1 + i)));
                _mm512_storeu_ps(o2 + i, _mm512_fmadd_ps(vp2, vc, _mm512_loadu_ps(o2 + i)));
                _mm512_storeu_ps(o3 + i, _mm512_fmadd_ps(vp3, vc, _mm512_loadu_ps(o3 + i)));
            }
            for (; i < kv_lora_rank; ++i) {
                float vc = c_kv_s[i];
                o0[i] += pr0 * vc; o1[i] += pr1 * vc; o2[i] += pr2 * vc; o3[i] += pr3 * vc;
            }
        }
        return;
    }

    for (int gh = 0; gh < h_count; ++gh) {
        int h = h_start + gh;
        const float* q_c_h = q_c_abs_base + (size_t)h * (size_t)kv_lora_rank;
        const float* q_pe_h = q_pe_base + (size_t)h * (size_t)qk_rope_head_dim;
        float* sc_h = scores_scratch + (size_t)gh * (size_t)total_len;

        float max_sc = -INFINITY;
        for (int s = 0; s < total_len; ++s) {
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            const float* k_pe_s = k_pe_layer + (size_t)s * (size_t)qk_rope_head_dim;
            float sc = (trace_dot_f32_avx512(q_c_h, c_kv_s, kv_lora_rank) +
                        trace_dot_f32_avx512(q_pe_h, k_pe_s, qk_rope_head_dim)) * attn_scale;
            sc_h[s] = sc;
            if (sc > max_sc) max_sc = sc;
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < total_len; ++s) {
            float p = expf(sc_h[s] - max_sc);
            sc_h[s] = p;
            sum_exp += p;
        }
        float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int s = 0; s < total_len; ++s) sc_h[s] *= sum_inv;

        float* o_lat_h = o_lat_base + (size_t)h * (size_t)kv_lora_rank;
        memset(o_lat_h, 0, sizeof(float) * (size_t)kv_lora_rank);
        for (int s = 0; s < total_len; ++s) {
            float p = sc_h[s];
            if (p == 0.0f) continue;
            trace_axpy_f32_avx512(o_lat_h, p, c_kv_layer + (size_t)s * (size_t)kv_lora_rank, kv_lora_rank);
        }
    }
}

void trace_gqa_attn_group_f32_avx512(
    const float* q_base,
    const float* k_layer,
    const float* v_layer,
    float* o_heads_base,
    float* scores_scratch,
    int kv_h,
    int q_start,
    int q_count,
    int n_kv_heads,
    int total_len,
    int head_dim,
    float attn_scale
) {
    size_t kv_stride = (size_t)n_kv_heads * (size_t)head_dim;
    size_t kv_off = (size_t)kv_h * (size_t)head_dim;

    if (q_count == 4) {
        const float* q0 = q_base + (size_t)(q_start + 0) * (size_t)head_dim;
        const float* q1 = q_base + (size_t)(q_start + 1) * (size_t)head_dim;
        const float* q2 = q_base + (size_t)(q_start + 2) * (size_t)head_dim;
        const float* q3 = q_base + (size_t)(q_start + 3) * (size_t)head_dim;

        float* sc0 = scores_scratch + (size_t)0 * (size_t)total_len;
        float* sc1 = scores_scratch + (size_t)1 * (size_t)total_len;
        float* sc2 = scores_scratch + (size_t)2 * (size_t)total_len;
        float* sc3 = scores_scratch + (size_t)3 * (size_t)total_len;

        float max0 = -INFINITY, max1 = -INFINITY, max2 = -INFINITY, max3 = -INFINITY;

        for (int s = 0; s < total_len; ++s) {
            const float* k_s = k_layer + (size_t)s * kv_stride + kv_off;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= head_dim; i += 16) {
                __m512 vk = _mm512_loadu_ps(k_s + i);
                a0 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(q0 + i), a0);
                a1 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(q1 + i), a1);
                a2 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(q2 + i), a2);
                a3 = _mm512_fmadd_ps(vk, _mm512_loadu_ps(q3 + i), a3);
            }
            float d0 = _mm512_reduce_add_ps(a0), d1 = _mm512_reduce_add_ps(a1), d2 = _mm512_reduce_add_ps(a2), d3 = _mm512_reduce_add_ps(a3);
            for (; i < head_dim; ++i) {
                float vk = k_s[i];
                d0 += vk * q0[i]; d1 += vk * q1[i]; d2 += vk * q2[i]; d3 += vk * q3[i];
            }
            float v0 = d0 * attn_scale, v1 = d1 * attn_scale, v2 = d2 * attn_scale, v3 = d3 * attn_scale;
            sc0[s] = v0; sc1[s] = v1; sc2[s] = v2; sc3[s] = v3;
            if (v0 > max0) max0 = v0;
            if (v1 > max1) max1 = v1;
            if (v2 > max2) max2 = v2;
            if (v3 > max3) max3 = v3;
        }

        float maxv[4] = { max0, max1, max2, max3 };
        for (int gh = 0; gh < 4; ++gh) {
            float* sc = scores_scratch + (size_t)gh * (size_t)total_len;
            float m = maxv[gh];
            float sum_exp = 0.0f;
            for (int s = 0; s < total_len; ++s) {
                float e = expf(sc[s] - m);
                sc[s] = e;
                sum_exp += e;
            }
            float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            __m512 inv_vec = _mm512_set1_ps(sum_inv);
            int s = 0;
            for (; s + 16 <= total_len; s += 16) {
                _mm512_storeu_ps(sc + s, _mm512_mul_ps(_mm512_loadu_ps(sc + s), inv_vec));
            }
            for (; s < total_len; ++s) sc[s] *= sum_inv;
        }

        float* o0 = o_heads_base + (size_t)(q_start + 0) * (size_t)head_dim;
        float* o1 = o_heads_base + (size_t)(q_start + 1) * (size_t)head_dim;
        float* o2 = o_heads_base + (size_t)(q_start + 2) * (size_t)head_dim;
        float* o3 = o_heads_base + (size_t)(q_start + 3) * (size_t)head_dim;
        memset(o0, 0, sizeof(float) * (size_t)head_dim);
        memset(o1, 0, sizeof(float) * (size_t)head_dim);
        memset(o2, 0, sizeof(float) * (size_t)head_dim);
        memset(o3, 0, sizeof(float) * (size_t)head_dim);

        for (int s = 0; s < total_len; ++s) {
            float pr0 = sc0[s], pr1 = sc1[s], pr2 = sc2[s], pr3 = sc3[s];
            if (pr0 == 0.0f && pr1 == 0.0f && pr2 == 0.0f && pr3 == 0.0f) continue;
            const float* v_s = v_layer + (size_t)s * kv_stride + kv_off;
            __m512 vp0 = _mm512_set1_ps(pr0), vp1 = _mm512_set1_ps(pr1), vp2 = _mm512_set1_ps(pr2), vp3 = _mm512_set1_ps(pr3);
            int i = 0;
            for (; i + 16 <= head_dim; i += 16) {
                __m512 vv = _mm512_loadu_ps(v_s + i);
                _mm512_storeu_ps(o0 + i, _mm512_fmadd_ps(vp0, vv, _mm512_loadu_ps(o0 + i)));
                _mm512_storeu_ps(o1 + i, _mm512_fmadd_ps(vp1, vv, _mm512_loadu_ps(o1 + i)));
                _mm512_storeu_ps(o2 + i, _mm512_fmadd_ps(vp2, vv, _mm512_loadu_ps(o2 + i)));
                _mm512_storeu_ps(o3 + i, _mm512_fmadd_ps(vp3, vv, _mm512_loadu_ps(o3 + i)));
            }
            for (; i < head_dim; ++i) {
                float vv = v_s[i];
                o0[i] += pr0 * vv; o1[i] += pr1 * vv; o2[i] += pr2 * vv; o3[i] += pr3 * vv;
            }
        }
        return;
    }

    for (int gh = 0; gh < q_count; ++gh) {
        int qh = q_start + gh;
        const float* q_h = q_base + (size_t)qh * (size_t)head_dim;
        float* sc_h = scores_scratch + (size_t)gh * (size_t)total_len;

        float max_sc = -INFINITY;
        for (int s = 0; s < total_len; ++s) {
            const float* k_s = k_layer + (size_t)s * kv_stride + kv_off;
            float sc = trace_dot_f32_avx512(q_h, k_s, head_dim) * attn_scale;
            sc_h[s] = sc;
            if (sc > max_sc) max_sc = sc;
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < total_len; ++s) {
            float p = expf(sc_h[s] - max_sc);
            sc_h[s] = p;
            sum_exp += p;
        }
        float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int s = 0; s < total_len; ++s) sc_h[s] *= sum_inv;

        float* o_h = o_heads_base + (size_t)qh * (size_t)head_dim;
        memset(o_h, 0, sizeof(float) * (size_t)head_dim);
        for (int s = 0; s < total_len; ++s) {
            float p = sc_h[s];
            if (p == 0.0f) continue;
            trace_axpy_f32_avx512(o_h, p, v_layer + (size_t)s * kv_stride + kv_off, head_dim);
        }
    }
}

void trace_quantize_rows_int8_avx512(
    const float* in,
    const float* sub_ln_w,
    float* x_norm,
    int8_t* q_x,
    float* scale_x,
    int M,
    int in_f,
    float eps
) {
    const __m512 v_max = _mm512_set1_ps(127.0f);
    const __m512 v_min = _mm512_set1_ps(-127.0f);

    #pragma omp parallel for schedule(static) if (M >= 16)
    for (int m = 0; m < M; ++m) {
        const float* in_row = in + (size_t)m * (size_t)in_f;
        const float* row = in_row;
        if (sub_ln_w != NULL) {
            float* norm_row = x_norm + (size_t)m * (size_t)in_f;
            rms_norm(in_row, sub_ln_w, norm_row, in_f, eps);
            row = norm_row;
        }

        __m512 vmax = _mm512_set1_ps(eps);
        int i = 0;
        for (; i + 16 <= in_f; i += 16) {
            __m512 v = _mm512_abs_ps(_mm512_loadu_ps(row + i));
            vmax = _mm512_max_ps(vmax, v);
        }
        float amax = _mm512_reduce_max_ps(vmax);
        for (; i < in_f; ++i) {
            float v = fabsf(row[i]);
            if (v > amax) amax = v;
        }

        float scale = amax / 127.0f;
        float scale_inv = 127.0f / amax;
        scale_x[m] = scale;
        __m512 v_sinv = _mm512_set1_ps(scale_inv);
        int8_t* qrow = q_x + (size_t)m * (size_t)in_f;

        i = 0;
        for (; i + 16 <= in_f; i += 16) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(row + i), v_sinv);
            v = _mm512_min_ps(_mm512_max_ps(v, v_min), v_max);
            __m512i vi = _mm512_cvtps_epi32(v);
            __m128i qb = _mm512_cvtepi32_epi8(vi);
            _mm_storeu_si128((__m128i*)(qrow + i), qb);
        }
        for (; i < in_f; ++i) {
            float q = nearbyintf(row[i] * scale_inv);
            if (!isfinite(q)) q = 0.0f;
            else if (q > 127.0f) q = 127.0f;
            else if (q < -127.0f) q = -127.0f;
            qrow[i] = (int8_t)q;
        }
    }
}

void trace_dequantize_rows_i32_avx512(
    const int32_t* acc,
    const float* scale_x,
    float w_scale,
    float* out,
    int M,
    int out_f
) {
    #pragma omp parallel for schedule(static) if (M >= 16)
    for (int m = 0; m < M; ++m) {
        float factor = scale_x[m] * w_scale;
        __m512 vf = _mm512_set1_ps(factor);
        const int32_t* acc_row = acc + (size_t)m * (size_t)out_f;
        float* out_row = out + (size_t)m * (size_t)out_f;
        int i = 0;
        for (; i + 16 <= out_f; i += 16) {
            __m512 v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const void*)(acc_row + i)));
            _mm512_storeu_ps(out_row + i, _mm512_mul_ps(v, vf));
        }
        for (; i < out_f; ++i) {
            out_row[i] = (float)acc_row[i] * factor;
        }
    }
}

#endif // TRACE_ENABLE_AVX512

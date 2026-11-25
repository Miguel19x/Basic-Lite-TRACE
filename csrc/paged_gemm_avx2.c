/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#if defined(TRACE_ENABLE_AVX2)

#include "paged_dispatch.h"
#include "trace_infer.h"
#include "paged_unpack.h"
#include "trace_math.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define OMP_MIN_WORK_THRESHOLD 65536
#define K_TILE_SIZE 512

static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(1, 0, 3, 2)));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(acc128);
}

static inline float hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline int32_t avx2_dot_product_packed_fast(
    const int8_t* x_row,
    const uint8_t* w_row_packed,
    int64_t cols
) {
    __m256i acc_vec = _mm256_setzero_si256();
    __m256i ones_u8 = _mm256_set1_epi8(1);
    __m256i ones_i16 = _mm256_set1_epi16(1);

    int64_t k = 0;
    for (; k + 32 <= cols; k += 32) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i*)(x_row + k));
        __m256i w_vec = avx2_unpack_32_weights_fast(w_row_packed + (k >> 2));
        __m256i prod = _mm256_sign_epi8(x_vec, w_vec);
        __m256i s16 = _mm256_maddubs_epi16(ones_u8, prod);
        __m256i s32 = _mm256_madd_epi16(s16, ones_i16);
        acc_vec = _mm256_add_epi32(acc_vec, s32);
    }
    int32_t acc = hsum_epi32_avx2(acc_vec);
    for (; k < cols; ++k) {
        uint8_t byte_val = w_row_packed[k >> 2];
        int shift = (k & 3) * 2;
        int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
        acc += (int32_t)x_row[k] * w;
    }
    return acc;
}

static inline int32_t avx2_dot_product_dense_int8(
    const int8_t* x_row,
    const int8_t* w_row,
    int64_t cols
) {
    __m256i acc_vec = _mm256_setzero_si256();
    __m256i ones_u8 = _mm256_set1_epi8(1);
    __m256i ones_i16 = _mm256_set1_epi16(1);

    int64_t k = 0;
    for (; k + 32 <= cols; k += 32) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i*)(x_row + k));
        __m256i w_vec = _mm256_loadu_si256((const __m256i*)(w_row + k));
        __m256i prod = _mm256_sign_epi8(x_vec, w_vec);
        __m256i s16 = _mm256_maddubs_epi16(ones_u8, prod);
        __m256i s32 = _mm256_madd_epi16(s16, ones_i16);
        acc_vec = _mm256_add_epi32(acc_vec, s32);
    }
    int32_t acc = hsum_epi32_avx2(acc_vec);
    for (; k < cols; ++k) {
        acc += (int32_t)x_row[k] * (int32_t)w_row[k];
    }
    return acc;
}

static inline const uint8_t* paged_get_row_packed_avx2(
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

void paged_direct_matmul_fast_c_avx2(
    const int8_t* x,
    const uint8_t* page_blob,
    int32_t* out,
    int64_t M,
    int64_t K
) {
    if (!x || !page_blob || !out || M <= 0 || K <= 0) return;
    const PagedHeader* hdr = (const PagedHeader*)page_blob;
    if (hdr->fmt_code != 1) return;
    const uint32_t rows = hdr->rows;
    const uint16_t cols = hdr->cols;
    if ((int64_t)cols > K) return;
    const uint8_t* payload = page_blob + sizeof(PagedHeader);
    int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;
    int64_t total_work = M * (int64_t)rows * (int64_t)cols;

    #pragma omp parallel for collapse(2) schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t r = 0; r < (int64_t)rows; ++r) {
            const int8_t* x_row = x + m * K;
            const uint8_t* w_row_packed = payload + r * row_stride_bytes;
            out[m * rows + r] = avx2_dot_product_packed_fast(x_row, w_row_packed, cols);
        }
    }
}

void paged_direct_matmul_fast_c_multi_avx2(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
) {
    if (!x || !page_blobs || !out || num_pages <= 0 || M <= 0 || K <= 0 || total_out_features <= 0) return;
    int64_t total_work = M * total_out_features * K;

    #pragma omp parallel for schedule(dynamic, 1) if (total_work >= OMP_MIN_WORK_THRESHOLD)
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
            int32_t* out_row = out + row_start;
            int64_t r = 0;
            for (; r + 4 <= (int64_t)rows; r += 4) {
                const uint8_t* w0_p = payload + (r + 0) * row_stride_bytes;
                const uint8_t* w1_p = payload + (r + 1) * row_stride_bytes;
                const uint8_t* w2_p = payload + (r + 2) * row_stride_bytes;
                const uint8_t* w3_p = payload + (r + 3) * row_stride_bytes;

                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                __m256i ones_u8 = _mm256_set1_epi8(1);
                __m256i ones_i16 = _mm256_set1_epi16(1);

                int64_t k = 0;
                for (; k + 32 <= cols; k += 32) {
                    __m256i x_vec = _mm256_loadu_si256((const __m256i*)(x_row + k));
                    __m256i w0 = avx2_unpack_32_weights_fast(w0_p + (k >> 2));
                    __m256i w1 = avx2_unpack_32_weights_fast(w1_p + (k >> 2));
                    __m256i w2 = avx2_unpack_32_weights_fast(w2_p + (k >> 2));
                    __m256i w3 = avx2_unpack_32_weights_fast(w3_p + (k >> 2));

                    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(x_vec, w0)), ones_i16));
                    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(x_vec, w1)), ones_i16));
                    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(x_vec, w2)), ones_i16));
                    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(x_vec, w3)), ones_i16));
                }

                int32_t dot0 = hsum_epi32_avx2(acc0);
                int32_t dot1 = hsum_epi32_avx2(acc1);
                int32_t dot2 = hsum_epi32_avx2(acc2);
                int32_t dot3 = hsum_epi32_avx2(acc3);

                for (; k < cols; ++k) {
                    int8_t x_val = x_row[k];
                    int shift = (k & 3) * 2;
                    dot0 += (int32_t)x_val * (((int32_t)((w0_p[k >> 2] >> shift) & 0x03)) - 1);
                    dot1 += (int32_t)x_val * (((int32_t)((w1_p[k >> 2] >> shift) & 0x03)) - 1);
                    dot2 += (int32_t)x_val * (((int32_t)((w2_p[k >> 2] >> shift) & 0x03)) - 1);
                    dot3 += (int32_t)x_val * (((int32_t)((w3_p[k >> 2] >> shift) & 0x03)) - 1);
                }

                out_row[r + 0] = dot0;
                out_row[r + 1] = dot1;
                out_row[r + 2] = dot2;
                out_row[r + 3] = dot3;
            }
            for (; r < (int64_t)rows; ++r) {
                const uint8_t* w_row_packed = payload + r * row_stride_bytes;
                out_row[r] = avx2_dot_product_packed_fast(x_row, w_row_packed, cols);
            }
        } else if (cols == 0) {
            for (int64_t m = 0; m < M; ++m) {
                memset(out + m * total_out_features + row_start, 0, sizeof(int32_t) * (size_t)rows);
            }
        } else {
            int8_t w_unpacked[4][K_TILE_SIZE] __attribute__((aligned(32)));
            __m256i ones_u8 = _mm256_set1_epi8(1);
            __m256i ones_i16 = _mm256_set1_epi16(1);

            int64_t r = 0;
            for (; r + 4 <= (int64_t)rows; r += 4) {
                const uint8_t* wp0 = payload + (r + 0) * row_stride_bytes;
                const uint8_t* wp1 = payload + (r + 1) * row_stride_bytes;
                const uint8_t* wp2 = payload + (r + 2) * row_stride_bytes;
                const uint8_t* wp3 = payload + (r + 3) * row_stride_bytes;

                for (int64_t k_block = 0; k_block < (int64_t)cols; k_block += K_TILE_SIZE) {
                    int64_t chunk_cols = (int64_t)cols - k_block;
                    if (chunk_cols > K_TILE_SIZE) chunk_cols = K_TILE_SIZE;

                    const uint8_t* wps[4] = { wp0, wp1, wp2, wp3 };
                    for (int rr = 0; rr < 4; ++rr) {
                        int64_t k = 0;
                        for (; k + 32 <= chunk_cols; k += 32) {
                            __m256i w_vec = avx2_unpack_32_weights_fast(wps[rr] + ((k_block + k) >> 2));
                            _mm256_storeu_si256((__m256i*)(w_unpacked[rr] + k), w_vec);
                        }
                        for (; k < chunk_cols; ++k) {
                            uint8_t byte_val = wps[rr][(k_block + k) >> 2];
                            int shift = ((k_block + k) & 3) * 2;
                            w_unpacked[rr][k] = (int8_t)(((int32_t)((byte_val >> shift) & 0x03)) - 1);
                        }
                    }

                    for (int64_t m = 0; m < M; ++m) {
                        const int8_t* x_row = x + m * K + k_block;
                        __m256i acc0 = _mm256_setzero_si256();
                        __m256i acc1 = _mm256_setzero_si256();
                        __m256i acc2 = _mm256_setzero_si256();
                        __m256i acc3 = _mm256_setzero_si256();

                        int64_t k = 0;
                        for (; k + 32 <= chunk_cols; k += 32) {
                            __m256i xv = _mm256_loadu_si256((const __m256i*)(x_row + k));
                            __m256i wv0 = _mm256_load_si256((const __m256i*)(w_unpacked[0] + k));
                            __m256i wv1 = _mm256_load_si256((const __m256i*)(w_unpacked[1] + k));
                            __m256i wv2 = _mm256_load_si256((const __m256i*)(w_unpacked[2] + k));
                            __m256i wv3 = _mm256_load_si256((const __m256i*)(w_unpacked[3] + k));

                            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(xv, wv0)), ones_i16));
                            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(xv, wv1)), ones_i16));
                            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(xv, wv2)), ones_i16));
                            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(xv, wv3)), ones_i16));
                        }

                        int32_t dot0 = hsum_epi32_avx2(acc0);
                        int32_t dot1 = hsum_epi32_avx2(acc1);
                        int32_t dot2 = hsum_epi32_avx2(acc2);
                        int32_t dot3 = hsum_epi32_avx2(acc3);

                        for (; k < chunk_cols; ++k) {
                            int32_t xv_s = (int32_t)x_row[k];
                            dot0 += xv_s * (int32_t)w_unpacked[0][k];
                            dot1 += xv_s * (int32_t)w_unpacked[1][k];
                            dot2 += xv_s * (int32_t)w_unpacked[2][k];
                            dot3 += xv_s * (int32_t)w_unpacked[3][k];
                        }

                        int32_t* out_dst = out + m * total_out_features + row_start + r;
                        if (k_block == 0) {
                            out_dst[0] = dot0;
                            out_dst[1] = dot1;
                            out_dst[2] = dot2;
                            out_dst[3] = dot3;
                        } else {
                            out_dst[0] += dot0;
                            out_dst[1] += dot1;
                            out_dst[2] += dot2;
                            out_dst[3] += dot3;
                        }
                    }
                }
            }

            for (; r < (int64_t)rows; ++r) {
                const uint8_t* w_row_packed = payload + r * row_stride_bytes;

                for (int64_t k_block = 0; k_block < (int64_t)cols; k_block += K_TILE_SIZE) {
                    int64_t chunk_cols = (int64_t)cols - k_block;
                    if (chunk_cols > K_TILE_SIZE) chunk_cols = K_TILE_SIZE;

                    int64_t k = 0;
                    for (; k + 32 <= chunk_cols; k += 32) {
                        __m256i w_vec = avx2_unpack_32_weights_fast(w_row_packed + ((k_block + k) >> 2));
                        _mm256_storeu_si256((__m256i*)(w_unpacked[0] + k), w_vec);
                    }
                    for (; k < chunk_cols; ++k) {
                        uint8_t byte_val = w_row_packed[(k_block + k) >> 2];
                        int shift = ((k_block + k) & 3) * 2;
                        w_unpacked[0][k] = (int8_t)(((int32_t)((byte_val >> shift) & 0x03)) - 1);
                    }

                    for (int64_t m = 0; m < M; ++m) {
                        const int8_t* x_row = x + m * K + k_block;
                        int32_t dot = avx2_dot_product_dense_int8(x_row, w_unpacked[0], chunk_cols);
                        if (k_block == 0) {
                            out[m * total_out_features + row_start + r] = dot;
                        } else {
                            out[m * total_out_features + row_start + r] += dot;
                        }
                    }
                }
            }
        }
    }
}

void paged_lowrank_matmul_row_c_avx2(
    const int8_t* x,
    const uint8_t* w_packed,
    int32_t* out,
    int64_t rows,
    int64_t K
) {
    int64_t padded_cols = (K + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;
    int64_t total_work = rows * K;

    #pragma omp parallel for schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_w = w_packed + r * row_stride_bytes;
        out[r] = avx2_dot_product_packed_fast(x, row_w, K);
    }
}

void ternary_addsub_cpu_kernel_avx2(
    const int8_t* x,
    const int8_t* w,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t total_work = M * N * K;

    #pragma omp parallel for collapse(2) schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const int8_t* x_row = x + m * K;
            const int8_t* w_row = w + n * K;
            out[m * N + n] = avx2_dot_product_dense_int8(x_row, w_row, K);
        }
    }
}

void paged_absorb_uk_f32_avx2(
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
    #pragma omp parallel for schedule(static) if (S * n_heads >= 16)
    for (int h = 0; h < n_heads; ++h) {
        for (int t = 0; t < S; ++t) {
            float* out_hc = q_c_abs + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;
            memset(out_hc, 0, sizeof(float) * (size_t)kv_lora_rank);
        }

        for (int d = 0; d < qk_nope_head_dim; ++d) {
            int r = h * qk_nope_head_dim + d;
            const uint8_t* row_p = paged_get_row_packed_avx2(page_ptrs, num_pages, r);
            if (!row_p) continue;

            int k = 0;
            for (; k + 32 <= kv_lora_rank; k += 32) {
                __m256i w_i8 = avx2_unpack_32_weights_fast(row_p + (k >> 2));
                __m128i lo16 = _mm256_castsi256_si128(w_i8);
                __m128i hi16 = _mm256_extracti128_si256(w_i8, 1);

                __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8)));
                __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8)));

                for (int t = 0; t < S; ++t) {
                    float v = q_c[((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)qk_nope_head_dim + (size_t)d];
                    if (v == 0.0f) continue;
                    __m256 v_vec = _mm256_set1_ps(v);
                    float* out_hc = q_c_abs + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank + (size_t)k;
                    _mm256_storeu_ps(out_hc + 0, _mm256_fmadd_ps(v_vec, w0, _mm256_loadu_ps(out_hc + 0)));
                    _mm256_storeu_ps(out_hc + 8, _mm256_fmadd_ps(v_vec, w1, _mm256_loadu_ps(out_hc + 8)));
                    _mm256_storeu_ps(out_hc + 16, _mm256_fmadd_ps(v_vec, w2, _mm256_loadu_ps(out_hc + 16)));
                    _mm256_storeu_ps(out_hc + 24, _mm256_fmadd_ps(v_vec, w3, _mm256_loadu_ps(out_hc + 24)));
                }
            }

            // Cola escalar segura para dimensiones no divisibles entre 32
            for (; k < kv_lora_rank; ++k) {
                uint8_t byte_val = row_p[k >> 2];
                int shift = (k & 3) * 2;
                float wf = (float)(((int32_t)((byte_val >> shift) & 0x03)) - 1);
                if (wf == 0.0f) continue;
                for (int t = 0; t < S; ++t) {
                    float v = q_c[((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)qk_nope_head_dim + (size_t)d];
                    q_c_abs[((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank + (size_t)k] += v * wf;
                }
            }
        }

        __m256 scale_vec = _mm256_set1_ps(scale);
        for (int t = 0; t < S; ++t) {
            float* out_hc = q_c_abs + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;
            int k = 0;
            for (; k + 8 <= kv_lora_rank; k += 8) {
                _mm256_storeu_ps(out_hc + k, _mm256_mul_ps(_mm256_loadu_ps(out_hc + k), scale_vec));
            }
            for (; k < kv_lora_rank; ++k) {
                out_hc[k] *= scale;
            }
        }
    }
}

void paged_absorb_uv_f32_avx2(
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
    #pragma omp parallel for schedule(static) if (S * n_heads >= 16)
    for (int h = 0; h < n_heads; ++h) {
        for (int v = 0; v < v_head_dim; ++v) {
            int r = h * v_head_dim + v;
            const uint8_t* row_p = paged_get_row_packed_avx2(page_ptrs, num_pages, r);
            if (!row_p) {
                for (int t = 0; t < S; ++t) {
                    o_heads[((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)v_head_dim + (size_t)v] = 0.0f;
                }
                continue;
            }

            int t = 0;
            for (; t + 4 <= S; t += 4) {
                const float* lat0 = o_latent + ((size_t)(t + 0) * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;
                const float* lat1 = o_latent + ((size_t)(t + 1) * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;
                const float* lat2 = o_latent + ((size_t)(t + 2) * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;
                const float* lat3 = o_latent + ((size_t)(t + 3) * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                int k = 0;
                for (; k + 32 <= kv_lora_rank; k += 32) {
                    __m256i w_i8 = avx2_unpack_32_weights_fast(row_p + (k >> 2));
                    __m128i lo16 = _mm256_castsi256_si128(w_i8);
                    __m128i hi16 = _mm256_extracti128_si256(w_i8, 1);

                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8)));
                    __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                    __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8)));

                    acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(lat0 + k + 0), acc0);
                    acc0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(lat0 + k + 8), acc0);
                    acc0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(lat0 + k + 16), acc0);
                    acc0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(lat0 + k + 24), acc0);

                    acc1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(lat1 + k + 0), acc1);
                    acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(lat1 + k + 8), acc1);
                    acc1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(lat1 + k + 16), acc1);
                    acc1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(lat1 + k + 24), acc1);

                    acc2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(lat2 + k + 0), acc2);
                    acc2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(lat2 + k + 8), acc2);
                    acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(lat2 + k + 16), acc2);
                    acc2 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(lat2 + k + 24), acc2);

                    acc3 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(lat3 + k + 0), acc3);
                    acc3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(lat3 + k + 8), acc3);
                    acc3 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(lat3 + k + 16), acc3);
                    acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(lat3 + k + 24), acc3);
                }

                float d0 = hsum_ps_avx2(acc0);
                float d1 = hsum_ps_avx2(acc1);
                float d2 = hsum_ps_avx2(acc2);
                float d3 = hsum_ps_avx2(acc3);

                for (; k < kv_lora_rank; ++k) {
                    uint8_t byte_val = row_p[k >> 2];
                    int shift = (k & 3) * 2;
                    float wf = (float)(((int32_t)((byte_val >> shift) & 0x03)) - 1);
                    d0 += lat0[k] * wf;
                    d1 += lat1[k] * wf;
                    d2 += lat2[k] * wf;
                    d3 += lat3[k] * wf;
                }

                o_heads[((size_t)(t + 0) * (size_t)n_heads + (size_t)h) * (size_t)v_head_dim + (size_t)v] = d0 * scale;
                o_heads[((size_t)(t + 1) * (size_t)n_heads + (size_t)h) * (size_t)v_head_dim + (size_t)v] = d1 * scale;
                o_heads[((size_t)(t + 2) * (size_t)n_heads + (size_t)h) * (size_t)v_head_dim + (size_t)v] = d2 * scale;
                o_heads[((size_t)(t + 3) * (size_t)n_heads + (size_t)h) * (size_t)v_head_dim + (size_t)v] = d3 * scale;
            }

            for (; t < S; ++t) {
                const float* lat_h = o_latent + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)kv_lora_rank;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                int k = 0;
                for (; k + 32 <= kv_lora_rank; k += 32) {
                    __m256i w_i8 = avx2_unpack_32_weights_fast(row_p + (k >> 2));
                    __m128i lo16 = _mm256_castsi256_si128(w_i8);
                    __m128i hi16 = _mm256_extracti128_si256(w_i8, 1);

                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8)));
                    __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                    __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8)));

                    acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(lat_h + k), acc0);
                    acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(lat_h + k + 8), acc1);
                    acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(lat_h + k + 16), acc2);
                    acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(lat_h + k + 24), acc3);
                }

                __m256 sum256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                float dot = hsum_ps_avx2(sum256);

                for (; k < kv_lora_rank; ++k) {
                    uint8_t byte_val = row_p[k >> 2];
                    int shift = (k & 3) * 2;
                    int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                    dot += lat_h[k] * (float)w;
                }
                o_heads[((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)v_head_dim + (size_t)v] = dot * scale;
            }
        }
    }
}

float trace_dot_f32_avx2(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    float dot = hsum_ps_avx2(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

void trace_axpy_f32_avx2(float* y, float a, const float* x, int n) {
    __m256 a_vec = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(a_vec, vx, vy));
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
}

void trace_mla_attn_group_f32_avx2(
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

        float max0 = -INFINITY;
        float max1 = -INFINITY;
        float max2 = -INFINITY;
        float max3 = -INFINITY;

        // Paso 1: Scores compartiendo c_kv[s] y k_pe[s] entre 4 cabezas en registros AVX2
        for (int s = 0; s < total_len; ++s) {
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            const float* k_pe_s = k_pe_layer + (size_t)s * (size_t)qk_rope_head_dim;

            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();

            int i = 0;
            for (; i + 8 <= kv_lora_rank; i += 8) {
                __m256 vc = _mm256_loadu_ps(c_kv_s + i);
                a0 = _mm256_fmadd_ps(vc, _mm256_loadu_ps(qc0 + i), a0);
                a1 = _mm256_fmadd_ps(vc, _mm256_loadu_ps(qc1 + i), a1);
                a2 = _mm256_fmadd_ps(vc, _mm256_loadu_ps(qc2 + i), a2);
                a3 = _mm256_fmadd_ps(vc, _mm256_loadu_ps(qc3 + i), a3);
            }
            float d0 = hsum_ps_avx2(a0);
            float d1 = hsum_ps_avx2(a1);
            float d2 = hsum_ps_avx2(a2);
            float d3 = hsum_ps_avx2(a3);
            for (; i < kv_lora_rank; ++i) {
                float vc = c_kv_s[i];
                d0 += vc * qc0[i];
                d1 += vc * qc1[i];
                d2 += vc * qc2[i];
                d3 += vc * qc3[i];
            }

            __m256 p0 = _mm256_setzero_ps();
            __m256 p1 = _mm256_setzero_ps();
            __m256 p2 = _mm256_setzero_ps();
            __m256 p3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= qk_rope_head_dim; j += 8) {
                __m256 vk = _mm256_loadu_ps(k_pe_s + j);
                p0 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(qp0 + j), p0);
                p1 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(qp1 + j), p1);
                p2 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(qp2 + j), p2);
                p3 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(qp3 + j), p3);
            }
            d0 += hsum_ps_avx2(p0);
            d1 += hsum_ps_avx2(p1);
            d2 += hsum_ps_avx2(p2);
            d3 += hsum_ps_avx2(p3);
            for (; j < qk_rope_head_dim; ++j) {
                float vk = k_pe_s[j];
                d0 += vk * qp0[j];
                d1 += vk * qp1[j];
                d2 += vk * qp2[j];
                d3 += vk * qp3[j];
            }

            float v0 = d0 * attn_scale;
            float v1 = d1 * attn_scale;
            float v2 = d2 * attn_scale;
            float v3 = d3 * attn_scale;
            sc0[s] = v0;
            sc1[s] = v1;
            sc2[s] = v2;
            sc3[s] = v3;
            if (v0 > max0) max0 = v0;
            if (v1 > max1) max1 = v1;
            if (v2 > max2) max2 = v2;
            if (v3 > max3) max3 = v3;
        }

        // Paso 2: Softmax estable in-place sobre scores_scratch
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
            __m256 inv_vec = _mm256_set1_ps(sum_inv);
            int s = 0;
            for (; s + 8 <= total_len; s += 8) {
                _mm256_storeu_ps(sc + s, _mm256_mul_ps(_mm256_loadu_ps(sc + s), inv_vec));
            }
            for (; s < total_len; ++s) {
                sc[s] *= sum_inv;
            }
        }

        // Paso 3: Agregacion compartiendo c_kv[s] entre 4 cabezas
        float* o0 = o_lat_base + (size_t)(h_start + 0) * (size_t)kv_lora_rank;
        float* o1 = o_lat_base + (size_t)(h_start + 1) * (size_t)kv_lora_rank;
        float* o2 = o_lat_base + (size_t)(h_start + 2) * (size_t)kv_lora_rank;
        float* o3 = o_lat_base + (size_t)(h_start + 3) * (size_t)kv_lora_rank;
        memset(o0, 0, sizeof(float) * (size_t)kv_lora_rank);
        memset(o1, 0, sizeof(float) * (size_t)kv_lora_rank);
        memset(o2, 0, sizeof(float) * (size_t)kv_lora_rank);
        memset(o3, 0, sizeof(float) * (size_t)kv_lora_rank);

        for (int s = 0; s < total_len; ++s) {
            float pr0 = sc0[s];
            float pr1 = sc1[s];
            float pr2 = sc2[s];
            float pr3 = sc3[s];
            if (pr0 == 0.0f && pr1 == 0.0f && pr2 == 0.0f && pr3 == 0.0f) continue;

            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            __m256 vp0 = _mm256_set1_ps(pr0);
            __m256 vp1 = _mm256_set1_ps(pr1);
            __m256 vp2 = _mm256_set1_ps(pr2);
            __m256 vp3 = _mm256_set1_ps(pr3);

            int i = 0;
            for (; i + 8 <= kv_lora_rank; i += 8) {
                __m256 vc = _mm256_loadu_ps(c_kv_s + i);
                _mm256_storeu_ps(o0 + i, _mm256_fmadd_ps(vp0, vc, _mm256_loadu_ps(o0 + i)));
                _mm256_storeu_ps(o1 + i, _mm256_fmadd_ps(vp1, vc, _mm256_loadu_ps(o1 + i)));
                _mm256_storeu_ps(o2 + i, _mm256_fmadd_ps(vp2, vc, _mm256_loadu_ps(o2 + i)));
                _mm256_storeu_ps(o3 + i, _mm256_fmadd_ps(vp3, vc, _mm256_loadu_ps(o3 + i)));
            }
            for (; i < kv_lora_rank; ++i) {
                float vc = c_kv_s[i];
                o0[i] += pr0 * vc;
                o1[i] += pr1 * vc;
                o2[i] += pr2 * vc;
                o3[i] += pr3 * vc;
            }
        }
        return;
    }

    // Resto (1..3 cabezas) sin despacho indirecto
    for (int gh = 0; gh < h_count; ++gh) {
        int h = h_start + gh;
        const float* q_c_h = q_c_abs_base + (size_t)h * (size_t)kv_lora_rank;
        const float* q_pe_h = q_pe_base + (size_t)h * (size_t)qk_rope_head_dim;
        float* sc_h = scores_scratch + (size_t)gh * (size_t)total_len;

        float max_sc = -INFINITY;
        for (int s = 0; s < total_len; ++s) {
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            const float* k_pe_s = k_pe_layer + (size_t)s * (size_t)qk_rope_head_dim;
            float dot_c = trace_dot_f32_avx2(q_c_h, c_kv_s, kv_lora_rank);
            float dot_pe = trace_dot_f32_avx2(q_pe_h, k_pe_s, qk_rope_head_dim);
            float sc = (dot_c + dot_pe) * attn_scale;
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
        for (int s = 0; s < total_len; ++s) {
            sc_h[s] *= sum_inv;
        }

        float* o_lat_h = o_lat_base + (size_t)h * (size_t)kv_lora_rank;
        memset(o_lat_h, 0, sizeof(float) * (size_t)kv_lora_rank);
        for (int s = 0; s < total_len; ++s) {
            float p = sc_h[s];
            if (p == 0.0f) continue;
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            trace_axpy_f32_avx2(o_lat_h, p, c_kv_s, kv_lora_rank);
        }
    }
}

float trace_dot_f32_f16_avx2(const float* x, const uint16_t* w_f16, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(w_f16 + i));
        __m128i h1 = _mm_loadu_si128((const __m128i*)(w_f16 + i + 8));
        __m256 w0 = _mm256_cvtph_ps(h0);
        __m256 w1 = _mm256_cvtph_ps(h1);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), w1, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(w_f16 + i));
        __m256 w0 = _mm256_cvtph_ps(h0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w0, acc0);
    }
    float dot = hsum_ps_avx2(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        dot += x[i] * trace_f16_to_f32(w_f16[i]);
    }
    return dot;
}

void trace_matvec_f32_f16_avx2(const float* x, const uint16_t* w_f16, float* out, int rows, int cols) {
    int blocks = rows / 4;
    #pragma omp parallel for schedule(static) if (rows >= 256)
    for (int b = 0; b < blocks; ++b) {
        int v = b * 4;
        const uint16_t* w0 = w_f16 + (size_t)(v + 0) * (size_t)cols;
        const uint16_t* w1 = w_f16 + (size_t)(v + 1) * (size_t)cols;
        const uint16_t* w2 = w_f16 + (size_t)(v + 2) * (size_t)cols;
        const uint16_t* w3 = w_f16 + (size_t)(v + 3) * (size_t)cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= cols; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            a0 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0 + i))), a0);
            a1 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1 + i))), a1);
            a2 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2 + i))), a2);
            a3 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3 + i))), a3);
        }
        float d0 = hsum_ps_avx2(a0), d1 = hsum_ps_avx2(a1), d2 = hsum_ps_avx2(a2), d3 = hsum_ps_avx2(a3);
        for (; i < cols; ++i) {
            float xv = x[i];
            d0 += xv * trace_f16_to_f32(w0[i]);
            d1 += xv * trace_f16_to_f32(w1[i]);
            d2 += xv * trace_f16_to_f32(w2[i]);
            d3 += xv * trace_f16_to_f32(w3[i]);
        }
        out[v + 0] = d0; out[v + 1] = d1; out[v + 2] = d2; out[v + 3] = d3;
    }
    for (int v = blocks * 4; v < rows; ++v) {
        out[v] = trace_dot_f32_f16_avx2(x, w_f16 + (size_t)v * (size_t)cols, cols);
    }
}

float trace_dot_f32_i8_avx2(const float* x, const int8_t* w_i8, float scale, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b = _mm_loadu_si128((const __m128i*)(w_i8 + i));
        __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8)));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), w1, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m128i b = _mm_loadl_epi64((const __m128i*)(w_i8 + i));
        __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w0, acc0);
    }
    float dot = hsum_ps_avx2(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        dot += x[i] * (float)w_i8[i];
    }
    return dot * scale;
}

void trace_gqa_attn_group_f32_avx2(
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

        float max0 = -INFINITY;
        float max1 = -INFINITY;
        float max2 = -INFINITY;
        float max3 = -INFINITY;

        for (int s = 0; s < total_len; ++s) {
            const float* k_s = k_layer + (size_t)s * kv_stride + kv_off;
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= head_dim; i += 8) {
                __m256 vk = _mm256_loadu_ps(k_s + i);
                a0 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(q0 + i), a0);
                a1 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(q1 + i), a1);
                a2 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(q2 + i), a2);
                a3 = _mm256_fmadd_ps(vk, _mm256_loadu_ps(q3 + i), a3);
            }
            float d0 = hsum_ps_avx2(a0);
            float d1 = hsum_ps_avx2(a1);
            float d2 = hsum_ps_avx2(a2);
            float d3 = hsum_ps_avx2(a3);
            for (; i < head_dim; ++i) {
                float vk = k_s[i];
                d0 += vk * q0[i];
                d1 += vk * q1[i];
                d2 += vk * q2[i];
                d3 += vk * q3[i];
            }
            float v0 = d0 * attn_scale;
            float v1 = d1 * attn_scale;
            float v2 = d2 * attn_scale;
            float v3 = d3 * attn_scale;
            sc0[s] = v0;
            sc1[s] = v1;
            sc2[s] = v2;
            sc3[s] = v3;
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
            __m256 inv_vec = _mm256_set1_ps(sum_inv);
            int s = 0;
            for (; s + 8 <= total_len; s += 8) {
                _mm256_storeu_ps(sc + s, _mm256_mul_ps(_mm256_loadu_ps(sc + s), inv_vec));
            }
            for (; s < total_len; ++s) {
                sc[s] *= sum_inv;
            }
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
            float pr0 = sc0[s];
            float pr1 = sc1[s];
            float pr2 = sc2[s];
            float pr3 = sc3[s];
            if (pr0 == 0.0f && pr1 == 0.0f && pr2 == 0.0f && pr3 == 0.0f) continue;

            const float* v_s = v_layer + (size_t)s * kv_stride + kv_off;
            __m256 vp0 = _mm256_set1_ps(pr0);
            __m256 vp1 = _mm256_set1_ps(pr1);
            __m256 vp2 = _mm256_set1_ps(pr2);
            __m256 vp3 = _mm256_set1_ps(pr3);

            int i = 0;
            for (; i + 8 <= head_dim; i += 8) {
                __m256 vv = _mm256_loadu_ps(v_s + i);
                _mm256_storeu_ps(o0 + i, _mm256_fmadd_ps(vp0, vv, _mm256_loadu_ps(o0 + i)));
                _mm256_storeu_ps(o1 + i, _mm256_fmadd_ps(vp1, vv, _mm256_loadu_ps(o1 + i)));
                _mm256_storeu_ps(o2 + i, _mm256_fmadd_ps(vp2, vv, _mm256_loadu_ps(o2 + i)));
                _mm256_storeu_ps(o3 + i, _mm256_fmadd_ps(vp3, vv, _mm256_loadu_ps(o3 + i)));
            }
            for (; i < head_dim; ++i) {
                float vv = v_s[i];
                o0[i] += pr0 * vv;
                o1[i] += pr1 * vv;
                o2[i] += pr2 * vv;
                o3[i] += pr3 * vv;
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
            float sc = trace_dot_f32_avx2(q_h, k_s, head_dim) * attn_scale;
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
        for (int s = 0; s < total_len; ++s) {
            sc_h[s] *= sum_inv;
        }

        float* o_h = o_heads_base + (size_t)qh * (size_t)head_dim;
        memset(o_h, 0, sizeof(float) * (size_t)head_dim);
        for (int s = 0; s < total_len; ++s) {
            float p = sc_h[s];
            if (p == 0.0f) continue;
            const float* v_s = v_layer + (size_t)s * kv_stride + kv_off;
            trace_axpy_f32_avx2(o_h, p, v_s, head_dim);
        }
    }
}

void trace_quantize_rows_int8_avx2(
    const float* in,
    const float* sub_ln_w,
    float* x_norm,
    int8_t* q_x,
    float* scale_x,
    int M,
    int in_f,
    float eps
) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 v_max = _mm256_set1_ps(127.0f);
    const __m256 v_min = _mm256_set1_ps(-127.0f);

    #pragma omp parallel for schedule(static) if (M >= 16)
    for (int m = 0; m < M; ++m) {
        const float* in_row = in + (size_t)m * (size_t)in_f;
        const float* row = in_row;
        if (sub_ln_w != NULL) {
            float* norm_row = x_norm + (size_t)m * (size_t)in_f;
            rms_norm(in_row, sub_ln_w, norm_row, in_f, eps);
            row = norm_row;
        }

        __m256 vmax = _mm256_set1_ps(eps);
        int i = 0;
        for (; i + 8 <= in_f; i += 8) {
            __m256 v = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(row + i));
            vmax = _mm256_max_ps(vmax, v);
        }
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        float amax = _mm_cvtss_f32(m128);
        for (; i < in_f; ++i) {
            float v = fabsf(row[i]);
            if (v > amax) amax = v;
        }

        float scale = amax / 127.0f;
        float scale_inv = 127.0f / amax;
        scale_x[m] = scale;
        __m256 v_sinv = _mm256_set1_ps(scale_inv);
        int8_t* qrow = q_x + (size_t)m * (size_t)in_f;

        i = 0;
        for (; i + 8 <= in_f; i += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + i), v_sinv);
            v = _mm256_min_ps(_mm256_max_ps(v, v_min), v_max);
            __m256i vi = _mm256_cvtps_epi32(v);
            __m128i lo = _mm256_castsi256_si128(vi);
            __m128i hi = _mm256_extracti128_si256(vi, 1);
            __m128i p16 = _mm_packs_epi32(lo, hi);
            __m128i p8 = _mm_packs_epi16(p16, p16);
            _mm_storel_epi64((__m128i*)(qrow + i), p8);
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

void trace_dequantize_rows_i32_avx2(
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
        __m256 vf = _mm256_set1_ps(factor);
        const int32_t* acc_row = acc + (size_t)m * (size_t)out_f;
        float* out_row = out + (size_t)m * (size_t)out_f;
        int i = 0;
        for (; i + 8 <= out_f; i += 8) {
            __m256 v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(acc_row + i)));
            _mm256_storeu_ps(out_row + i, _mm256_mul_ps(v, vf));
        }
        for (; i < out_f; ++i) {
            out_row[i] = (float)acc_row[i] * factor;
        }
    }
}

#endif // TRACE_ENABLE_AVX2

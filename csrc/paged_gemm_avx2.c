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
#include <stdint.h>
#include <string.h>
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
    const PagedHeader* hdr = (const PagedHeader*)page_blob;
    if (hdr->fmt_code != 1) return;
    const uint32_t rows = hdr->rows;
    const uint16_t cols = hdr->cols;
    const uint8_t* payload = page_blob + sizeof(PagedHeader);
    int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;
    int64_t total_work = M * (int64_t)rows * (int64_t)cols;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
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
    int64_t total_work = M * total_out_features * K;

    #pragma omp parallel for schedule(dynamic, 1) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t p = 0; p < num_pages; ++p) {
        const uint8_t* blob = page_blobs[p];
        const PagedHeader* hdr = (const PagedHeader*)blob;
        if (hdr->fmt_code != 1) continue;

        const uint32_t row_start = hdr->row_start;
        const uint32_t rows = hdr->rows;
        const uint16_t cols = hdr->cols;
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
        } else {
            for (int64_t r = 0; r < (int64_t)rows; ++r) {
                const uint8_t* w_row_packed = payload + r * row_stride_bytes;
                int8_t w_unpacked[K_TILE_SIZE] __attribute__((aligned(32)));

                for (int64_t k_block = 0; k_block < (int64_t)cols; k_block += K_TILE_SIZE) {
                    int64_t chunk_cols = (int64_t)cols - k_block;
                    if (chunk_cols > K_TILE_SIZE) chunk_cols = K_TILE_SIZE;

                    int64_t k = 0;
                    for (; k + 32 <= chunk_cols; k += 32) {
                        __m256i w_vec = avx2_unpack_32_weights_fast(w_row_packed + ((k_block + k) >> 2));
                        _mm256_storeu_si256((__m256i*)(w_unpacked + k), w_vec);
                    }
                    for (; k < chunk_cols; ++k) {
                        uint8_t byte_val = w_row_packed[(k_block + k) >> 2];
                        int shift = ((k_block + k) & 3) * 2;
                        w_unpacked[k] = (int8_t)(((int32_t)((byte_val >> shift) & 0x03)) - 1);
                    }

                    for (int64_t m = 0; m < M; ++m) {
                        const int8_t* x_row = x + m * K + k_block;
                        int32_t dot = avx2_dot_product_dense_int8(x_row, w_unpacked, chunk_cols);
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

    #pragma omp parallel for schedule(static) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
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

    #pragma omp parallel for collapse(2) schedule(static) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
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
    for (int t = 0; t < S; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float* out_hc = q_c_abs + (t * n_heads + h) * kv_lora_rank;
            memset(out_hc, 0, sizeof(float) * kv_lora_rank);

            for (int d = 0; d < qk_nope_head_dim; ++d) {
                int r = h * qk_nope_head_dim + d;
                const uint8_t* row_p = paged_get_row_packed_avx2(page_ptrs, num_pages, r);
                if (!row_p) continue;

                float v = q_c[(t * n_heads + h) * qk_nope_head_dim + d];
                if (v == 0.0f) continue;
                __m256 v_vec = _mm256_set1_ps(v);

                int k = 0;
                for (; k + 32 <= kv_lora_rank; k += 32) {
                    __m256i w_i8 = avx2_unpack_32_weights_fast(row_p + (k >> 2));
                    __m128i lo16 = _mm256_castsi256_si128(w_i8);
                    __m128i hi16 = _mm256_extracti128_si256(w_i8, 1);

                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8)));
                    __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                    __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8)));

                    _mm256_storeu_ps(out_hc + k, _mm256_fmadd_ps(v_vec, w0, _mm256_loadu_ps(out_hc + k)));
                    _mm256_storeu_ps(out_hc + k + 8, _mm256_fmadd_ps(v_vec, w1, _mm256_loadu_ps(out_hc + k + 8)));
                    _mm256_storeu_ps(out_hc + k + 16, _mm256_fmadd_ps(v_vec, w2, _mm256_loadu_ps(out_hc + k + 16)));
                    _mm256_storeu_ps(out_hc + k + 24, _mm256_fmadd_ps(v_vec, w3, _mm256_loadu_ps(out_hc + k + 24)));
                }

                // Cola escalar segura para dimensiones no divisibles entre 32
                for (; k < kv_lora_rank; ++k) {
                    uint8_t byte_val = row_p[k >> 2];
                    int shift = (k & 3) * 2;
                    int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                    out_hc[k] += v * (float)w;
                }
            }

            for (int k = 0; k < kv_lora_rank; ++k) {
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
    for (int t = 0; t < S; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const float* lat_h = o_latent + (t * n_heads + h) * kv_lora_rank;
            float* out_h = o_heads + (t * n_heads + h) * v_head_dim;

            for (int v = 0; v < v_head_dim; ++v) {
                int r = h * v_head_dim + v;
                const uint8_t* row_p = paged_get_row_packed_avx2(page_ptrs, num_pages, r);
                if (!row_p) {
                    out_h[v] = 0.0f;
                    continue;
                }

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

                // Cola escalar segura para dimensiones no divisibles entre 32
                for (; k < kv_lora_rank; ++k) {
                    uint8_t byte_val = row_p[k >> 2];
                    int shift = (k & 3) * 2;
                    int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                    dot += lat_h[k] * (float)w;
                }
                out_h[v] = dot * scale;
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

#endif // TRACE_ENABLE_AVX2

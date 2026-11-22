/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#include "trace_infer.h"
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#define OMP_MIN_WORK_THRESHOLD 65536
#define K_TILE_SIZE 512

static inline __m256i avx2_unpack_32_weights_fast(const uint8_t* p) {
    __m128i raw8 = _mm_loadl_epi64((const __m128i*)p);
    __m128i low4 = raw8;
    __m128i high4 = _mm_srli_si128(raw8, 4);
    __m256i raw256 = _mm256_set_m128i(high4, low4);

    __m256i shuf_mask = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    );
    __m256i dup = _mm256_shuffle_epi8(raw256, shuf_mask);

    __m256i s0 = dup;
    __m256i s2 = _mm256_srli_epi16(dup, 2);
    __m256i s4 = _mm256_srli_epi16(dup, 4);
    __m256i s6 = _mm256_srli_epi16(dup, 6);

    __m256i mask_odd_bytes = _mm256_set1_epi32((int32_t)0xFF00FF00);
    __m256i s0_s2 = _mm256_blendv_epi8(s0, s2, mask_odd_bytes);
    __m256i s4_s6 = _mm256_blendv_epi8(s4, s6, mask_odd_bytes);

    __m256i mask_high_bytes = _mm256_set1_epi32((int32_t)0xFFFF0000);
    __m256i combined = _mm256_blendv_epi8(s0_s2, s4_s6, mask_high_bytes);

    __m256i codes = _mm256_and_si256(combined, _mm256_set1_epi8(0x03));
    return _mm256_sub_epi8(codes, _mm256_set1_epi8(1));
}

static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(1, 0, 3, 2)));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(acc128);
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

void paged_direct_matmul_fast_c(
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

void paged_direct_matmul_fast_c_multi(
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
            // Camino optimizado M=1: procesar 4 filas simultáneas reutilizando x_vec
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
            // Camino M > 1: Tiling por bloques K=512 para no desbordar la pila en capas grandes
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

void paged_lowrank_matmul_row_c(
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

void ternary_addsub_cpu_kernel(
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

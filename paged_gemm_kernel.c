/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#pragma pack(push, 1)
typedef struct {
    uint32_t row_start;
    uint32_t rows;
    uint16_t cols;
    uint16_t fmt_code; // 0: compact, 1: fast
    float scale_w;
    uint32_t len_p1;
    uint32_t len_p2;
} PagedHeader;
#pragma pack(pop)

// Umbral mínimo de operaciones (M * N * K) para compensar el coste de bifurcación OpenMP
#define OMP_MIN_WORK_THRESHOLD 65536

/*
 * Desempaqueta 8 bytes (32 pesos empaquetados en 2 bits) a un vector AVX2 __m256i
 * de 32 valores int8_t en {-1, 0, +1} preservando estrictamente el orden [w0, w1, ..., w31].
 *
 * Mapeo Fast en Paged BitCOS:
 *   c = (w + 1) & 0x03  =>  0: -1, 1: 0, 2: +1
 * Cada byte b contiene 4 códigos:
 *   bits 0..1: w0, bits 2..3: w1, bits 4..5: w2, bits 6..7: w3
 */
static inline __m256i avx2_unpack_32_weights_fast(const uint8_t* p) {
    // 1. Cargar 8 bytes en la parte baja de un registro de 128 bits
    __m128i raw8 = _mm_loadl_epi64((const __m128i*)p);

    // 2. Distribuir bytes 0..3 al carril bajo de 128 bits y bytes 4..7 al carril alto
    __m128i low4 = raw8;
    __m128i high4 = _mm_srli_si128(raw8, 4);
    __m256i raw256 = _mm256_set_m128i(high4, low4);

    // 3. Duplicar cada byte 4 veces dentro de cada carril de 128 bits mediante vpshufb
    __m256i shuf_mask = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    );
    __m256i dup = _mm256_shuffle_epi8(raw256, shuf_mask);

    // 4. Extraer los campos de 2 bits por palabra de 16 bits
    __m256i s0 = dup;
    __m256i s2 = _mm256_srli_epi16(dup, 2);
    __m256i s4 = _mm256_srli_epi16(dup, 4);
    __m256i s6 = _mm256_srli_epi16(dup, 6);

    // 5. Mezclar carriles:
    // mask_odd_bytes tiene 0xFF en bytes 1 y 3 (bytes impares de cada palabra de 16 bits)
    __m256i mask_odd_bytes = _mm256_set1_epi32((int32_t)0xFF00FF00);
    __m256i s0_s2 = _mm256_blendv_epi8(s0, s2, mask_odd_bytes);
    __m256i s4_s6 = _mm256_blendv_epi8(s4, s6, mask_odd_bytes);

    // combined tiene s0_s2 en bytes 0..1 y s4_s6 en bytes 2..3
    __m256i mask_high_bytes = _mm256_set1_epi32((int32_t)0xFFFF0000);
    __m256i combined = _mm256_blendv_epi8(s0_s2, s4_s6, mask_high_bytes);

    // 6. Enmascarar con 0x03 y restar 1 para recuperar el valor ternario {-1, 0, +1}
    __m256i codes = _mm256_and_si256(combined, _mm256_set1_epi8(0x03));
    return _mm256_sub_epi8(codes, _mm256_set1_epi8(1));
}

/*
 * Producto punto vector-vector empaquetado Fast con acumulación exacta en int32.
 */
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

        // vpsignb niega x_i si w_i == -1, pone 0 si w_i == 0, y mantiene si w_i == 1.
        // Como x in [-127, 127], nunca se produce desbordamiento de -128.
        __m256i prod = _mm256_sign_epi8(x_vec, w_vec);

        // Suma adyacente con signo a int16: prod[2i] + prod[2i+1] (rango [-254, 254])
        __m256i s16 = _mm256_maddubs_epi16(ones_u8, prod);

        // Suma adyacente a int32: s16[2j] + s16[2j+1]
        __m256i s32 = _mm256_madd_epi16(s16, ones_i16);

        acc_vec = _mm256_add_epi32(acc_vec, s32);
    }

    // Reducción horizontal de acc_vec (8 x int32 -> 1 x int32)
    __m128i acc128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc_vec),
        _mm256_extracti128_si256(acc_vec, 1)
    );
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(1, 0, 3, 2)));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(2, 3, 0, 1)));
    int32_t acc = _mm_cvtsi128_si32(acc128);

    // Residuo escalar para cols no múltiplos de 32 (ej. cols=50)
    for (; k < cols; ++k) {
        uint8_t byte_val = w_row_packed[k >> 2];
        int shift = (k & 3) * 2;
        int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
        acc += (int32_t)x_row[k] * w;
    }

    return acc;
}

/*
 * Producto punto vector-vector denso int8 con acumulación exacta en int32.
 */
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

    __m128i acc128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc_vec),
        _mm256_extracti128_si256(acc_vec, 1)
    );
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(1, 0, 3, 2)));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, _MM_SHUFFLE(2, 3, 0, 1)));
    int32_t acc = _mm_cvtsi128_si32(acc128);

    for (; k < cols; ++k) {
        acc += (int32_t)x_row[k] * (int32_t)w_row[k];
    }
    return acc;
}

/*
 * Kernel C optimizado para una página individual Paged BitCOS Fast Format.
 */
void paged_direct_matmul_fast_c(
    const int8_t* x,              // [M, K]
    const uint8_t* page_blob,     // Búfer completo de la página binaria
    int32_t* out,                 // [M, rows]
    int64_t M,
    int64_t K
) {
    const PagedHeader* hdr = (const PagedHeader*)page_blob;
    if (hdr->fmt_code != 1) {
        return; // Solo admite fast format (fmt_code == 1)
    }
    const uint32_t rows = hdr->rows;
    const uint16_t cols = hdr->cols;
    const uint8_t* payload = page_blob + sizeof(PagedHeader);

    int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;

    int64_t total_work = M * (int64_t)rows * (int64_t)cols;
    omp_set_num_threads(2);

    #pragma omp parallel for collapse(2) schedule(static) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t r = 0; r < (int64_t)rows; ++r) {
            const int8_t* x_row = x + m * K;
            const uint8_t* w_row_packed = payload + r * row_stride_bytes;
            out[m * rows + r] = avx2_dot_product_packed_fast(x_row, w_row_packed, cols);
        }
    }
}

/*
 * Kernel Fisionado Multi-Página: procesa todas las páginas del tensor en un único pase.
 */
void paged_direct_matmul_fast_c_multi(
    const int8_t* x,                  // [M, K]
    const uint8_t* const* page_blobs, // Array de punteros a cada página
    int64_t num_pages,
    int32_t* out,                     // [M, total_out_features]
    int64_t M,
    int64_t K,
    int64_t total_out_features
) {
    int64_t total_work = M * total_out_features * K;
    omp_set_num_threads(2);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t p = 0; p < num_pages; ++p) {
        const uint8_t* blob = page_blobs[p];
        const PagedHeader* hdr = (const PagedHeader*)blob;
        if (hdr->fmt_code != 1) {
            continue; // Omitir páginas que no correspondan a formato Fast
        }
        const uint32_t row_start = hdr->row_start;
        const uint32_t rows = hdr->rows;
        const uint16_t cols = hdr->cols;
        const uint8_t* payload = blob + sizeof(PagedHeader);

        int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
        int64_t row_stride_bytes = padded_cols >> 2;

        for (int64_t r = 0; r < (int64_t)rows; ++r) {
            const uint8_t* w_row_packed = payload + r * row_stride_bytes;
            if (M > 1 && cols <= 512) {
                int8_t w_unpacked[512] __attribute__((aligned(32)));
                int64_t k = 0;
                for (; k + 32 <= cols; k += 32) {
                    __m256i w_vec = avx2_unpack_32_weights_fast(w_row_packed + (k >> 2));
                    _mm256_storeu_si256((__m256i*)(w_unpacked + k), w_vec);
                }
                for (; k < cols; ++k) {
                    uint8_t byte_val = w_row_packed[k >> 2];
                    int shift = (k & 3) * 2;
                    w_unpacked[k] = (int8_t)(((int32_t)((byte_val >> shift) & 0x03)) - 1);
                }
                for (int64_t m = 0; m < M; ++m) {
                    const int8_t* x_row = x + m * K;
                    out[m * total_out_features + row_start + r] =
                        avx2_dot_product_dense_int8(x_row, w_unpacked, cols);
                }
            } else {
                for (int64_t m = 0; m < M; ++m) {
                    const int8_t* x_row = x + m * K;
                    int32_t* out_row = out + m * total_out_features + row_start;
                    out_row[r] = avx2_dot_product_packed_fast(x_row, w_row_packed, cols);
                }
            }
        }
    }
}

/*
 * Multiplica un vector x por una matriz de pesos 2-bit empaquetada.
 */
void paged_lowrank_matmul_row_c(
    const int8_t* x,          // [K]
    const uint8_t* w_packed,  // [rows * row_stride_bytes]
    int32_t* out,             // [rows]
    int64_t rows,
    int64_t K
) {
    int64_t padded_cols = (K + 31) / 32 * 32;
    int64_t row_stride_bytes = padded_cols >> 2;
    int64_t total_work = rows * K;
    omp_set_num_threads(2);

    #pragma omp parallel for schedule(static) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_w = w_packed + r * row_stride_bytes;
        out[r] = avx2_dot_product_packed_fast(x, row_w, K);
    }
}

/*
 * Kernel de multiplicación ternaria aditiva pura sobre matrices densas int8_t {-1, 0, +1}.
 * Implementado con AVX2 (vpsignb + maddubs + madd + add).
 */
void ternary_addsub_cpu_kernel(
    const int8_t* x,          // [M, K]
    const int8_t* w,          // [N, K]
    int32_t* out,             // [M, N]
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t total_work = M * N * K;
    omp_set_num_threads(2);

    #pragma omp parallel for collapse(2) schedule(static) num_threads(2) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const int8_t* x_row = x + m * K;
            const int8_t* w_row = w + n * K;
            out[m * N + n] = avx2_dot_product_dense_int8(x_row, w_row, K);
        }
    }
}

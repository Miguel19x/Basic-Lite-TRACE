/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#ifndef PAGED_UNPACK_H
#define PAGED_UNPACK_H

#include <stdint.h>

#ifndef PAGED_HEADER_DEFINED
#define PAGED_HEADER_DEFINED
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
#endif

#if defined(__AVX2__)
#include <immintrin.h>

// Desempaqueta 32 pesos ternarios contiguos codificados a 2-bit en formato fast
// Mapeo: 0 -> -1, 1 -> 0, 2 -> +1
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
#endif

#endif // PAGED_UNPACK_H

/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#ifndef TRACE_MATH_H
#define TRACE_MATH_H

#include <math.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>

static inline void rms_norm(const float* x, const float* w, float* y, int d, float eps) {
    if (!x || !y || d <= 0) return;
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256d d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(vx));
        __m256d d1 = _mm256_cvtps_pd(_mm256_extractf128_ps(vx, 1));
        s0 = _mm256_add_pd(s0, _mm256_mul_pd(d0, d0));
        s1 = _mm256_add_pd(s1, _mm256_mul_pd(d1, d1));
    }
    __m256d sum_d = _mm256_add_pd(s0, s1);
    __m128d lo = _mm256_castpd256_pd128(sum_d);
    __m128d hi = _mm256_extractf128_pd(sum_d, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    double sum_sq = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum128, sum128));
    for (; i < d; ++i) sum_sq += (double)x[i] * (double)x[i];

    double denom = sum_sq / (double)d + (double)eps;
    float rstd = (denom > 0.0) ? (float)(1.0 / sqrt(denom)) : 0.0f;
    __m256 vrstd = _mm256_set1_ps(rstd);

    int j = 0;
    if (w) {
        for (; j + 8 <= d; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vw = _mm256_loadu_ps(w + j);
            _mm256_storeu_ps(y + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrstd), vw));
        }
        for (; j < d; ++j) y[j] = x[j] * rstd * w[j];
    } else {
        for (; j + 8 <= d; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            _mm256_storeu_ps(y + j, _mm256_mul_ps(vx, vrstd));
        }
        for (; j < d; ++j) y[j] = x[j] * rstd;
    }
}

static inline float avx2_dot_product_f32(const float* a, const float* b, int n) {
    __m256 sum256 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(va, vb));
    }
    __m128 lo = _mm256_castps256_ps128(sum256);
    __m128 hi = _mm256_extractf128_ps(sum256, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ss(s128, _mm_shuffle_ps(s128, s128, 0x55));
    float acc = _mm_cvtss_f32(s128);

    for (; d < n; ++d) {
        acc += a[d] * b[d];
    }
    return acc;
}

#else

static inline void rms_norm(const float* x, const float* w, float* y, int d, float eps) {
    if (!x || !y || d <= 0) return;
    double sum_sq = 0.0;
    for (int i = 0; i < d; ++i) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    double denom = sum_sq / (double)d + (double)eps;
    float rstd = (denom > 0.0) ? (float)(1.0 / sqrt(denom)) : 0.0f;
    if (w) {
        for (int i = 0; i < d; ++i) y[i] = x[i] * rstd * w[i];
    } else {
        for (int i = 0; i < d; ++i) y[i] = x[i] * rstd;
    }
}

static inline float avx2_dot_product_f32(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

#endif

#endif // TRACE_MATH_H

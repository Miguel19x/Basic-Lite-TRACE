/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "paged_dispatch.h"
#include "trace_infer.h"
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define OMP_MIN_WORK_THRESHOLD 65536

static inline const uint8_t* paged_get_row_packed_scalar(
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

void paged_direct_matmul_fast_c_scalar(
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
            int32_t acc = 0;
            for (int64_t k = 0; k < (int64_t)cols; ++k) {
                uint8_t byte_val = w_row_packed[k >> 2];
                int shift = (k & 3) * 2;
                int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                acc += (int32_t)x_row[k] * w;
            }
            out[m * rows + r] = acc;
        }
    }
}

void paged_direct_matmul_fast_c_multi_scalar(
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
    for (int64_t p = 0; p < (int64_t)num_pages; ++p) {
        const uint8_t* blob = page_blobs[p];
        const PagedHeader* hdr = (const PagedHeader*)blob;
        if (hdr->fmt_code != 1) continue;

        const uint32_t row_start = hdr->row_start;
        const uint32_t rows = hdr->rows;
        const uint16_t cols = hdr->cols;
        const uint8_t* payload = blob + sizeof(PagedHeader);
        int64_t padded_cols = ((int64_t)cols + 31) / 32 * 32;
        int64_t row_stride_bytes = padded_cols >> 2;

        for (int64_t m = 0; m < M; ++m) {
            const int8_t* x_row = x + m * K;
            for (int64_t r = 0; r < (int64_t)rows; ++r) {
                const uint8_t* w_row_packed = payload + r * row_stride_bytes;
                int32_t acc = 0;
                for (int64_t k = 0; k < (int64_t)cols; ++k) {
                    uint8_t byte_val = w_row_packed[k >> 2];
                    int shift = (k & 3) * 2;
                    int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                    acc += (int32_t)x_row[k] * w;
                }
                out[m * total_out_features + row_start + r] = acc;
            }
        }
    }
}

void paged_lowrank_matmul_row_c_scalar(
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
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k) {
            uint8_t byte_val = row_w[k >> 2];
            int shift = (k & 3) * 2;
            int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
            acc += (int32_t)x[k] * w;
        }
        out[r] = acc;
    }
}

void ternary_addsub_cpu_kernel_scalar(
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
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                acc += (int32_t)x_row[k] * (int32_t)w_row[k];
            }
            out[m * N + n] = acc;
        }
    }
}

void paged_absorb_uk_f32_scalar(
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
                const uint8_t* row_p = paged_get_row_packed_scalar(page_ptrs, num_pages, r);
                if (!row_p) continue;

                float v = q_c[(t * n_heads + h) * qk_nope_head_dim + d];
                if (v == 0.0f) continue;

                for (int k = 0; k < kv_lora_rank; ++k) {
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

void paged_absorb_uv_f32_scalar(
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
                const uint8_t* row_p = paged_get_row_packed_scalar(page_ptrs, num_pages, r);
                if (!row_p) {
                    out_h[v] = 0.0f;
                    continue;
                }

                float dot = 0.0f;
                for (int k = 0; k < kv_lora_rank; ++k) {
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

float trace_dot_f32_scalar(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

void trace_axpy_f32_scalar(float* y, float a, const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

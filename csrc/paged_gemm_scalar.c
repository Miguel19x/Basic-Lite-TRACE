/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "paged_dispatch.h"
#include "trace_infer.h"
#include "trace_math.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
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
    if (!x || !page_blobs || !out || num_pages <= 0 || M <= 0 || K <= 0 || total_out_features <= 0) return;
    int64_t total_work = M * total_out_features * K;

    #pragma omp parallel for schedule(dynamic, 1) if (total_work >= OMP_MIN_WORK_THRESHOLD)
    for (int64_t p = 0; p < (int64_t)num_pages; ++p) {
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

    #pragma omp parallel for schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
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

    #pragma omp parallel for collapse(2) schedule(static) if (total_work >= OMP_MIN_WORK_THRESHOLD)
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

void trace_mla_attn_group_f32_scalar(
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
    for (int gh = 0; gh < h_count; ++gh) {
        int h = h_start + gh;
        const float* q_c_h = q_c_abs_base + (size_t)h * (size_t)kv_lora_rank;
        const float* q_pe_h = q_pe_base + (size_t)h * (size_t)qk_rope_head_dim;
        float* sc_h = scores_scratch + (size_t)gh * (size_t)total_len;

        float max_sc = -INFINITY;
        for (int s = 0; s < total_len; ++s) {
            const float* c_kv_s = c_kv_layer + (size_t)s * (size_t)kv_lora_rank;
            const float* k_pe_s = k_pe_layer + (size_t)s * (size_t)qk_rope_head_dim;
            float dot_c = trace_dot_f32_scalar(q_c_h, c_kv_s, kv_lora_rank);
            float dot_pe = trace_dot_f32_scalar(q_pe_h, k_pe_s, qk_rope_head_dim);
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
            trace_axpy_f32_scalar(o_lat_h, p, c_kv_s, kv_lora_rank);
        }
    }
}

float trace_dot_f32_f16_scalar(const float* x, const uint16_t* w_f16, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += x[i] * trace_f16_to_f32(w_f16[i]);
    }
    return acc;
}

void trace_matvec_f32_f16_scalar(const float* x, const uint16_t* w_f16, float* out, int rows, int cols) {
    #pragma omp parallel for schedule(static) if (rows >= 256)
    for (int v = 0; v < rows; ++v) {
        out[v] = trace_dot_f32_f16_scalar(x, w_f16 + (size_t)v * (size_t)cols, cols);
    }
}

float trace_dot_f32_i8_scalar(const float* x, const int8_t* w_i8, float scale, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += x[i] * (float)w_i8[i];
    }
    return acc * scale;
}

void trace_gqa_attn_group_f32_scalar(
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

    for (int gh = 0; gh < q_count; ++gh) {
        int qh = q_start + gh;
        const float* q_h = q_base + (size_t)qh * (size_t)head_dim;
        float* sc_h = scores_scratch + (size_t)gh * (size_t)total_len;

        float max_sc = -INFINITY;
        for (int s = 0; s < total_len; ++s) {
            const float* k_s = k_layer + (size_t)s * kv_stride + kv_off;
            float sc = trace_dot_f32_scalar(q_h, k_s, head_dim) * attn_scale;
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
            trace_axpy_f32_scalar(o_h, p, v_s, head_dim);
        }
    }
}

void trace_quantize_rows_int8_scalar(
    const float* in,
    const float* sub_ln_w,
    float* x_norm,
    int8_t* q_x,
    float* scale_x,
    int M,
    int in_f,
    float eps
) {
    #pragma omp parallel for schedule(static) if (M >= 16)
    for (int m = 0; m < M; ++m) {
        const float* in_row = in + (size_t)m * (size_t)in_f;
        const float* row = in_row;
        if (sub_ln_w != NULL) {
            float* norm_row = x_norm + (size_t)m * (size_t)in_f;
            rms_norm(in_row, sub_ln_w, norm_row, in_f, eps);
            row = norm_row;
        }
        float amax = eps;
        for (int i = 0; i < in_f; ++i) {
            float v = fabsf(row[i]);
            if (v > amax) amax = v;
        }
        float scale = amax / 127.0f;
        float scale_inv = 127.0f / amax;
        scale_x[m] = scale;
        int8_t* qrow = q_x + (size_t)m * (size_t)in_f;
        for (int i = 0; i < in_f; ++i) {
            float q = nearbyintf(row[i] * scale_inv);
            if (!isfinite(q)) q = 0.0f;
            else if (q > 127.0f) q = 127.0f;
            else if (q < -127.0f) q = -127.0f;
            qrow[i] = (int8_t)q;
        }
    }
}

void trace_dequantize_rows_i32_scalar(
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
        const int32_t* acc_row = acc + (size_t)m * (size_t)out_f;
        float* out_row = out + (size_t)m * (size_t)out_f;
        for (int i = 0; i < out_f; ++i) {
            out_row[i] = (float)acc_row[i] * factor;
        }
    }
}

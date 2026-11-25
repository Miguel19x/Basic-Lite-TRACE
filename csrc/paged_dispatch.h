/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#ifndef PAGED_DISPATCH_H
#define PAGED_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Firmas de punteros de función para el catálogo de kernels
typedef void (*paged_direct_matmul_fast_c_fn)(
    const int8_t* x,
    const uint8_t* page_blob,
    int32_t* out,
    int64_t M,
    int64_t K
);

typedef void (*paged_direct_matmul_fast_c_multi_fn)(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
);

typedef void (*paged_lowrank_matmul_row_c_fn)(
    const int8_t* x,
    const uint8_t* w_packed,
    int32_t* out,
    int64_t rows,
    int64_t K
);

typedef void (*ternary_addsub_cpu_kernel_fn)(
    const int8_t* x,
    const int8_t* w,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t N
);

typedef void (*paged_absorb_uk_f32_fn)(
    const float* q_c,
    const uint8_t* const* page_ptrs,
    int num_pages,
    float scale,
    float* q_c_abs,
    int n_heads,
    int qk_nope_head_dim,
    int kv_lora_rank,
    int S
);

typedef void (*paged_absorb_uv_f32_fn)(
    const float* o_latent,
    const uint8_t* const* page_ptrs,
    int num_pages,
    float scale,
    float* o_heads,
    int n_heads,
    int v_head_dim,
    int kv_lora_rank,
    int S
);

typedef float (*trace_dot_f32_fn)(
    const float* a,
    const float* b,
    int n
);

typedef void (*trace_axpy_f32_fn)(
    float* y,
    float a,
    const float* x,
    int n
);

typedef void (*trace_mla_attn_group_f32_fn)(
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
);

typedef float (*trace_dot_f32_f16_fn)(
    const float* x,
    const uint16_t* w_f16,
    int n
);

typedef float (*trace_dot_f32_i8_fn)(
    const float* x,
    const int8_t* w_i8,
    float scale,
    int n
);

typedef void (*trace_gqa_attn_group_f32_fn)(
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
);

// Funciones públicas exportadas con despacho dinámico transparente
void paged_direct_matmul_fast_c(
    const int8_t* x,
    const uint8_t* page_blob,
    int32_t* out,
    int64_t M,
    int64_t K
);

void paged_direct_matmul_fast_c_multi(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
);

void paged_lowrank_matmul_row_c(
    const int8_t* x,
    const uint8_t* w_packed,
    int32_t* out,
    int64_t rows,
    int64_t K
);

void ternary_addsub_cpu_kernel(
    const int8_t* x,
    const int8_t* w,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t N
);

void paged_absorb_uk_f32(
    const float* q_c,
    const uint8_t* const* page_ptrs,
    int num_pages,
    float scale,
    float* q_c_abs,
    int n_heads,
    int qk_nope_head_dim,
    int kv_lora_rank,
    int S
);

void paged_absorb_uv_f32(
    const float* o_latent,
    const uint8_t* const* page_ptrs,
    int num_pages,
    float scale,
    float* o_heads,
    int n_heads,
    int v_head_dim,
    int kv_lora_rank,
    int S
);

float trace_dot_f32(
    const float* a,
    const float* b,
    int n
);

void trace_axpy_f32(
    float* y,
    float a,
    const float* x,
    int n
);

void trace_mla_attn_group_f32(
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
);

float trace_dot_f32_f16(
    const float* x,
    const uint16_t* w_f16,
    int n
);

float trace_dot_f32_i8(
    const float* x,
    const int8_t* w_i8,
    float scale,
    int n
);

void trace_gqa_attn_group_f32(
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
);

// Consulta y configuración de ISA activa
const char* trace_cpu_active_isa(void);
void trace_cpu_set_isa(const char* isa_name);

#ifdef __cplusplus
}
#endif

#endif // PAGED_DISPATCH_H

/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "paged_dispatch.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Prototipos de implementaciones internas
// 1. Escalar C99
void paged_direct_matmul_fast_c_scalar(const int8_t* x, const uint8_t* page_blob, int32_t* out, int64_t M, int64_t K);
void paged_direct_matmul_fast_c_multi_scalar(const int8_t* x, const uint8_t* const* page_blobs, int64_t num_pages, int32_t* out, int64_t M, int64_t K, int64_t total_out_features);
void paged_lowrank_matmul_row_c_scalar(const int8_t* x, const uint8_t* w_packed, int32_t* out, int64_t rows, int64_t K);
void ternary_addsub_cpu_kernel_scalar(const int8_t* x, const int8_t* w, int32_t* out, int64_t M, int64_t K, int64_t N);
void paged_absorb_uk_f32_scalar(const float* q_c, const uint8_t* const* page_ptrs, int num_pages, float scale, float* q_c_abs, int n_heads, int qk_nope_head_dim, int kv_lora_rank, int S);
void paged_absorb_uv_f32_scalar(const float* o_latent, const uint8_t* const* page_ptrs, int num_pages, float scale, float* o_heads, int n_heads, int v_head_dim, int kv_lora_rank, int S);
float trace_dot_f32_scalar(const float* a, const float* b, int n);
void trace_axpy_f32_scalar(float* y, float a, const float* x, int n);
void trace_mla_attn_group_f32_scalar(const float* q_c_abs_base, const float* q_pe_base, const float* c_kv_layer, const float* k_pe_layer, float* o_lat_base, float* scores_scratch, int h_start, int h_count, int total_len, int kv_lora_rank, int qk_rope_head_dim, float attn_scale);
float trace_dot_f32_f16_scalar(const float* x, const uint16_t* w_f16, int n);
float trace_dot_f32_i8_scalar(const float* x, const int8_t* w_i8, float scale, int n);
void trace_gqa_attn_group_f32_scalar(const float* q_base, const float* k_layer, const float* v_layer, float* o_heads_base, float* scores_scratch, int kv_h, int q_start, int q_count, int n_kv_heads, int total_len, int head_dim, float attn_scale);

// 2. AVX2 + FMA + F16C
#if defined(TRACE_ENABLE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
void paged_direct_matmul_fast_c_avx2(const int8_t* x, const uint8_t* page_blob, int32_t* out, int64_t M, int64_t K);
void paged_direct_matmul_fast_c_multi_avx2(const int8_t* x, const uint8_t* const* page_blobs, int64_t num_pages, int32_t* out, int64_t M, int64_t K, int64_t total_out_features);
void paged_lowrank_matmul_row_c_avx2(const int8_t* x, const uint8_t* w_packed, int32_t* out, int64_t rows, int64_t K);
void ternary_addsub_cpu_kernel_avx2(const int8_t* x, const int8_t* w, int32_t* out, int64_t M, int64_t K, int64_t N);
void paged_absorb_uk_f32_avx2(const float* q_c, const uint8_t* const* page_ptrs, int num_pages, float scale, float* q_c_abs, int n_heads, int qk_nope_head_dim, int kv_lora_rank, int S);
void paged_absorb_uv_f32_avx2(const float* o_latent, const uint8_t* const* page_ptrs, int num_pages, float scale, float* o_heads, int n_heads, int v_head_dim, int kv_lora_rank, int S);
float trace_dot_f32_avx2(const float* a, const float* b, int n);
void trace_axpy_f32_avx2(float* y, float a, const float* x, int n);
void trace_mla_attn_group_f32_avx2(const float* q_c_abs_base, const float* q_pe_base, const float* c_kv_layer, const float* k_pe_layer, float* o_lat_base, float* scores_scratch, int h_start, int h_count, int total_len, int kv_lora_rank, int qk_rope_head_dim, float attn_scale);
float trace_dot_f32_f16_avx2(const float* x, const uint16_t* w_f16, int n);
float trace_dot_f32_i8_avx2(const float* x, const int8_t* w_i8, float scale, int n);
void trace_gqa_attn_group_f32_avx2(const float* q_base, const float* k_layer, const float* v_layer, float* o_heads_base, float* scores_scratch, int kv_h, int q_start, int q_count, int n_kv_heads, int total_len, int head_dim, float attn_scale);
#endif

typedef struct {
    const char* isa_name;
    paged_direct_matmul_fast_c_fn paged_direct_matmul_fast_c;
    paged_direct_matmul_fast_c_multi_fn paged_direct_matmul_fast_c_multi;
    paged_lowrank_matmul_row_c_fn paged_lowrank_matmul_row_c;
    ternary_addsub_cpu_kernel_fn ternary_addsub_cpu_kernel;
    paged_absorb_uk_f32_fn paged_absorb_uk_f32;
    paged_absorb_uv_f32_fn paged_absorb_uv_f32;
    trace_dot_f32_fn trace_dot_f32;
    trace_axpy_f32_fn trace_axpy_f32;
    trace_mla_attn_group_f32_fn trace_mla_attn_group_f32;
    trace_dot_f32_f16_fn trace_dot_f32_f16;
    trace_dot_f32_i8_fn trace_dot_f32_i8;
    trace_gqa_attn_group_f32_fn trace_gqa_attn_group_f32;
} TraceKernelTable;

static const TraceKernelTable g_scalar_kernels = {
    "scalar",
    paged_direct_matmul_fast_c_scalar,
    paged_direct_matmul_fast_c_multi_scalar,
    paged_lowrank_matmul_row_c_scalar,
    ternary_addsub_cpu_kernel_scalar,
    paged_absorb_uk_f32_scalar,
    paged_absorb_uv_f32_scalar,
    trace_dot_f32_scalar,
    trace_axpy_f32_scalar,
    trace_mla_attn_group_f32_scalar,
    trace_dot_f32_f16_scalar,
    trace_dot_f32_i8_scalar,
    trace_gqa_attn_group_f32_scalar
};

#if defined(TRACE_ENABLE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
static const TraceKernelTable g_avx2_kernels = {
    "avx2",
    paged_direct_matmul_fast_c_avx2,
    paged_direct_matmul_fast_c_multi_avx2,
    paged_lowrank_matmul_row_c_avx2,
    ternary_addsub_cpu_kernel_avx2,
    paged_absorb_uk_f32_avx2,
    paged_absorb_uv_f32_avx2,
    trace_dot_f32_avx2,
    trace_axpy_f32_avx2,
    trace_mla_attn_group_f32_avx2,
    trace_dot_f32_f16_avx2,
    trace_dot_f32_i8_avx2,
    trace_gqa_attn_group_f32_avx2
};
#endif

static _Atomic(const TraceKernelTable*) g_active_table = NULL;

static const TraceKernelTable* select_autodetect_table(void) {
#if defined(TRACE_ENABLE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        return &g_avx2_kernels;
    }
#endif
    return &g_scalar_kernels;
}

static const TraceKernelTable* trace_cpu_init_dispatch(void) {
    const TraceKernelTable* tbl = atomic_load_explicit(&g_active_table, memory_order_acquire);
    if (tbl) return tbl;

    const char* env_isa = getenv("TRACE_CPU_ISA");
    if (env_isa && env_isa[0] != '\0' && strcmp(env_isa, "auto") != 0) {
        trace_cpu_set_isa(env_isa);
        return atomic_load_explicit(&g_active_table, memory_order_acquire);
    }

    const TraceKernelTable* detected = select_autodetect_table();
    atomic_store_explicit(&g_active_table, detected, memory_order_release);
    return detected;
}

void trace_cpu_set_isa(const char* isa_name) {
    if (!isa_name || strcmp(isa_name, "auto") == 0) {
        atomic_store_explicit(&g_active_table, select_autodetect_table(), memory_order_release);
        return;
    }

    if (strcmp(isa_name, "scalar") == 0) {
        atomic_store_explicit(&g_active_table, &g_scalar_kernels, memory_order_release);
        return;
    }

#if defined(TRACE_ENABLE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    if (strcmp(isa_name, "avx2") == 0 &&
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        atomic_store_explicit(&g_active_table, &g_avx2_kernels, memory_order_release);
        return;
    }
#endif

    // Fallback a escalar si la ISA solicitada no está compilada o soportada en este target
    atomic_store_explicit(&g_active_table, &g_scalar_kernels, memory_order_release);
}

__attribute__((constructor))
static void trace_cpu_auto_init(void) {
    trace_cpu_init_dispatch();
}

const char* trace_cpu_active_isa(void) {
    return trace_cpu_init_dispatch()->isa_name;
}

void paged_direct_matmul_fast_c(const int8_t* x, const uint8_t* page_blob, int32_t* out, int64_t M, int64_t K) {
    trace_cpu_init_dispatch()->paged_direct_matmul_fast_c(x, page_blob, out, M, K);
}

void paged_direct_matmul_fast_c_multi(const int8_t* x, const uint8_t* const* page_blobs, int64_t num_pages, int32_t* out, int64_t M, int64_t K, int64_t total_out_features) {
    trace_cpu_init_dispatch()->paged_direct_matmul_fast_c_multi(x, page_blobs, num_pages, out, M, K, total_out_features);
}

void paged_lowrank_matmul_row_c(const int8_t* x, const uint8_t* w_packed, int32_t* out, int64_t rows, int64_t K) {
    trace_cpu_init_dispatch()->paged_lowrank_matmul_row_c(x, w_packed, out, rows, K);
}

void ternary_addsub_cpu_kernel(const int8_t* x, const int8_t* w, int32_t* out, int64_t M, int64_t K, int64_t N) {
    trace_cpu_init_dispatch()->ternary_addsub_cpu_kernel(x, w, out, M, K, N);
}

void paged_absorb_uk_f32(const float* q_c, const uint8_t* const* page_ptrs, int num_pages, float scale, float* q_c_abs, int n_heads, int qk_nope_head_dim, int kv_lora_rank, int S) {
    trace_cpu_init_dispatch()->paged_absorb_uk_f32(q_c, page_ptrs, num_pages, scale, q_c_abs, n_heads, qk_nope_head_dim, kv_lora_rank, S);
}

void paged_absorb_uv_f32(const float* o_latent, const uint8_t* const* page_ptrs, int num_pages, float scale, float* o_heads, int n_heads, int v_head_dim, int kv_lora_rank, int S) {
    trace_cpu_init_dispatch()->paged_absorb_uv_f32(o_latent, page_ptrs, num_pages, scale, o_heads, n_heads, v_head_dim, kv_lora_rank, S);
}

float trace_dot_f32(const float* a, const float* b, int n) {
    return trace_cpu_init_dispatch()->trace_dot_f32(a, b, n);
}

void trace_axpy_f32(float* y, float a, const float* x, int n) {
    trace_cpu_init_dispatch()->trace_axpy_f32(y, a, x, n);
}

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
) {
    trace_cpu_init_dispatch()->trace_mla_attn_group_f32(
        q_c_abs_base, q_pe_base, c_kv_layer, k_pe_layer, o_lat_base,
        scores_scratch, h_start, h_count, total_len, kv_lora_rank,
        qk_rope_head_dim, attn_scale
    );
}

float trace_dot_f32_f16(const float* x, const uint16_t* w_f16, int n) {
    return trace_cpu_init_dispatch()->trace_dot_f32_f16(x, w_f16, n);
}

float trace_dot_f32_i8(const float* x, const int8_t* w_i8, float scale, int n) {
    return trace_cpu_init_dispatch()->trace_dot_f32_i8(x, w_i8, scale, n);
}

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
) {
    trace_cpu_init_dispatch()->trace_gqa_attn_group_f32(
        q_base, k_layer, v_layer, o_heads_base, scores_scratch,
        kv_h, q_start, q_count, n_kv_heads, total_len, head_dim, attn_scale
    );
}

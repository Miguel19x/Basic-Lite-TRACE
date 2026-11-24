/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "paged_dispatch.h"
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

// 2. AVX2 + FMA
#if defined(TRACE_ENABLE_AVX2)
void paged_direct_matmul_fast_c_avx2(const int8_t* x, const uint8_t* page_blob, int32_t* out, int64_t M, int64_t K);
void paged_direct_matmul_fast_c_multi_avx2(const int8_t* x, const uint8_t* const* page_blobs, int64_t num_pages, int32_t* out, int64_t M, int64_t K, int64_t total_out_features);
void paged_lowrank_matmul_row_c_avx2(const int8_t* x, const uint8_t* w_packed, int32_t* out, int64_t rows, int64_t K);
void ternary_addsub_cpu_kernel_avx2(const int8_t* x, const int8_t* w, int32_t* out, int64_t M, int64_t K, int64_t N);
void paged_absorb_uk_f32_avx2(const float* q_c, const uint8_t* const* page_ptrs, int num_pages, float scale, float* q_c_abs, int n_heads, int qk_nope_head_dim, int kv_lora_rank, int S);
void paged_absorb_uv_f32_avx2(const float* o_latent, const uint8_t* const* page_ptrs, int num_pages, float scale, float* o_heads, int n_heads, int v_head_dim, int kv_lora_rank, int S);
float trace_dot_f32_avx2(const float* a, const float* b, int n);
void trace_axpy_f32_avx2(float* y, float a, const float* x, int n);
#endif

// Tabla de punteros activos
static paged_direct_matmul_fast_c_fn g_paged_direct_matmul_fast_c = paged_direct_matmul_fast_c_scalar;
static paged_direct_matmul_fast_c_multi_fn g_paged_direct_matmul_fast_c_multi = paged_direct_matmul_fast_c_multi_scalar;
static paged_lowrank_matmul_row_c_fn g_paged_lowrank_matmul_row_c = paged_lowrank_matmul_row_c_scalar;
static ternary_addsub_cpu_kernel_fn g_ternary_addsub_cpu_kernel = ternary_addsub_cpu_kernel_scalar;
static paged_absorb_uk_f32_fn g_paged_absorb_uk_f32 = paged_absorb_uk_f32_scalar;
static paged_absorb_uv_f32_fn g_paged_absorb_uv_f32 = paged_absorb_uv_f32_scalar;
static trace_dot_f32_fn g_trace_dot_f32 = trace_dot_f32_scalar;
static trace_axpy_f32_fn g_trace_axpy_f32 = trace_axpy_f32_scalar;

static char g_active_isa[16] = "scalar";
static int g_initialized = 0;

static void trace_cpu_autodetect(void) {
#if defined(TRACE_ENABLE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        trace_cpu_set_isa("avx2");
        return;
    }
#endif
    trace_cpu_set_isa("scalar");
}

static void trace_cpu_init_dispatch(void) {
    if (g_initialized) return;
    g_initialized = 1;

    const char* env_isa = getenv("TRACE_CPU_ISA");
    if (env_isa && strlen(env_isa) > 0 && strcmp(env_isa, "auto") != 0) {
        trace_cpu_set_isa(env_isa);
        return;
    }

    trace_cpu_autodetect();
}

void trace_cpu_set_isa(const char* isa_name) {
    if (!isa_name || strcmp(isa_name, "auto") == 0) {
        g_initialized = 1;
        trace_cpu_autodetect();
        return;
    }

    if (strcmp(isa_name, "scalar") == 0) {
        g_paged_direct_matmul_fast_c = paged_direct_matmul_fast_c_scalar;
        g_paged_direct_matmul_fast_c_multi = paged_direct_matmul_fast_c_multi_scalar;
        g_paged_lowrank_matmul_row_c = paged_lowrank_matmul_row_c_scalar;
        g_ternary_addsub_cpu_kernel = ternary_addsub_cpu_kernel_scalar;
        g_paged_absorb_uk_f32 = paged_absorb_uk_f32_scalar;
        g_paged_absorb_uv_f32 = paged_absorb_uv_f32_scalar;
        g_trace_dot_f32 = trace_dot_f32_scalar;
        g_trace_axpy_f32 = trace_axpy_f32_scalar;
        snprintf(g_active_isa, sizeof(g_active_isa), "%s", "scalar");
        g_initialized = 1;
        return;
    }

#if defined(TRACE_ENABLE_AVX2)
    if (strcmp(isa_name, "avx2") == 0) {
        g_paged_direct_matmul_fast_c = paged_direct_matmul_fast_c_avx2;
        g_paged_direct_matmul_fast_c_multi = paged_direct_matmul_fast_c_multi_avx2;
        g_paged_lowrank_matmul_row_c = paged_lowrank_matmul_row_c_avx2;
        g_ternary_addsub_cpu_kernel = ternary_addsub_cpu_kernel_avx2;
        g_paged_absorb_uk_f32 = paged_absorb_uk_f32_avx2;
        g_paged_absorb_uv_f32 = paged_absorb_uv_f32_avx2;
        g_trace_dot_f32 = trace_dot_f32_avx2;
        g_trace_axpy_f32 = trace_axpy_f32_avx2;
        snprintf(g_active_isa, sizeof(g_active_isa), "%s", "avx2");
        g_initialized = 1;
        return;
    }
#endif

    // Fallback a escalar si la ISA solicitada no está compilada en este target
    g_paged_direct_matmul_fast_c = paged_direct_matmul_fast_c_scalar;
    g_paged_direct_matmul_fast_c_multi = paged_direct_matmul_fast_c_multi_scalar;
    g_paged_lowrank_matmul_row_c = paged_lowrank_matmul_row_c_scalar;
    g_ternary_addsub_cpu_kernel = ternary_addsub_cpu_kernel_scalar;
    g_paged_absorb_uk_f32 = paged_absorb_uk_f32_scalar;
    g_paged_absorb_uv_f32 = paged_absorb_uv_f32_scalar;
    g_trace_dot_f32 = trace_dot_f32_scalar;
    g_trace_axpy_f32 = trace_axpy_f32_scalar;
    snprintf(g_active_isa, sizeof(g_active_isa), "%s", "scalar");
    g_initialized = 1;
}

__attribute__((constructor))
static void trace_cpu_auto_init(void) {
    trace_cpu_init_dispatch();
}

const char* trace_cpu_active_isa(void) {
    trace_cpu_init_dispatch();
    return g_active_isa;
}

void paged_direct_matmul_fast_c(const int8_t* x, const uint8_t* page_blob, int32_t* out, int64_t M, int64_t K) {
    trace_cpu_init_dispatch();
    g_paged_direct_matmul_fast_c(x, page_blob, out, M, K);
}

void paged_direct_matmul_fast_c_multi(const int8_t* x, const uint8_t* const* page_blobs, int64_t num_pages, int32_t* out, int64_t M, int64_t K, int64_t total_out_features) {
    trace_cpu_init_dispatch();
    g_paged_direct_matmul_fast_c_multi(x, page_blobs, num_pages, out, M, K, total_out_features);
}

void paged_lowrank_matmul_row_c(const int8_t* x, const uint8_t* w_packed, int32_t* out, int64_t rows, int64_t K) {
    trace_cpu_init_dispatch();
    g_paged_lowrank_matmul_row_c(x, w_packed, out, rows, K);
}

void ternary_addsub_cpu_kernel(const int8_t* x, const int8_t* w, int32_t* out, int64_t M, int64_t K, int64_t N) {
    trace_cpu_init_dispatch();
    g_ternary_addsub_cpu_kernel(x, w, out, M, K, N);
}

void paged_absorb_uk_f32(const float* q_c, const uint8_t* const* page_ptrs, int num_pages, float scale, float* q_c_abs, int n_heads, int qk_nope_head_dim, int kv_lora_rank, int S) {
    trace_cpu_init_dispatch();
    g_paged_absorb_uk_f32(q_c, page_ptrs, num_pages, scale, q_c_abs, n_heads, qk_nope_head_dim, kv_lora_rank, S);
}

void paged_absorb_uv_f32(const float* o_latent, const uint8_t* const* page_ptrs, int num_pages, float scale, float* o_heads, int n_heads, int v_head_dim, int kv_lora_rank, int S) {
    trace_cpu_init_dispatch();
    g_paged_absorb_uv_f32(o_latent, page_ptrs, num_pages, scale, o_heads, n_heads, v_head_dim, kv_lora_rank, S);
}

float trace_dot_f32(const float* a, const float* b, int n) {
    trace_cpu_init_dispatch();
    return g_trace_dot_f32(a, b, n);
}

void trace_axpy_f32(float* y, float a, const float* x, int n) {
    trace_cpu_init_dispatch();
    g_trace_axpy_f32(y, a, x, n);
}

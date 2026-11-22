#define _GNU_SOURCE
#include "trace_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <fenv.h>
#include <omp.h>
#include <immintrin.h>
#include <time.h>

#ifdef TRACE_PROFILE
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t g_prof_total_ns = 0;
static uint64_t g_prof_prequant_ns = 0;
static uint64_t g_prof_gemm_ns = 0;
static uint64_t g_prof_dequant_ns = 0;
static uint64_t g_prof_sdpa_ns = 0;
static uint64_t g_prof_rope_qknorm_ns = 0;
static uint64_t g_prof_moe_router_ns = 0;
static uint64_t g_prof_silu_updater_ns = 0;
static uint64_t g_prof_lm_head_ns = 0;
static uint64_t g_prof_call_count = 0;

void trace_model_reset_profile(void) {
    g_prof_total_ns = 0;
    g_prof_prequant_ns = 0;
    g_prof_gemm_ns = 0;
    g_prof_dequant_ns = 0;
    g_prof_sdpa_ns = 0;
    g_prof_rope_qknorm_ns = 0;
    g_prof_moe_router_ns = 0;
    g_prof_silu_updater_ns = 0;
    g_prof_lm_head_ns = 0;
    g_prof_call_count = 0;
}

void trace_model_print_profile(void) {
    if (g_prof_call_count == 0 || g_prof_total_ns == 0) {
        printf("  [Perfilado]: Sin datos registrados (0 llamadas)\n");
        return;
    }
    double total_ms = (double)g_prof_total_ns / 1e6;
    double calls = (double)g_prof_call_count;
    double prequant_ms = (double)g_prof_prequant_ns / 1e6;
    double gemm_ms = (double)g_prof_gemm_ns / 1e6;
    double dequant_ms = (double)g_prof_dequant_ns / 1e6;
    double sdpa_ms = (double)g_prof_sdpa_ns / 1e6;
    double rope_ms = (double)g_prof_rope_qknorm_ns / 1e6;
    double moe_ms = (double)g_prof_moe_router_ns / 1e6;
    double silu_ms = (double)g_prof_silu_updater_ns / 1e6;
    double lm_ms = (double)g_prof_lm_head_ns / 1e6;

    double sum_blocks_ms = prequant_ms + gemm_ms + dequant_ms + sdpa_ms + rope_ms + moe_ms + silu_ms + lm_ms;
    double otros_ms = total_ms - sum_blocks_ms;
    if (otros_ms < 0.0) otros_ms = 0.0;

    printf("\n"
           "  +----------------------+---------------+---------------+-----------+\n"
           "  | Bloque / Componente  | Tiempo Total  | Tiempo / Mues | Porcentaje|\n"
           "  +----------------------+---------------+---------------+-----------+\n"
           "  | GEMM (Kernel AVX2)   | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | Pre-Cuant (RMS+AVX2) | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | De-Cuantización AVX2 | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | SDPA (Atención AVX2) | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | RoPE + Q/K RMSNorm   | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | Router MoE (Top-K)   | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | SiLU + AnswerUpdater | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | lm_head (Lazy AVX2)  | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  | Otros / Resto        | %8.2f ms | %8.2f ms | %8.1f%% |\n"
           "  +----------------------+---------------+---------------+-----------+\n"
           "  | TOTAL PARED (Wall)   | %8.2f ms | %8.2f ms |   100.0%% |\n"
           "  +----------------------+---------------+---------------+-----------+\n",
           gemm_ms, gemm_ms / calls, (gemm_ms / total_ms) * 100.0,
           prequant_ms, prequant_ms / calls, (prequant_ms / total_ms) * 100.0,
           dequant_ms, dequant_ms / calls, (dequant_ms / total_ms) * 100.0,
           sdpa_ms, sdpa_ms / calls, (sdpa_ms / total_ms) * 100.0,
           rope_ms, rope_ms / calls, (rope_ms / total_ms) * 100.0,
           moe_ms, moe_ms / calls, (moe_ms / total_ms) * 100.0,
           silu_ms, silu_ms / calls, (silu_ms / total_ms) * 100.0,
           lm_ms, lm_ms / calls, (lm_ms / total_ms) * 100.0,
           otros_ms, otros_ms / calls, (otros_ms / total_ms) * 100.0,
           total_ms, total_ms / calls);
}

void trace_model_print_profile_report(void) {
    trace_model_print_profile();
}
#else
void trace_model_reset_profile(void) {}
void trace_model_print_profile(void) {}
void trace_model_print_profile_report(void) {}
#endif

#pragma pack(push, 1)
typedef struct {
    uint32_t row_start;
    uint32_t rows;
    uint16_t cols;
    uint16_t fmt_code;
    float    scale_w;
    uint32_t len_p1;
    uint32_t len_p2;
} PagedHeader;
#pragma pack(pop)

// Firma exacta de paged_gemm_kernel.c (7 argumentos con int64_t)
void paged_direct_matmul_fast_c_multi(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
);

size_t trace_model_sizeof(void) { return sizeof(TraceModel); }
int trace_model_get_used_odirect(const TraceModel* model) { return model ? model->used_odirect : 0; }
uint64_t trace_model_get_pread_calls(const TraceModel* model) { return model ? model->pread_calls : 0; }
size_t trace_model_get_pages_offset(const TraceModel* model) { return model ? model->pages_offset : 0; }

static inline void rms_norm(const float* x, const float* w, float* y, int d, float eps) {
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

    float rstd = (float)(1.0 / sqrt(sum_sq / (double)d + (double)eps));
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

// BitLinear forward con nearbyintf() (round-to-nearest-even idéntico a torch.round)
static void paged_linear_forward(
    const TracePagedLinear* layer,
    const float* x,     // [M, in_features]
    float* y,           // [M, out_features]
    int64_t M,
    float eps,
    TraceWorkspace* ws
) {
    if (M <= 0) return;
    int64_t in_f = layer->in_features;
    int64_t out_f = layer->out_features;
    float q_max = 127.0f;

    float* x_norm = ws ? ws->x_norm : (float*)malloc(M * in_f * sizeof(float));
    int8_t* q_x = ws ? ws->q_x : (int8_t*)malloc(M * in_f * sizeof(int8_t));
    int32_t* acc_int32 = ws ? ws->acc_int32 : (int32_t*)malloc(M * out_f * sizeof(int32_t));
    float* scale_x = ws ? ws->scale_x : (float*)malloc(M * sizeof(float));

    __m256 pos127 = _mm256_set1_ps(127.0f);
    __m256 neg127 = _mm256_set1_ps(-127.0f);

#ifdef TRACE_PROFILE
    uint64_t t_pre0 = get_time_ns();
#endif
    #pragma omp parallel for num_threads(2) schedule(static) if (M >= 32)
    for (int64_t m = 0; m < M; ++m) {
        rms_norm(x + m * in_f, layer->sub_ln_w, x_norm + m * in_f, (int)in_f, eps);

        // 1. amax vectorizado con AVX2
        float amax = 0.0f;
        __m256 max_vec = _mm256_setzero_ps();
        const float* xn_ptr = x_norm + m * in_f;
        int64_t k = 0;
        for (; k + 8 <= in_f; k += 8) {
            __m256 v = _mm256_loadu_ps(xn_ptr + k);
            __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            max_vec = _mm256_max_ps(max_vec, abs_v);
        }
        float max_arr[8];
        _mm256_storeu_ps(max_arr, max_vec);
        for (int i = 0; i < 8; ++i) {
            if (max_arr[i] > amax) amax = max_arr[i];
        }
        for (; k < in_f; ++k) {
            float val = fabsf(xn_ptr[k]);
            if (val > amax) amax = val;
        }
        if (amax < eps) amax = eps;
        scale_x[m] = amax;

        // 2. Cuantización vectorizada pura en registros AVX2
        float inv_s = q_max / amax;
        __m256 inv_s_vec = _mm256_set1_ps(inv_s);
        int64_t k_blk = 0;
        for (; k_blk + 8 <= in_f; k_blk += 8) {
            __m256 vx = _mm256_loadu_ps(xn_ptr + k_blk);
            __m256 v_scaled = _mm256_mul_ps(vx, inv_s_vec);
            __m256 v_rounded = _mm256_round_ps(v_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 v_clamped = _mm256_min_ps(_mm256_max_ps(v_rounded, neg127), pos127);
            __m256i vi32 = _mm256_cvtps_epi32(v_clamped);
            __m128i vi16 = _mm_packs_epi32(_mm256_castsi256_si128(vi32), _mm256_extracti128_si256(vi32, 1));
            __m128i vi8 = _mm_packs_epi16(vi16, vi16);
            _mm_storel_epi64((__m128i*)(q_x + m * in_f + k_blk), vi8);
        }
        for (; k_blk < in_f; ++k_blk) {
            float v = nearbyintf(xn_ptr[k_blk] * inv_s);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            q_x[m * in_f + k_blk] = (int8_t)v;
        }
    }
#ifdef TRACE_PROFILE
    uint64_t t_pre1 = get_time_ns();
    g_prof_prequant_ns += (t_pre1 - t_pre0);

    // Llamada con los 7 argumentos exactos
    uint64_t t_gemm0 = get_time_ns();
#endif
    paged_direct_matmul_fast_c_multi(
        q_x,
        layer->page_ptrs,
        (int64_t)layer->num_pages,
        acc_int32,
        M,
        in_f,
        out_f
    );
#ifdef TRACE_PROFILE
    uint64_t t_gemm1 = get_time_ns();
    g_prof_gemm_ns += (t_gemm1 - t_gemm0);

    // 3. De-cuantización vectorizada con AVX2 en 1 hilo (sin sobrecarga de barrera)
    uint64_t t_deq0 = get_time_ns();
#endif
    for (int64_t m = 0; m < M; ++m) {
        float factor = (scale_x[m] / q_max) * layer->scale;
        __m256 factor_vec = _mm256_set1_ps(factor);
        int64_t j = 0;
        for (; j + 8 <= out_f; j += 8) {
            __m256i acc_i = _mm256_loadu_si256((const __m256i*)(acc_int32 + m * out_f + j));
            __m256 acc_f = _mm256_cvtepi32_ps(acc_i);
            _mm256_storeu_ps(y + m * out_f + j, _mm256_mul_ps(acc_f, factor_vec));
        }
        for (; j < out_f; ++j) {
            y[m * out_f + j] = (float)acc_int32[m * out_f + j] * factor;
        }
    }
#ifdef TRACE_PROFILE
    uint64_t t_deq1 = get_time_ns();
    g_prof_dequant_ns += (t_deq1 - t_deq0);
#endif

    if (!ws) {
        free(x_norm);
        free(q_x);
        free(acc_int32);
        free(scale_x);
    }
}

static inline float avx2_dot_product_48(const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps();
    for (int d = 0; d < 48; d += 8) {
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

static inline float avx2_dot_product_192(const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps();
    for (int d = 0; d < 192; d += 8) {
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

static int init_workspace(TraceWorkspace* ws, int max_seq_len, int d_model, int d_ff_expert) {
    int max_dim = (d_ff_expert > d_model) ? d_ff_expert : d_model;
    int err = 0;
    #define ALIGN_ALLOC(ptr, sz) do { \
        if (posix_memalign((void**)&(ptr), 64, (sz)) != 0) err = -1; \
    } while(0)

    ALIGN_ALLOC(ws->x_norm, max_seq_len * max_dim * sizeof(float));
    ALIGN_ALLOC(ws->q_x, max_seq_len * max_dim * sizeof(int8_t));
    ALIGN_ALLOC(ws->acc_int32, max_seq_len * max_dim * sizeof(int32_t));
    ALIGN_ALLOC(ws->scale_x, max_seq_len * sizeof(float));

    ALIGN_ALLOC(ws->a, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->z_r, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->y_r, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->anchor_feat, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->joint_state, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->norm_z_buf, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->norm_y_buf, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->e_r, d_model * sizeof(float));
    ALIGN_ALLOC(ws->e_t, d_model * sizeof(float));
    ALIGN_ALLOC(ws->q, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->k, max_seq_len * (d_model / 2) * sizeof(float));
    ALIGN_ALLOC(ws->v, max_seq_len * (d_model / 2) * sizeof(float));
    ALIGN_ALLOC(ws->attn_out_heads, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->attn_final, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->z_mid, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->z_norm_moe, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->moe_out, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->shared_up_buf, max_seq_len * d_ff_expert * sizeof(float));
    ALIGN_ALLOC(ws->shared_down_buf, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->fusion, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->delta_y, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->z_next, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->normed_y, max_seq_len * d_model * sizeof(float));

    ALIGN_ALLOC(ws->exp_tokens_in, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->exp_up_buf, max_seq_len * d_ff_expert * sizeof(float));
    ALIGN_ALLOC(ws->exp_down_buf, max_seq_len * d_model * sizeof(float));
    ALIGN_ALLOC(ws->exp_tok_idx, max_seq_len * sizeof(int));
    ALIGN_ALLOC(ws->exp_tok_gate, max_seq_len * sizeof(float));

    #undef ALIGN_ALLOC
    return err;
}

static void free_workspace(TraceWorkspace* ws) {
    if (!ws) return;
    free(ws->x_norm); free(ws->q_x); free(ws->acc_int32); free(ws->scale_x);
    free(ws->a); free(ws->z_r); free(ws->y_r); free(ws->anchor_feat);
    free(ws->joint_state); free(ws->norm_z_buf); free(ws->norm_y_buf);
    free(ws->e_r); free(ws->e_t); free(ws->q); free(ws->k); free(ws->v);
    free(ws->attn_out_heads); free(ws->attn_final); free(ws->z_mid);
    free(ws->z_norm_moe); free(ws->moe_out); free(ws->shared_up_buf);
    free(ws->shared_down_buf); free(ws->fusion); free(ws->delta_y);
    free(ws->z_next); free(ws->normed_y);
    free(ws->exp_tokens_in); free(ws->exp_up_buf); free(ws->exp_down_buf);
    free(ws->exp_tok_idx); free(ws->exp_tok_gate);
    memset(ws, 0, sizeof(TraceWorkspace));
}

// Escáner de claves JSON embebidas
static const char* scan_json_key(const char* start, const char* end, const char* key) {
    size_t klen = strlen(key);
    const char* p = start;
    while (p + klen + 3 < end) {
        if (*p == '"' && strncmp(p + 1, key, klen) == 0 && p[klen + 1] == '"') {
            const char* colon = p + klen + 2;
            while (colon < end && (*colon == ' ' || *colon == '\t' || *colon == ':')) colon++;
            return colon;
        }
        p++;
    }
    return NULL;
}

// Carga O_DIRECT dinámica sin constantes hardcodeadas
int trace_model_load_odirect(const char* bin_path, TraceModel* model) {
    memset(model, 0, sizeof(TraceModel));
    omp_set_num_threads(2);

    int fd = open(bin_path, O_RDONLY | O_DIRECT);
    int used_odirect = 1;
    if (fd < 0) {
        fd = open(bin_path, O_RDONLY);
        used_odirect = 0;
    }
    if (fd < 0) return -1;

    model->fd = fd;
    model->used_odirect = used_odirect;
    model->pread_calls = 0;

    void* scratch = NULL;
    if (posix_memalign(&scratch, 4096, 4096) != 0) {
        close(fd);
        return -2;
    }

    ssize_t r = pread(fd, scratch, 4096, 0);
    model->pread_calls++;
    if (r < 64) {
        free(scratch);
        close(fd);
        return -3;
    }

    const uint8_t* p = (const uint8_t*)scratch;
    if (memcmp(p, "PBITCOS1", 8) != 0) {
        free(scratch);
        close(fd);
        return -4;
    }

    uint32_t page_sz, fmt_code, total_pages;
    uint64_t idx_off, exp_off, res_off, pages_off;
    memcpy(&page_sz, p + 8, 4);
    memcpy(&fmt_code, p + 12, 4);
    memcpy(&total_pages, p + 16, 4);
    memcpy(&idx_off, p + 20, 8);
    memcpy(&exp_off, p + 28, 8);
    memcpy(&res_off, p + 36, 8);
    memcpy(&pages_off, p + 44, 8);

    model->num_pages = total_pages;
    model->page_size = page_sz;
    model->pages_offset = pages_off;
    free(scratch);

    // Configuración base de TRACE
    model->cfg.vocab_size = 260;
    model->cfg.d_model = 192;
    model->cfg.n_heads = 4;
    model->cfg.n_kv_heads = 2;
    model->cfg.head_dim = 48;
    model->cfg.d_ff_expert = 384;
    model->cfg.num_experts = 4;
    model->cfg.num_recursions = 2;
    model->cfg.block_size = 16;
    model->cfg.max_seq_len = 160;
    model->cfg.eps = 1e-6f;
    model->cfg.pad_token_id = 256;
    model->cfg.mask_token_id = 257;
    model->cfg.bos_token_id = 258;
    model->cfg.eos_token_id = 259;

    // Precomputar tabla RoPE estática (160 * 24)
    for (int i = 0; i < 160; ++i) {
        for (int j = 0; j < 24; ++j) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * j) / 48.0f);
            model->rope_cos[i * 24 + j] = cosf((float)i * freq);
            model->rope_sin[i * 24 + j] = sinf((float)i * freq);
        }
    }

    if (init_workspace(&model->workspace, 160, 192, 384) != 0) {
        trace_model_free(model);
        return -5;
    }

    // Asignar Page Pool alineado en RAM
    size_t pool_size = (size_t)total_pages * page_sz;
    if (posix_memalign((void**)&model->page_pool, 4096, pool_size) != 0) {
        close(fd);
        return -5;
    }

    // Carga de páginas ternarias mediante O_DIRECT
    for (size_t pid = 0; pid < total_pages; ++pid) {
        size_t off = pages_off + pid * page_sz;
        ssize_t n = pread(fd, model->page_pool + pid * page_sz, page_sz, (off_t)off);
        model->pread_calls++;
        if (n != (ssize_t)page_sz) {
            free(model->page_pool);
            close(fd);
            return -6;
        }
    }

    // Descriptor para leer metadatos e índice JSON
    int fd_meta = open(bin_path, O_RDONLY);
    if (fd_meta < 0) {
        free(model->page_pool);
        close(fd);
        return -7;
    }

    size_t json_len = exp_off - idx_off;
    char* json_buf = (char*)malloc(json_len + 1);
    ssize_t meta_read = pread(fd_meta, json_buf, json_len, (off_t)idx_off);
    if (meta_read != (ssize_t)json_len) {
        free(json_buf);
        free(model->page_pool);
        close(fd_meta);
        close(fd);
        return -8;
    }
    json_buf[json_len] = '\0';

    int mapped_count = 0;
    const char* curr = json_buf;
    const char* json_end = json_buf + json_len;

    // Parseo dinámico del catálogo JSON
    while ((curr = scan_json_key(curr, json_end, "name")) != NULL) {
        const char* name_start = curr;
        while (name_start < json_end && *name_start != '"') name_start++;
        name_start++;
        const char* name_end = name_start;
        while (name_end < json_end && *name_end != '"') name_end++;

        char name[128];
        size_t nlen = name_end - name_start;
        if (nlen >= sizeof(name)) nlen = sizeof(name) - 1;
        if (nlen > 0) memcpy(name, name_start, nlen);
        name[nlen] = '\0';

        const char* obj_end = name_end;
        while (obj_end < json_end && *obj_end != '}') obj_end++;

        const char* res_ptr = scan_json_key(name_end, obj_end, "is_resident");
        int is_resident = (res_ptr && strncmp(res_ptr, "true", 4) == 0);

        if (is_resident) {
            const char* off_ptr = scan_json_key(name_end, obj_end, "offset");
            const char* len_ptr = scan_json_key(name_end, obj_end, "length");
            uint64_t t_offset = off_ptr ? strtoull(off_ptr, NULL, 10) : 0;
            uint64_t t_length = len_ptr ? strtoull(len_ptr, NULL, 10) : 0;

            float** dest_ptr = NULL;
            if (strcmp(name, "token_emb.weight") == 0) dest_ptr = &model->token_emb;
            else if (strcmp(name, "abacus_emb.embed.weight") == 0) dest_ptr = &model->abacus_emb;
            else if (strcmp(name, "stage_emb.weight") == 0) dest_ptr = &model->stage_emb;
            else if (strcmp(name, "resident_core.step_encoder.proj.weight") == 0) dest_ptr = &model->step_proj;
            else if (strcmp(name, "resident_core.noise_mlp.0.weight") == 0) dest_ptr = &model->noise_mlp_w0;
            else if (strcmp(name, "resident_core.noise_mlp.0.bias") == 0) dest_ptr = &model->noise_mlp_b0;
            else if (strcmp(name, "resident_core.noise_mlp.2.weight") == 0) dest_ptr = &model->noise_mlp_w2;
            else if (strcmp(name, "resident_core.noise_mlp.2.bias") == 0) dest_ptr = &model->noise_mlp_b2;
            else if (strcmp(name, "resident_core.norm_z.weight") == 0) dest_ptr = &model->norm_z;
            else if (strcmp(name, "resident_core.norm_y.weight") == 0) dest_ptr = &model->norm_y;
            else if (strcmp(name, "resident_core.norm_moe.weight") == 0) dest_ptr = &model->norm_moe;
            else if (strcmp(name, "resident_core.norm_refine.weight") == 0) dest_ptr = &model->norm_refine;
            else if (strcmp(name, "resident_core.attn.q_norm.weight") == 0) dest_ptr = &model->attn_q_norm;
            else if (strcmp(name, "resident_core.attn.k_norm.weight") == 0) dest_ptr = &model->attn_k_norm;
            else if (strcmp(name, "resident_core.paged_moe.router.weight") == 0) dest_ptr = &model->router_w;
            else if (strcmp(name, "resident_core.halt_gate.weight") == 0) dest_ptr = &model->halt_gate_w;
            else if (strcmp(name, "final_norm.weight") == 0) dest_ptr = &model->final_norm;
            else if (strcmp(name, "lm_head.weight") == 0) dest_ptr = &model->lm_head_w;
            // sub_ln de las 16 capas
            else if (strcmp(name, "resident_core.anchor_proj.sub_ln.weight") == 0) dest_ptr = &model->anchor_proj.sub_ln_w;
            else if (strcmp(name, "resident_core.attn.q_proj.sub_ln.weight") == 0) dest_ptr = &model->attn_q.sub_ln_w;
            else if (strcmp(name, "resident_core.attn.k_proj.sub_ln.weight") == 0) dest_ptr = &model->attn_k.sub_ln_w;
            else if (strcmp(name, "resident_core.attn.v_proj.sub_ln.weight") == 0) dest_ptr = &model->attn_v.sub_ln_w;
            else if (strcmp(name, "resident_core.attn.o_proj.sub_ln.weight") == 0) dest_ptr = &model->attn_o.sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.0.up.sub_ln.weight") == 0) dest_ptr = &model->exp_up[0].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.0.down.sub_ln.weight") == 0) dest_ptr = &model->exp_down[0].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.1.up.sub_ln.weight") == 0) dest_ptr = &model->exp_up[1].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.1.down.sub_ln.weight") == 0) dest_ptr = &model->exp_down[1].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.2.up.sub_ln.weight") == 0) dest_ptr = &model->exp_up[2].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.2.down.sub_ln.weight") == 0) dest_ptr = &model->exp_down[2].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.3.up.sub_ln.weight") == 0) dest_ptr = &model->exp_up[3].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.3.down.sub_ln.weight") == 0) dest_ptr = &model->exp_down[3].sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.shared_expert.up.sub_ln.weight") == 0) dest_ptr = &model->shared_up.sub_ln_w;
            else if (strcmp(name, "resident_core.paged_moe.shared_expert.down.sub_ln.weight") == 0) dest_ptr = &model->shared_down.sub_ln_w;
            else if (strcmp(name, "resident_core.answer_updater.sub_ln.weight") == 0) dest_ptr = &model->answer_updater.sub_ln_w;

            if (dest_ptr) {
                *dest_ptr = (float*)malloc(t_length);
                if (pread(fd_meta, *dest_ptr, t_length, (off_t)(res_off + t_offset)) == (ssize_t)t_length) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.alpha_z") == 0) {
                if (pread(fd_meta, &model->alpha_z, sizeof(float), (off_t)(res_off + t_offset)) == sizeof(float)) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.beta_y") == 0) {
                if (pread(fd_meta, &model->beta_y, sizeof(float), (off_t)(res_off + t_offset)) == sizeof(float)) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.carry_gate") == 0) {
                if (pread(fd_meta, &model->carry_gate, sizeof(float), (off_t)(res_off + t_offset)) == sizeof(float)) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.halt_gate.bias") == 0) {
                if (pread(fd_meta, &model->halt_gate_b, sizeof(float), (off_t)(res_off + t_offset)) == sizeof(float)) {
                    mapped_count++;
                }
            }
        } else {
            TracePagedLinear* target_layer = NULL;
            if (strcmp(name, "resident_core.anchor_proj") == 0) target_layer = &model->anchor_proj;
            else if (strcmp(name, "resident_core.attn.q_proj") == 0) target_layer = &model->attn_q;
            else if (strcmp(name, "resident_core.attn.k_proj") == 0) target_layer = &model->attn_k;
            else if (strcmp(name, "resident_core.attn.v_proj") == 0) target_layer = &model->attn_v;
            else if (strcmp(name, "resident_core.attn.o_proj") == 0) target_layer = &model->attn_o;
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.0.up") == 0) target_layer = &model->exp_up[0];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.0.down") == 0) target_layer = &model->exp_down[0];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.1.up") == 0) target_layer = &model->exp_up[1];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.1.down") == 0) target_layer = &model->exp_down[1];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.2.up") == 0) target_layer = &model->exp_up[2];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.2.down") == 0) target_layer = &model->exp_down[2];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.3.up") == 0) target_layer = &model->exp_up[3];
            else if (strcmp(name, "resident_core.paged_moe.expert_bank.3.down") == 0) target_layer = &model->exp_down[3];
            else if (strcmp(name, "resident_core.paged_moe.shared_expert.up") == 0) target_layer = &model->shared_up;
            else if (strcmp(name, "resident_core.paged_moe.shared_expert.down") == 0) target_layer = &model->shared_down;
            else if (strcmp(name, "resident_core.answer_updater") == 0) target_layer = &model->answer_updater;

            if (target_layer) {
                const char* sh_ptr = scan_json_key(name_end, obj_end, "shape");
                if (sh_ptr) {
                    while (sh_ptr < obj_end && *sh_ptr != '[') sh_ptr++;
                    sh_ptr++;
                    target_layer->out_features = (int)strtol(sh_ptr, (char**)&sh_ptr, 10);
                    while (sh_ptr < obj_end && (*sh_ptr == ',' || *sh_ptr == ' ')) sh_ptr++;
                    target_layer->in_features = (int)strtol(sh_ptr, NULL, 10);
                }

                const char* pids_ptr = scan_json_key(name_end, obj_end, "page_ids");
                if (pids_ptr) {
                    while (pids_ptr < obj_end && *pids_ptr != '[') pids_ptr++;
                    pids_ptr++;
                    int pids[32];
                    int n_pids = 0;
                    while (pids_ptr < obj_end && *pids_ptr != ']') {
                        while (pids_ptr < obj_end && (*pids_ptr == ' ' || *pids_ptr == ',')) pids_ptr++;
                        if (*pids_ptr == ']') break;
                        pids[n_pids++] = (int)strtol(pids_ptr, (char**)&pids_ptr, 10);
                    }
                    target_layer->num_pages = n_pids;
                    uint8_t** ptrs = (uint8_t**)malloc(n_pids * sizeof(uint8_t*));
                    float sum_sc = 0.0f;
                    for (int i = 0; i < n_pids; ++i) {
                        ptrs[i] = model->page_pool + pids[i] * page_sz;
                        const PagedHeader* ph = (const PagedHeader*)ptrs[i];
                        sum_sc += ph->scale_w;
                    }
                    target_layer->page_ptrs = (const uint8_t* const*)ptrs;
                    target_layer->scale = sum_sc / (float)n_pids;
                }
                mapped_count++;
            }
        }
        curr = obj_end;
    }

    free(json_buf);
    close(fd_meta);

    if (mapped_count != 54) {
        trace_model_free(model);
        return -8;
    }
    return 0;
}

void trace_model_free(TraceModel* model) {
    if (!model) return;
    free_workspace(&model->workspace);
    if (model->page_pool) { free(model->page_pool); model->page_pool = NULL; }
    if (model->fd >= 0) { close(model->fd); model->fd = -1; }

    free(model->token_emb); free(model->abacus_emb); free(model->stage_emb);
    free(model->step_proj);
    free(model->noise_mlp_w0); free(model->noise_mlp_b0);
    free(model->noise_mlp_w2); free(model->noise_mlp_b2);
    free(model->norm_z); free(model->norm_y);
    free(model->norm_moe); free(model->norm_refine);
    free(model->attn_q_norm); free(model->attn_k_norm);
    free(model->router_w); free(model->halt_gate_w);
    free(model->final_norm); free(model->lm_head_w);

    #define FREE_L(l) do { free((void*)l.page_ptrs); free(l.sub_ln_w); } while(0)
    FREE_L(model->anchor_proj);
    FREE_L(model->attn_q); FREE_L(model->attn_k);
    FREE_L(model->attn_v); FREE_L(model->attn_o);
    for (int i = 0; i < 4; ++i) {
        FREE_L(model->exp_up[i]);
        FREE_L(model->exp_down[i]);
    }
    FREE_L(model->shared_up); FREE_L(model->shared_down);
    FREE_L(model->answer_updater);
    #undef FREE_L
}

static void extract_abacus_ranks(const int32_t* token_ids, int L, int mask_token_id, int* ranks) {
    int cur_rank = 0;
    for (int i = L - 1; i >= 0; --i) {
        int t = token_ids[i];
        int is_digit = (t >= '0' && t <= '9') || (t == mask_token_id);
        if (is_digit) {
            ranks[i] = (cur_rank > 14) ? 14 : cur_rank;
            cur_rank++;
        } else {
            ranks[i] = 15;
            cur_rank = 0;
        }
    }
}

void trace_forward_backbone_lazy(
    TraceModel* model,
    const int32_t* token_ids,
    int seq_len,
    int stage_id,
    int prompt_len,
    const int* target_indices,
    int num_targets,
    float* out_logits
) {
    int L = seq_len;
    int D = model->cfg.d_model;
    float eps = model->cfg.eps;
    TraceWorkspace* ws = &model->workspace;

    int abacus_ranks[160];
    extract_abacus_ranks(token_ids, L, model->cfg.mask_token_id, abacus_ranks);

    float* a = ws->a;
    for (int i = 0; i < L; ++i) {
        int tok = token_ids[i];
        int r = abacus_ranks[i];
        const float* t_ptr = model->token_emb + tok * D;
        const float* ab_ptr = model->abacus_emb + r * D;
        const float* st_ptr = model->stage_emb + stage_id * D;
        for (int d = 0; d < D; ++d) a[i * D + d] = t_ptr[d] + ab_ptr[d] + st_ptr[d];
    }

    float* z_r = ws->z_r;
    float* y_r = ws->y_r;
    memcpy(z_r, a, L * D * sizeof(float));
    memset(y_r, 0, L * D * sizeof(float));

    float* anchor_feat = ws->anchor_feat;
    paged_linear_forward(&model->anchor_proj, a, anchor_feat, L, eps, ws);

    float* joint_state = ws->joint_state;
    float* norm_z_buf = ws->norm_z_buf;
    float* norm_y_buf = ws->norm_y_buf;
    float* e_r = ws->e_r;
    float* e_t = ws->e_t;

    // noise_mlp
    float noise_in[2] = {1.0f, 0.25f};
    float h_noise[192];
    for (int i = 0; i < 192; ++i) {
        float val = model->noise_mlp_w0[i * 2 + 0] * noise_in[0] +
                    model->noise_mlp_w0[i * 2 + 1] * noise_in[1] +
                    model->noise_mlp_b0[i];
        h_noise[i] = val / (1.0f + expf(-val));
    }
    for (int i = 0; i < 192; ++i) {
        float acc = model->noise_mlp_b2[i];
        for (int k = 0; k < 192; ++k) acc += model->noise_mlp_w2[i * 192 + k] * h_noise[k];
        e_t[i] = acc;
    }

    int R = model->cfg.num_recursions;
    for (int r = 0; r < R; ++r) {
        float u_norm = (float)(r + 1) / (float)R;
        float step_scale = 2.0f / (float)(R + 1);

        float angle_cat[192];
        for (int i = 0; i < 96; ++i) {
            float freq = expf((float)i * (-logf(100.0f) / 95.0f));
            float ang = u_norm * 3.141592653589793f * freq;
            angle_cat[i] = sinf(ang);
            angle_cat[i + 96] = cosf(ang);
        }
        for (int i = 0; i < D; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < 192; ++k) acc += model->step_proj[i * 192 + k] * angle_cat[k];
            e_r[i] = acc;
        }

        for (int i = 0; i < L; ++i) {
            rms_norm(z_r + i * D, model->norm_z, norm_z_buf + i * D, D, eps);
            rms_norm(y_r + i * D, model->norm_y, norm_y_buf + i * D, D, eps);
            float* js_row = joint_state + i * D;
            const float* nz_row = norm_z_buf + i * D;
            const float* ny_row = norm_y_buf + i * D;
            const float* af_row = anchor_feat + i * D;
            int d = 0;
            for (; d + 8 <= D; d += 8) {
                __m256 v_nz = _mm256_loadu_ps(nz_row + d);
                __m256 v_ny = _mm256_loadu_ps(ny_row + d);
                __m256 v_af = _mm256_loadu_ps(af_row + d);
                __m256 v_et = _mm256_loadu_ps(e_t + d);
                __m256 v_er = _mm256_loadu_ps(e_r + d);
                __m256 s = _mm256_add_ps(_mm256_add_ps(v_nz, v_ny), _mm256_add_ps(v_af, _mm256_add_ps(v_et, v_er)));
                _mm256_storeu_ps(js_row + d, s);
            }
            for (; d < D; ++d) {
                js_row[d] = nz_row[d] + ny_row[d] + af_row[d] + e_t[d] + e_r[d];
            }
        }

        // GQA Attention
        float* q = ws->q;
        float* k = ws->k;
        float* v = ws->v;
        paged_linear_forward(&model->attn_q, joint_state, q, L, eps, ws);
        paged_linear_forward(&model->attn_k, joint_state, k, L, eps, ws);
        paged_linear_forward(&model->attn_v, joint_state, v, L, eps, ws);

        // RoPE con tabla estática precomputada
#ifdef TRACE_PROFILE
        uint64_t t_rope0 = get_time_ns();
#endif
        for (int i = 0; i < L; ++i) {
            for (int h = 0; h < 4; ++h) rms_norm(q + i * D + h * 48, model->attn_q_norm, q + i * D + h * 48, 48, eps);
            for (int h = 0; h < 2; ++h) rms_norm(k + i * 96 + h * 48, model->attn_k_norm, k + i * 96 + h * 48, 48, eps);

            const float* cos_row = model->rope_cos + i * 24;
            const float* sin_row = model->rope_sin + i * 24;

            for (int j = 0; j < 24; ++j) {
                float cos_v = cos_row[j];
                float sin_v = sin_row[j];
                for (int h = 0; h < 4; ++h) {
                    float* q_ptr = q + i * D + h * 48;
                    float v0 = q_ptr[j], v1 = q_ptr[j + 24];
                    q_ptr[j] = v0 * cos_v - v1 * sin_v;
                    q_ptr[j + 24] = v1 * cos_v + v0 * sin_v;
                }
                for (int h = 0; h < 2; ++h) {
                    float* k_ptr = k + i * 96 + h * 48;
                    float v0 = k_ptr[j], v1 = k_ptr[j + 24];
                    k_ptr[j] = v0 * cos_v - v1 * sin_v;
                    k_ptr[j + 24] = v1 * cos_v + v0 * sin_v;
                }
            }
        }
#ifdef TRACE_PROFILE
        uint64_t t_rope1 = get_time_ns();
        g_prof_rope_qknorm_ns += (t_rope1 - t_rope0);
#endif

        float* attn_out_heads = ws->attn_out_heads;
        float scale_attn = 1.0f / sqrtf(48.0f);
        int t_end = (prompt_len > 0) ? (prompt_len + 35) : L;
        if (t_end > L) t_end = L;

#ifdef TRACE_PROFILE
        uint64_t t_sdpa0 = get_time_ns();
#endif
        #pragma omp parallel for num_threads(2) schedule(static)
        for (int h = 0; h < 4; ++h) {
            int kv_h = h / 2;
            for (int i = 0; i < L; ++i) {
                const float* q_vec = q + i * D + h * 48;
                float scores[160];
                float max_s = -INFINITY;

                for (int j = 0; j < L; ++j) {
                    int allowed = 0;
                    if (prompt_len <= 0) {
                        allowed = (j / 16 <= i / 16);
                    } else {
                        if (i < prompt_len) allowed = (j / 16 <= i / 16) && (j < prompt_len);
                        else if (i < t_end) allowed = (j < t_end);
                        else allowed = (j / 16 <= i / 16);
                    }
                    if (!allowed) {
                        scores[j] = -INFINITY;
                        continue;
                    }
                    const float* k_vec = k + j * 96 + kv_h * 48;
                    float dot = avx2_dot_product_48(q_vec, k_vec);
                    scores[j] = dot * scale_attn;
                    if (scores[j] > max_s) max_s = scores[j];
                }
                float weights[160];
                double sum_exp = 0.0;
                for (int j = 0; j < L; ++j) {
                    if (scores[j] == -INFINITY) {
                        weights[j] = 0.0f;
                    } else {
                        float e = expf(scores[j] - max_s);
                        weights[j] = e;
                        sum_exp += (double)e;
                    }
                }
                float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0f;
                for (int j = 0; j < L; ++j) weights[j] *= inv_sum;

                float out_h[48];
                memset(out_h, 0, sizeof(out_h));
                for (int j = 0; j < L; ++j) {
                    float w = weights[j];
                    if (w > 0.0f) {
                        const float* v_row = v + j * 96 + kv_h * 48;
                        __m256 vw = _mm256_set1_ps(w);
                        for (int d = 0; d < 48; d += 8) {
                            __m256 vv = _mm256_loadu_ps(v_row + d);
                            __m256 vo = _mm256_loadu_ps(out_h + d);
                            vo = _mm256_add_ps(vo, _mm256_mul_ps(vw, vv));
                            _mm256_storeu_ps(out_h + d, vo);
                        }
                    }
                }
                memcpy(attn_out_heads + i * D + h * 48, out_h, 48 * sizeof(float));
            }
        }
#ifdef TRACE_PROFILE
        uint64_t t_sdpa1 = get_time_ns();
        g_prof_sdpa_ns += (t_sdpa1 - t_sdpa0);
#endif

        float* attn_final = ws->attn_final;
        paged_linear_forward(&model->attn_o, attn_out_heads, attn_final, L, eps, ws);

        float eff_alpha = model->alpha_z * step_scale;
        float eff_beta = model->beta_y * step_scale;

        float* z_mid = ws->z_mid;
        __m256 valpha = _mm256_set1_ps(eff_alpha);
        int idx_m = 0;
        int total_ld = L * D;
        for (; idx_m + 8 <= total_ld; idx_m += 8) {
            __m256 vz = _mm256_loadu_ps(z_r + idx_m);
            __m256 va = _mm256_loadu_ps(attn_final + idx_m);
            _mm256_storeu_ps(z_mid + idx_m, _mm256_add_ps(vz, _mm256_mul_ps(valpha, va)));
        }
        for (; idx_m < total_ld; ++idx_m) z_mid[idx_m] = z_r[idx_m] + eff_alpha * attn_final[idx_m];

        // MoE
        float* z_norm_moe = ws->z_norm_moe;
        for (int i = 0; i < L; ++i) rms_norm(z_mid + i * D, model->norm_moe, z_norm_moe + i * D, D, eps);

        float* moe_out = ws->moe_out;
        memset(moe_out, 0, L * D * sizeof(float));

        // Experto compartido sobre tokens no pad
        float* shared_up_buf = ws->shared_up_buf;
        paged_linear_forward(&model->shared_up, z_norm_moe, shared_up_buf, L, eps, ws);
        __m256 zero_vec = _mm256_setzero_ps();
        int total_sh = L * 384;
        int idx_sh = 0;
        for (; idx_sh + 8 <= total_sh; idx_sh += 8) {
            __m256 v = _mm256_loadu_ps(shared_up_buf + idx_sh);
            __m256 pos = _mm256_max_ps(v, zero_vec);
            _mm256_storeu_ps(shared_up_buf + idx_sh, _mm256_mul_ps(pos, pos));
        }
        for (; idx_sh < total_sh; ++idx_sh) {
            float val = shared_up_buf[idx_sh];
            shared_up_buf[idx_sh] = (val > 0.0f) ? val * val : 0.0f;
        }
        float* shared_down_buf = ws->shared_down_buf;
        paged_linear_forward(&model->shared_down, shared_up_buf, shared_down_buf, L, eps, ws);

        for (int i = 0; i < L; ++i) {
            if (token_ids[i] != model->cfg.pad_token_id) {
                float* moe_row = moe_out + i * D;
                const float* sh_row = shared_down_buf + i * D;
                int d = 0;
                for (; d + 8 <= D; d += 8) {
                    __m256 vm = _mm256_loadu_ps(moe_row + d);
                    __m256 vs = _mm256_loadu_ps(sh_row + d);
                    _mm256_storeu_ps(moe_row + d, _mm256_add_ps(vm, vs));
                }
                for (; d < D; ++d) moe_row[d] += sh_row[d];
            }
        }

        // Enrutamiento Top-K por bloques de 16 tokens (Agrupado por experto)
#ifdef TRACE_PROFILE
        uint64_t t_moe0 = get_time_ns();
#endif
        int num_blocks = L / 16;
        float scores_blk[10][16][4];
        for (int b = 0; b < num_blocks; ++b) {
            for (int t = 0; t < 16; ++t) {
                int tok_idx = b * 16 + t;
                if (token_ids[tok_idx] == model->cfg.pad_token_id) {
                    for (int e = 0; e < 4; ++e) scores_blk[b][t][e] = 0.0f;
                    continue;
                }
                float max_l = -INFINITY;
                float logits_e[4];
                for (int e = 0; e < 4; ++e) {
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) dot += (double)z_norm_moe[tok_idx * D + d] * (double)model->router_w[e * D + d];
                    logits_e[e] = (float)dot;
                    if (logits_e[e] > max_l) max_l = logits_e[e];
                }
                double sum_exp = 0.0;
                for (int e = 0; e < 4; ++e) {
                    scores_blk[b][t][e] = expf(logits_e[e] - max_l);
                    sum_exp += (double)scores_blk[b][t][e];
                }
                float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0f;
                for (int e = 0; e < 4; ++e) scores_blk[b][t][e] *= inv_sum;
            }
        }
#ifdef TRACE_PROFILE
        uint64_t t_moe1 = get_time_ns();
        g_prof_moe_router_ns += (t_moe1 - t_moe0);
#endif

        for (int e = 0; e < 4; ++e) {
            int count_e = 0;
            for (int b = 0; b < num_blocks; ++b) {
                int idxs[16];
                for (int t = 0; t < 16; ++t) idxs[t] = t;
                for (int i = 0; i < 15; ++i) {
                    for (int j = i + 1; j < 16; ++j) {
                        if (scores_blk[b][idxs[j]][e] > scores_blk[b][idxs[i]][e]) {
                            int tmp = idxs[i]; idxs[i] = idxs[j]; idxs[j] = tmp;
                        }
                    }
                }

                for (int sel = 0; sel < 6; ++sel) {
                    int t_idx = idxs[sel];
                    float gate = scores_blk[b][t_idx][e];
                    if (gate > 1e-4f) {
                        int g_tok = b * 16 + t_idx;
                        ws->exp_tok_idx[count_e] = g_tok;
                        ws->exp_tok_gate[count_e] = gate;
                        memcpy(ws->exp_tokens_in + count_e * D, z_norm_moe + g_tok * D, D * sizeof(float));
                        count_e++;
                    }
                }
            }

            if (count_e > 0) {
                paged_linear_forward(&model->exp_up[e], ws->exp_tokens_in, ws->exp_up_buf, count_e, eps, ws);
                int total_up = count_e * 384;
                int idx_up = 0;
                for (; idx_up + 8 <= total_up; idx_up += 8) {
                    __m256 v = _mm256_loadu_ps(ws->exp_up_buf + idx_up);
                    __m256 pos = _mm256_max_ps(v, zero_vec);
                    _mm256_storeu_ps(ws->exp_up_buf + idx_up, _mm256_mul_ps(pos, pos));
                }
                for (; idx_up < total_up; ++idx_up) {
                    float val = ws->exp_up_buf[idx_up];
                    ws->exp_up_buf[idx_up] = (val > 0.0f) ? val * val : 0.0f;
                }
                paged_linear_forward(&model->exp_down[e], ws->exp_up_buf, ws->exp_down_buf, count_e, eps, ws);

                for (int k = 0; k < count_e; ++k) {
                    int g_tok = ws->exp_tok_idx[k];
                    float gate = ws->exp_tok_gate[k];
                    __m256 vgate = _mm256_set1_ps(gate);
                    const float* down_row = ws->exp_down_buf + k * D;
                    float* moe_row = moe_out + g_tok * D;
                    int d = 0;
                    for (; d + 8 <= D; d += 8) {
                        __m256 vm = _mm256_loadu_ps(moe_row + d);
                        __m256 vd = _mm256_loadu_ps(down_row + d);
                        _mm256_storeu_ps(moe_row + d, _mm256_add_ps(vm, _mm256_mul_ps(vd, vgate)));
                    }
                    for (; d < D; ++d) {
                        moe_row[d] += down_row[d] * gate;
                    }
                }
            }
        }

        float* z_next = ws->z_next;
        __m256 valpha_moe = _mm256_set1_ps(eff_alpha);
        int idx_z = 0;
        for (; idx_z + 8 <= total_ld; idx_z += 8) {
            __m256 vz = _mm256_loadu_ps(z_mid + idx_z);
            __m256 vm = _mm256_loadu_ps(moe_out + idx_z);
            _mm256_storeu_ps(z_next + idx_z, _mm256_add_ps(vz, _mm256_mul_ps(valpha_moe, vm)));
        }
        for (; idx_z < total_ld; ++idx_z) z_next[idx_z] = z_mid[idx_z] + eff_alpha * moe_out[idx_z];

        // Answer updater
#ifdef TRACE_PROFILE
        uint64_t t_silu0 = get_time_ns();
#endif
        float* fusion = ws->fusion;
        for (int i = 0; i < L; ++i) {
            float* f_row = fusion + i * D;
            const float* y_row = y_r + i * D;
            const float* zn_row = z_next + i * D;
            int d = 0;
            for (; d + 8 <= D; d += 8) {
                __m256 vy = _mm256_loadu_ps(y_row + d);
                __m256 vz = _mm256_loadu_ps(zn_row + d);
                _mm256_storeu_ps(f_row + d, _mm256_add_ps(vy, vz));
            }
            for (; d < D; ++d) f_row[d] = y_row[d] + zn_row[d];
            rms_norm(f_row, model->norm_refine, f_row, D, eps);
        }
#ifdef TRACE_PROFILE
        uint64_t t_silu1 = get_time_ns();
#endif

        float* delta_y = ws->delta_y;
        paged_linear_forward(&model->answer_updater, fusion, delta_y, L, eps, ws);

#ifdef TRACE_PROFILE
        uint64_t t_silu2 = get_time_ns();
#endif
        for (int i = 0; i < L * D; ++i) {
            float v = delta_y[i];
            float silu_v = v / (1.0f + expf(-v));
            y_r[i] = y_r[i] + eff_beta * silu_v;
            z_r[i] = z_next[i];
        }
#ifdef TRACE_PROFILE
        uint64_t t_silu3 = get_time_ns();
        g_prof_silu_updater_ns += ((t_silu1 - t_silu0) + (t_silu3 - t_silu2));
#endif
    }

    float* normed_y = ws->normed_y;
    for (int i = 0; i < L; ++i) rms_norm(y_r + i * D, model->final_norm, normed_y + i * D, D, eps);

#ifdef TRACE_PROFILE
    uint64_t t_lm0 = get_time_ns();
#endif
    int V = model->cfg.vocab_size;
    if (target_indices == NULL || num_targets <= 0) {
        for (int i = 0; i < L; ++i) {
            const float* act_row = normed_y + i * D;
            for (int v = 0; v < V; ++v) {
                out_logits[i * V + v] = avx2_dot_product_192(act_row, model->lm_head_w + v * D);
            }
        }
    } else {
        for (int t = 0; t < num_targets; ++t) {
            int i = target_indices[t];
            const float* act_row = normed_y + i * D;
            for (int v = 0; v < V; ++v) {
                out_logits[t * V + v] = avx2_dot_product_192(act_row, model->lm_head_w + v * D);
            }
        }
    }
#ifdef TRACE_PROFILE
    uint64_t t_lm1 = get_time_ns();
    g_prof_lm_head_ns += (t_lm1 - t_lm0);
#endif
}

void trace_forward_backbone(
    TraceModel* model,
    const int32_t* token_ids,
    int seq_len,
    int stage_id,
    int prompt_len,
    float* out_logits
) {
    trace_forward_backbone_lazy(model, token_ids, seq_len, stage_id, prompt_len, NULL, 0, out_logits);
}

// Inferencia Causal determinista de 2 etapas
int trace_generate_causal_two_step(
    TraceModel* model,
    const uint8_t* prompt_bytes,
    int prompt_len,
    char* out_str,
    int max_out_len
) {
#ifdef TRACE_PROFILE
    uint64_t t_gen0 = get_time_ns();
#endif
    int max_seq = model->cfg.max_seq_len;
    int32_t seq[160];
    for (int i = 0; i < max_seq; ++i) seq[i] = model->cfg.pad_token_id;

    seq[0] = model->cfg.bos_token_id;
    for (int i = 0; i < prompt_len; ++i) seq[1 + i] = prompt_bytes[i];

    const char* template_str = "pw: ???\npms: ???\nwatt: ???\nms: ???\n";
    int t_len = (int)strlen(template_str);
    int p_end = 1 + prompt_len;
    for (int i = 0; i < t_len; ++i) seq[p_end + i] = (uint8_t)template_str[i];
    seq[p_end + t_len] = model->cfg.eos_token_id;

    int total_tokens = p_end + t_len + 1;
    int L_trim = ((total_tokens + 15) / 16) * 16;
    if (L_trim > max_seq) L_trim = max_seq;

    int pw_start = p_end + 4, pms_start = p_end + 13;
    int watt_start = p_end + 23, ms_start = p_end + 31;

    for (int i = 0; i < 3; ++i) {
        seq[pw_start + i] = model->cfg.mask_token_id;
        seq[pms_start + i] = model->cfg.mask_token_id;
        seq[watt_start + i] = model->cfg.mask_token_id;
        seq[ms_start + i] = model->cfg.mask_token_id;
    }

    float target_logits[6 * 260];

    // Pasada 1 (Etapa 0)
    int targets_0[6] = {
        pw_start + 0, pw_start + 1, pw_start + 2,
        pms_start + 0, pms_start + 1, pms_start + 2
    };
    trace_forward_backbone_lazy(model, seq, L_trim, 0, p_end, targets_0, 6, target_logits);
    for (int t = 0; t < 6; ++t) {
        int best_v = 0; float max_l = -INFINITY;
        for (int v = 0; v < model->cfg.vocab_size; ++v) {
            float val = target_logits[t * model->cfg.vocab_size + v];
            if (val > max_l) { max_l = val; best_v = v; }
        }
        seq[targets_0[t]] = best_v;
    }

    // Pasada 2 (Etapa 1: Unidades)
    int targets_1[2] = { watt_start + 2, ms_start + 2 };
    trace_forward_backbone_lazy(model, seq, L_trim, 1, p_end, targets_1, 2, target_logits);
    for (int t = 0; t < 2; ++t) {
        int best_v = 0; float max_l = -INFINITY;
        for (int v = 0; v < model->cfg.vocab_size; ++v) {
            float val = target_logits[t * model->cfg.vocab_size + v];
            if (val > max_l) { max_l = val; best_v = v; }
        }
        seq[targets_1[t]] = best_v;
    }

    // Pasada 3 (Etapa 2: Decenas)
    int targets_2[2] = { watt_start + 1, ms_start + 1 };
    trace_forward_backbone_lazy(model, seq, L_trim, 2, p_end, targets_2, 2, target_logits);
    for (int t = 0; t < 2; ++t) {
        int best_v = 0; float max_l = -INFINITY;
        for (int v = 0; v < model->cfg.vocab_size; ++v) {
            float val = target_logits[t * model->cfg.vocab_size + v];
            if (val > max_l) { max_l = val; best_v = v; }
        }
        seq[targets_2[t]] = best_v;
    }

    // Pasada 4 (Etapa 3: Centenas)
    int targets_3[2] = { watt_start + 0, ms_start + 0 };
    trace_forward_backbone_lazy(model, seq, L_trim, 3, p_end, targets_3, 2, target_logits);
    for (int t = 0; t < 2; ++t) {
        int best_v = 0; float max_l = -INFINITY;
        for (int v = 0; v < model->cfg.vocab_size; ++v) {
            float val = target_logits[t * model->cfg.vocab_size + v];
            if (val > max_l) { max_l = val; best_v = v; }
        }
        seq[targets_3[t]] = best_v;
    }

    int written = 0;
    for (int i = 0; i < t_len && written < max_out_len - 1; ++i) {
        out_str[written++] = (char)seq[p_end + i];
    }
    out_str[written] = '\0';
#ifdef TRACE_PROFILE
    uint64_t t_gen1 = get_time_ns();
    g_prof_total_ns += (t_gen1 - t_gen0);
    g_prof_call_count++;
#endif
    return written;
}

#ifdef TRACE_INFER_MAIN
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: %s <modelo.bin> [prompt]\n", argv[0]);
        return 1;
    }
    TraceModel model;
    if (trace_model_load_odirect(argv[1], &model) != 0) {
        printf("Error cargando %s\n", argv[1]);
        return 1;
    }
    printf("trace_infer: cargado con O_DIRECT=%d (preads=%lu, pages_offset=%lu)\n",
           model.used_odirect, (unsigned long)model.pread_calls, (unsigned long)model.pages_offset);
    const char* p = (argc >= 3) ? argv[2] : "M: 4 | S: 2 | P: 3";
    char out[256];
    trace_generate_causal_two_step(&model, (const uint8_t*)p, (int)strlen(p), out, 256);
    printf("Salida:\n%s\n", out);
    trace_model_free(&model);
    return 0;
}
#endif

/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "trace_lm.h"
#include "trace_lm_kv.h"
#include "paged_dispatch.h"
#include "../trace_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

TraceLMWorkspace* trace_lm_workspace_create(const TraceLMConfig* cfg, int max_batch_seq) {
    if (!cfg || max_batch_seq <= 0) return NULL;

    TraceLMWorkspace* ws = (TraceLMWorkspace*)calloc(1, sizeof(TraceLMWorkspace));
    if (!ws) return NULL;

    int d_model = cfg->d_model;
    int d_ff = cfg->d_ff;
    int n_heads = cfg->n_heads;
    int v_dim = n_heads * cfg->v_head_dim;
    int q_nope_dim = n_heads * cfg->qk_nope_head_dim;
    int q_rope_dim = n_heads * cfg->qk_rope_head_dim;
    int kv_lora_rank = cfg->kv_lora_rank;

    // Dimensionamiento seguro contra desbordamiento en ffn.down y proyecciones
    int max_in_dim = d_model;
    if (d_ff > max_in_dim) max_in_dim = d_ff;
    if (v_dim > max_in_dim) max_in_dim = v_dim;

    int max_out_dim = d_model;
    if (d_ff > max_out_dim) max_out_dim = d_ff;
    if (q_nope_dim > max_out_dim) max_out_dim = q_nope_dim;
    if (q_rope_dim > max_out_dim) max_out_dim = q_rope_dim;
    if (kv_lora_rank > max_out_dim) max_out_dim = kv_lora_rank;

    ws->max_batch_seq = max_batch_seq;
    ws->max_in_dim = max_in_dim;
    ws->max_out_dim = max_out_dim;

    #define ALLOC_MEM(ptr, sz) do { \
        if (posix_memalign((void**)&(ptr), 64, (sz)) != 0) { \
            trace_lm_workspace_free(ws); \
            return NULL; \
        } \
    } while(0)

    ALLOC_MEM(ws->x, (size_t)max_batch_seq * (size_t)d_model * sizeof(float));
    ALLOC_MEM(ws->h_norm, (size_t)max_batch_seq * (size_t)d_model * sizeof(float));
    ALLOC_MEM(ws->x_norm, (size_t)max_batch_seq * (size_t)max_in_dim * sizeof(float));
    ALLOC_MEM(ws->q_x, (size_t)max_batch_seq * (size_t)max_in_dim * sizeof(int8_t));
    ALLOC_MEM(ws->scale_x, (size_t)max_batch_seq * sizeof(float));
    ALLOC_MEM(ws->acc_int32, (size_t)max_batch_seq * (size_t)max_out_dim * sizeof(int32_t));

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
#endif
    ws->max_threads = max_threads;

    ALLOC_MEM(ws->q_c, (size_t)max_batch_seq * (size_t)q_nope_dim * sizeof(float));
    ALLOC_MEM(ws->q_pe, (size_t)max_batch_seq * (size_t)q_rope_dim * sizeof(float));
    ALLOC_MEM(ws->q_c_absorbed, (size_t)max_batch_seq * (size_t)n_heads * (size_t)kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->scores, (size_t)max_threads * 4 * (size_t)cfg->max_seq_len * sizeof(float));
    ws->attn_probs = NULL;
    ALLOC_MEM(ws->o_latent, (size_t)n_heads * (size_t)max_batch_seq * (size_t)kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->o_heads, (size_t)max_batch_seq * (size_t)v_dim * sizeof(float));
    ALLOC_MEM(ws->ffn_hidden, (size_t)max_batch_seq * (size_t)d_ff * sizeof(float));

    ALLOC_MEM(ws->c_kv_tmp, (size_t)max_batch_seq * (size_t)kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->c_kv_tmp_single, (size_t)kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->k_pe_tmp, (size_t)max_batch_seq * (size_t)cfg->qk_rope_head_dim * sizeof(float));
    ALLOC_MEM(ws->attn_out, (size_t)max_batch_seq * (size_t)d_model * sizeof(float));
    ALLOC_MEM(ws->ffn_out, (size_t)max_batch_seq * (size_t)d_model * sizeof(float));
    ALLOC_MEM(ws->c_kv_dequant, (size_t)cfg->max_seq_len * (size_t)kv_lora_rank * sizeof(float));
    #undef ALLOC_MEM

    return ws;
}

void trace_lm_workspace_free(TraceLMWorkspace* ws) {
    if (!ws) return;
    if (ws->x) free(ws->x);
    if (ws->h_norm) free(ws->h_norm);
    if (ws->x_norm) free(ws->x_norm);
    if (ws->q_x) free(ws->q_x);
    if (ws->scale_x) free(ws->scale_x);
    if (ws->acc_int32) free(ws->acc_int32);
    if (ws->q_c) free(ws->q_c);
    if (ws->q_pe) free(ws->q_pe);
    if (ws->q_c_absorbed) free(ws->q_c_absorbed);
    if (ws->scores) free(ws->scores);
    if (ws->attn_probs) free(ws->attn_probs);
    if (ws->o_latent) free(ws->o_latent);
    if (ws->o_heads) free(ws->o_heads);
    if (ws->ffn_hidden) free(ws->ffn_hidden);
    if (ws->c_kv_tmp) free(ws->c_kv_tmp);
    if (ws->c_kv_tmp_single) free(ws->c_kv_tmp_single);
    if (ws->k_pe_tmp) free(ws->k_pe_tmp);
    if (ws->attn_out) free(ws->attn_out);
    if (ws->ffn_out) free(ws->ffn_out);
    if (ws->c_kv_dequant) free(ws->c_kv_dequant);
    free(ws);
}

// Proyeccion BitLinear estandar con sub_ln + cuantizacion INT8 + kernel AVX2 Add-Sub
static void paged_linear_forward_act_int8(
    const float* in,
    const TraceLMPagedLinear* lin,
    float* out,
    int M,
    TraceLMWorkspace* ws,
    float eps
) {
    int in_f = lin->in_features;
    int out_f = lin->out_features;

    // 1-2. RMSNorm sub_ln + Cuantizacion simetrica a INT8 fusionados por fila (caliente en L1)
    #pragma omp parallel for schedule(static) if (M >= 16)
    for (int m = 0; m < M; ++m) {
        float* row = ws->x_norm + (size_t)m * (size_t)in_f;
        rms_norm(in + (size_t)m * (size_t)in_f, lin->sub_ln_w, row, in_f, eps);

        float amax = eps;
        for (int i = 0; i < in_f; ++i) {
            float v = fabsf(row[i]);
            if (v > amax) amax = v;
        }
        float scale = amax / 127.0f;
        float scale_inv = 127.0f / amax;
        ws->scale_x[m] = scale;

        int8_t* qrow = ws->q_x + (size_t)m * (size_t)in_f;
        for (int i = 0; i < in_f; ++i) {
            float q = nearbyintf(row[i] * scale_inv);
            if (!isfinite(q)) q = 0.0f;
            else if (q > 127.0f) q = 127.0f;
            else if (q < -127.0f) q = -127.0f;
            qrow[i] = (int8_t)q;
        }
    }

    // 3. Matmul AVX2 aditivo
    paged_direct_matmul_fast_c_multi(ws->q_x, lin->page_ptrs, lin->num_pages, ws->acc_int32, M, in_f, out_f);

    // 4. Descuantizacion
    #pragma omp parallel for schedule(static) if (M >= 16)
    for (int m = 0; m < M; ++m) {
        float factor = ws->scale_x[m] * lin->scale;
        const int32_t* acc_row = ws->acc_int32 + (size_t)m * (size_t)out_f;
        float* out_row = out + (size_t)m * (size_t)out_f;
        for (int i = 0; i < out_f; ++i) {
            out_row[i] = (float)acc_row[i] * factor;
        }
    }
}

// Aplicacion de RoPE intercalado: output[0::2] = x1*c - x2*s, output[1::2] = x2*c + x1*s
static void apply_rope_c(
    float* v,
    int d_pe,
    int pos,
    const float* rope_cos,
    const float* rope_sin
) {
    const float* c = rope_cos + pos * d_pe;
    const float* s = rope_sin + pos * d_pe;

    for (int j = 0; j < d_pe; j += 2) {
        float x1 = v[j];
        float x2 = v[j + 1];
        float c_val = c[j];
        float s_val = s[j];
        v[j] = x1 * c_val - x2 * s_val;
        v[j + 1] = x2 * c_val + x1 * s_val;
    }
}

int trace_lm_forward(
    TraceLMModel* model,
    TraceLMKVCache* kv,
    const int* input_tokens,
    int num_tokens,
    float* out_logits,
    int last_token_only,
    TraceLMWorkspace* ws
) {
    // 1. Validaciones obligatorias de seguridad
    if (!model || !kv || !ws || !out_logits || !input_tokens) return -1;
    if (num_tokens <= 0) return 0;
    if (num_tokens > kv->max_seq_len || num_tokens > model->cfg.max_seq_len) return -2;

    // 2. Prefix Caching real sin casos borde
    int common_pfx = trace_lm_kv_find_common_prefix(kv, input_tokens, num_tokens);
    if (common_pfx == num_tokens && num_tokens > 0) {
        common_pfx = num_tokens - 1; // Garantiza S >= 1 para devolver logits validos del ultimo token
        kv->last_common_prefix = common_pfx;
    }

    int past_len = common_pfx;
    int S = num_tokens - past_len;
    if (S > ws->max_batch_seq) return -3;

    int d_model = model->cfg.d_model;
    int d_ff = model->cfg.d_ff;
    int n_heads = model->cfg.n_heads;
    int qk_nope_head_dim = model->cfg.qk_nope_head_dim;
    int qk_rope_head_dim = model->cfg.qk_rope_head_dim;
    int v_head_dim = model->cfg.v_head_dim;
    int kv_lora_rank = model->cfg.kv_lora_rank;
    float eps = model->cfg.eps;
    float attn_scale = 1.0f / sqrtf((float)(qk_nope_head_dim + qk_rope_head_dim));

    // 3. Lookup de embeddings lexicos para los S tokens nuevos
    for (int t = 0; t < S; ++t) {
        int tok = input_tokens[past_len + t];
        if (tok < 0 || tok >= model->cfg.vocab_size) tok = model->cfg.pad_token_id;
        const float* emb_src = model->tok_emb + (size_t)tok * (size_t)d_model;
        memcpy(ws->x + (size_t)t * (size_t)d_model, emb_src, sizeof(float) * (size_t)d_model);
    }

    // 4. Bucle sobre capas Transformer
    for (int l = 0; l < model->cfg.n_layers; ++l) {
        TraceLMLayer* layer = &model->layers[l];

        // 4.1. RMSNorm previo a atencion (escribe en h_norm, preservando x_norm para sub_ln)
        #pragma omp parallel for schedule(static) if (S >= 16)
        for (int t = 0; t < S; ++t) {
            rms_norm(ws->x + (size_t)t * (size_t)d_model, layer->attn_norm_w, ws->h_norm + (size_t)t * (size_t)d_model, d_model, eps);
        }

        // 4.2. Proyeccion KV latente: w_dkv
        paged_linear_forward_act_int8(ws->h_norm, &layer->w_dkv, ws->c_kv_tmp, S, ws, eps);

        // 4.3. Normalizacion de cuello de botella: kv_norm
        #pragma omp parallel for schedule(static) if (S >= 16)
        for (int t = 0; t < S; ++t) {
            rms_norm(ws->c_kv_tmp + (size_t)t * (size_t)kv_lora_rank, layer->kv_norm_w, ws->c_kv_tmp + (size_t)t * (size_t)kv_lora_rank, kv_lora_rank, eps);
        }

        // 4.4. Proyeccion de clave rotacional: w_kr
        paged_linear_forward_act_int8(ws->h_norm, &layer->w_kr, ws->k_pe_tmp, S, ws, eps);
        #pragma omp parallel for schedule(static) if (S >= 16)
        for (int t = 0; t < S; ++t) {
            apply_rope_c(ws->k_pe_tmp + (size_t)t * (size_t)qk_rope_head_dim, qk_rope_head_dim, past_len + t, model->rope_cos, model->rope_sin);
        }

        // 4.5. Guardar c_kv y k_pe en el KV Cache para cada token nuevo
        #pragma omp parallel for schedule(static) if (S >= 16)
        for (int t = 0; t < S; ++t) {
            trace_lm_kv_store(kv, l, past_len + t, ws->c_kv_tmp + (size_t)t * (size_t)kv_lora_rank, ws->k_pe_tmp + (size_t)t * (size_t)qk_rope_head_dim);
        }

        // 4.6. Proyecciones de consulta: w_q y w_qr
        paged_linear_forward_act_int8(ws->h_norm, &layer->w_q, ws->q_c, S, ws, eps);
        paged_linear_forward_act_int8(ws->h_norm, &layer->w_qr, ws->q_pe, S, ws, eps);

        // Aplicar RoPE sobre q_pe
        #pragma omp parallel for schedule(static) if (S >= 16)
        for (int t = 0; t < S; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                float* q_pe_h = ws->q_pe + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)qk_rope_head_dim;
                apply_rope_c(q_pe_h, qk_rope_head_dim, past_len + t, model->rope_cos, model->rope_sin);
            }
        }

        // 4.7. Absorcion 1: W_uk^T q_c via kernel despachado (0 matrices FP32 en RAM)
        paged_absorb_uk_f32(ws->q_c, layer->w_uk.page_ptrs, layer->w_uk.num_pages, layer->w_uk.scale, ws->q_c_absorbed, n_heads, qk_nope_head_dim, kv_lora_rank, S);

        // 4.8. Atencion Causal Multi-Head Latente (descuantizacion unica por capa + 4 cabezas compartiendo KV)
        int end_len = past_len + S;
        const float* c_kv_layer = NULL;
        if (kv->use_int8_kv) {
            size_t layer_scale_base = (size_t)l * (size_t)kv->max_seq_len;
            #pragma omp parallel for schedule(static) if (end_len >= 64)
            for (int s = 0; s < end_len; ++s) {
                size_t scale_idx = layer_scale_base + (size_t)s;
                float sc = kv->c_kv_scales[scale_idx];
                const int8_t* src = kv->c_kv_i8 + scale_idx * (size_t)kv_lora_rank;
                float* dst = ws->c_kv_dequant + (size_t)s * (size_t)kv_lora_rank;
                for (int c = 0; c < kv_lora_rank; ++c) {
                    dst[c] = (float)src[c] * sc;
                }
            }
            c_kv_layer = ws->c_kv_dequant;
        } else {
            c_kv_layer = kv->c_kv_fp32 + (size_t)l * (size_t)kv->max_seq_len * (size_t)kv_lora_rank;
        }
        const float* k_pe_layer = kv->k_pe + (size_t)l * (size_t)kv->max_seq_len * (size_t)qk_rope_head_dim;

        int n_threads = 1;
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
        if (n_threads > ws->max_threads) n_threads = ws->max_threads;
        if (n_threads < 1) n_threads = 1;
#endif

        if (S >= 2) {
            #pragma omp parallel for schedule(guided) num_threads(n_threads)
            for (int t = 0; t < S; ++t) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                float* sc_buf = ws->scores + (size_t)tid * 4 * (size_t)model->cfg.max_seq_len;
                int total_len = past_len + t + 1;
                const float* q_c_t = ws->q_c_absorbed + (size_t)t * (size_t)n_heads * (size_t)kv_lora_rank;
                const float* q_pe_t = ws->q_pe + (size_t)t * (size_t)n_heads * (size_t)qk_rope_head_dim;
                float* o_lat_t = ws->o_latent + (size_t)t * (size_t)n_heads * (size_t)kv_lora_rank;

                for (int h = 0; h < n_heads; h += 4) {
                    int h_cnt = (h + 4 <= n_heads) ? 4 : (n_heads - h);
                    trace_mla_attn_group_f32(
                        q_c_t, q_pe_t, c_kv_layer, k_pe_layer, o_lat_t,
                        sc_buf, h, h_cnt, total_len, kv_lora_rank, qk_rope_head_dim, attn_scale
                    );
                }
            }
        } else {
            int total_len = past_len + 1;
            int num_groups = (n_heads + 3) / 4;
            #pragma omp parallel for schedule(static) num_threads(n_threads) if (num_groups >= 2 && total_len >= 64)
            for (int g = 0; g < num_groups; ++g) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                float* sc_buf = ws->scores + (size_t)tid * 4 * (size_t)model->cfg.max_seq_len;
                int h = g * 4;
                int h_cnt = (h + 4 <= n_heads) ? 4 : (n_heads - h);
                trace_mla_attn_group_f32(
                    ws->q_c_absorbed, ws->q_pe, c_kv_layer, k_pe_layer, ws->o_latent,
                    sc_buf, h, h_cnt, total_len, kv_lora_rank, qk_rope_head_dim, attn_scale
                );
            }
        }

        // 4.9. Absorcion 2: W_uv . o_latent via kernel despachado
        paged_absorb_uv_f32(ws->o_latent, layer->w_uv.page_ptrs, layer->w_uv.num_pages, layer->w_uv.scale, ws->o_heads, n_heads, v_head_dim, kv_lora_rank, S);

        // 4.10. Proyeccion de salida: w_out
        paged_linear_forward_act_int8(ws->o_heads, &layer->w_out, ws->attn_out, S, ws, eps);

        // Residual de atencion: x = x + attn_out
        int total_d = S * d_model;
        #pragma omp parallel for schedule(static) if (total_d >= 4096)
        for (int i = 0; i < total_d; ++i) {
            ws->x[i] += ws->attn_out[i];
        }

        // 4.11. FFN: RMSNorm -> ffn_up -> Squared-ReLU -> ffn_down
        #pragma omp parallel for schedule(static) if (S >= 16)
        for (int t = 0; t < S; ++t) {
            rms_norm(ws->x + (size_t)t * (size_t)d_model, layer->ffn_norm_w, ws->h_norm + (size_t)t * (size_t)d_model, d_model, eps);
        }
        paged_linear_forward_act_int8(ws->h_norm, &layer->ffn_up, ws->ffn_hidden, S, ws, eps);

        // Activacion Squared-ReLU: relu(x)^2
        int total_ff = S * d_ff;
        #pragma omp parallel for schedule(static) if (total_ff >= 4096)
        for (int i = 0; i < total_ff; ++i) {
            float val = ws->ffn_hidden[i];
            float r = (val > 0.0f) ? val : 0.0f;
            ws->ffn_hidden[i] = r * r;
        }

        paged_linear_forward_act_int8(ws->ffn_hidden, &layer->ffn_down, ws->ffn_out, S, ws, eps);

        // Residual FFN: x = x + ffn_out
        #pragma omp parallel for schedule(static) if (total_d >= 4096)
        for (int i = 0; i < total_d; ++i) {
            ws->x[i] += ws->ffn_out[i];
        }
    }

    // 5. Final RMSNorm
    #pragma omp parallel for schedule(static) if (S >= 16)
    for (int t = 0; t < S; ++t) {
        rms_norm(ws->x + (size_t)t * (size_t)d_model, model->final_norm_w, ws->x_norm + (size_t)t * (size_t)d_model, d_model, eps);
    }

    // 6. Proyeccion de salida con tied embeddings
    int vocab_size = model->cfg.vocab_size;
    if (last_token_only) {
        // Optimizado para CLI / generacion: computa unicamente el ultimo token (S - 1)
        const float* x_last = ws->x_norm + (size_t)(S - 1) * (size_t)d_model;
        #pragma omp parallel for schedule(static) if (vocab_size >= 256)
        for (int v = 0; v < vocab_size; ++v) {
            const float* emb_v = model->lm_head_w + (size_t)v * (size_t)d_model;
            out_logits[v] = trace_dot_f32(x_last, emb_v, d_model);
        }
    } else {
        // Escribe logits para todos los S tokens nuevos
        for (int t = 0; t < S; ++t) {
            const float* x_t = ws->x_norm + (size_t)t * (size_t)d_model;
            float* logits_t = out_logits + (size_t)t * (size_t)vocab_size;
            #pragma omp parallel for schedule(static) if (vocab_size >= 256)
            for (int v = 0; v < vocab_size; ++v) {
                const float* emb_v = model->lm_head_w + (size_t)v * (size_t)d_model;
                logits_t[v] = trace_dot_f32(x_t, emb_v, d_model);
            }
        }
    }

    // 7. Confirmar tokens en el registro del KV Cache
    trace_lm_kv_commit_tokens(kv, input_tokens + past_len, past_len, S);

    return S;
}

void trace_lm_apply_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* tokens,
    int num_tokens,
    float penalty
) {
    if (!logits || vocab_size <= 0 || !tokens || num_tokens <= 0 ||
        !(penalty > 1.0f) || !isfinite(penalty)) {
        return;
    }

    uint8_t* seen = (uint8_t*)calloc((size_t)vocab_size, sizeof(uint8_t));
    if (!seen) return;

    for (int i = 0; i < num_tokens; ++i) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size && !seen[tok]) {
            seen[tok] = 1;
            if (logits[tok] > 0.0f) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }

    free(seen);
}

int trace_lm_sample_greedy(const float* logits, int vocab_size) {
    if (!logits || vocab_size <= 0) return 0;
    int best_tok = 0;
    float best_sc = logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits[v] > best_sc) {
            best_sc = logits[v];
            best_tok = v;
        }
    }
    return best_tok;
}

typedef struct {
    int id;
    float prob;
} TraceTokenProb;

static int compare_token_probs(const void* a, const void* b) {
    const TraceTokenProb* ta = (const TraceTokenProb*)a;
    const TraceTokenProb* tb = (const TraceTokenProb*)b;
    float pa = ta->prob;
    float pb = tb->prob;
    int na = isnan(pa);
    int nb = isnan(pb);
    if (na != nb) return na ? 1 : -1;
    if (pb > pa) return 1;
    if (pb < pa) return -1;
    if (ta->id < tb->id) return -1;
    if (ta->id > tb->id) return 1;
    return 0;
}

static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    if (x == 0) x = 0x853c49e6748fea9bULL;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static inline float rand_uniform(uint64_t* rng_state) {
    if (!rng_state) return 0.5f;
    uint64_t r = xorshift64(rng_state);
    return (float)(r >> 40) * (1.0f / 16777216.0f);
}

int trace_lm_sample(
    const float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t* rng_state
) {
    if (!logits || vocab_size <= 0) return 0;
    if (!(temperature > 0.0f) || !isfinite(temperature)) {
        return trace_lm_sample_greedy(logits, vocab_size);
    }

    TraceTokenProb* items = (TraceTokenProb*)malloc(sizeof(TraceTokenProb) * (size_t)vocab_size);
    if (!items) return trace_lm_sample_greedy(logits, vocab_size);

    float max_sc = -INFINITY;
    for (int v = 0; v < vocab_size; ++v) {
        float scaled = logits[v] / temperature;
        if (scaled > max_sc) max_sc = scaled;
        items[v].id = v;
        items[v].prob = scaled;
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        float p = expf(items[v].prob - max_sc);
        items[v].prob = p;
        sum_exp += p;
    }
    float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        items[v].prob *= sum_inv;
    }

    // Ordenar de mayor a menor probabilidad para top_k y top_p
    qsort(items, (size_t)vocab_size, sizeof(TraceTokenProb), compare_token_probs);

    int n_candidates = vocab_size;

    // Top-k filtering si top_k > 0
    if (top_k > 0 && top_k < n_candidates) {
        n_candidates = top_k;
    }

    // Top-p (nucleus) filtering si top_p > 0.0f y top_p < 1.0f
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        int cutoff = 1;
        for (int i = 0; i < n_candidates; ++i) {
            cum += items[i].prob;
            cutoff = i + 1;
            if (cum >= top_p) {
                break;
            }
        }
        n_candidates = cutoff;
    }

    // Renormalizar los candidatos seleccionados
    float cum_sum = 0.0f;
    for (int i = 0; i < n_candidates; ++i) {
        cum_sum += items[i].prob;
    }
    float renorm = (cum_sum > 0.0f) ? (1.0f / cum_sum) : 1.0f;

    // Muestreo uniforme en [0, 1)
    float coin = rand_uniform(rng_state);
    float cum = 0.0f;
    int sampled_tok = items[n_candidates - 1].id;
    for (int i = 0; i < n_candidates; ++i) {
        cum += items[i].prob * renorm;
        if (coin <= cum) {
            sampled_tok = items[i].id;
            break;
        }
    }

    free(items);
    return sampled_tok;
}

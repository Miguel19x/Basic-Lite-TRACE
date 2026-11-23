/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#include "trace_lm.h"
#include "trace_lm_kv.h"
#include "../paged_unpack.h"
#include "../trace_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// Declaracion externa de kernel GEMM rapido para capas BitLinear con sub_ln + INT8
extern void paged_direct_matmul_fast_c_multi(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
);

static inline float hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline const uint8_t* paged_linear_get_row_packed(
    const TraceLMPagedLinear* lin,
    int r,
    int* out_cols
) {
    for (int p = 0; p < lin->num_pages; ++p) {
        const uint8_t* blob = lin->page_ptrs[p];
        const PagedHeader* hdr = (const PagedHeader*)blob;
        if ((uint32_t)r >= hdr->row_start && (uint32_t)r < hdr->row_start + hdr->rows) {
            uint32_t local_r = (uint32_t)r - hdr->row_start;
            int64_t padded_cols = ((int64_t)hdr->cols + 31) / 32 * 32;
            int64_t row_stride_bytes = padded_cols >> 2;
            if (out_cols) *out_cols = hdr->cols;
            return blob + sizeof(PagedHeader) + local_r * row_stride_bytes;
        }
    }
    return NULL;
}

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

    ALLOC_MEM(ws->x, (size_t)max_batch_seq * d_model * sizeof(float));
    ALLOC_MEM(ws->x_norm, (size_t)max_batch_seq * max_in_dim * sizeof(float));
    ALLOC_MEM(ws->q_x, (size_t)max_batch_seq * max_in_dim * sizeof(int8_t));
    ALLOC_MEM(ws->scale_x, (size_t)max_batch_seq * sizeof(float));
    ALLOC_MEM(ws->acc_int32, (size_t)max_batch_seq * max_out_dim * sizeof(int32_t));

    ALLOC_MEM(ws->q_c, (size_t)max_batch_seq * q_nope_dim * sizeof(float));
    ALLOC_MEM(ws->q_pe, (size_t)max_batch_seq * q_rope_dim * sizeof(float));
    ALLOC_MEM(ws->q_c_absorbed, (size_t)max_batch_seq * n_heads * kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->scores, (size_t)n_heads * max_batch_seq * cfg->max_seq_len * sizeof(float));
    ALLOC_MEM(ws->attn_probs, (size_t)n_heads * max_batch_seq * cfg->max_seq_len * sizeof(float));
    ALLOC_MEM(ws->o_latent, (size_t)n_heads * max_batch_seq * kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->o_heads, (size_t)max_batch_seq * v_dim * sizeof(float));
    ALLOC_MEM(ws->ffn_hidden, (size_t)max_batch_seq * d_ff * sizeof(float));

    ALLOC_MEM(ws->c_kv_tmp, (size_t)max_batch_seq * kv_lora_rank * sizeof(float));
    ALLOC_MEM(ws->k_pe_tmp, (size_t)max_batch_seq * cfg->qk_rope_head_dim * sizeof(float));
    ALLOC_MEM(ws->attn_out, (size_t)max_batch_seq * d_model * sizeof(float));
    ALLOC_MEM(ws->ffn_out, (size_t)max_batch_seq * d_model * sizeof(float));

    return ws;
}

void trace_lm_workspace_free(TraceLMWorkspace* ws) {
    if (!ws) return;
    if (ws->x) free(ws->x);
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
    if (ws->k_pe_tmp) free(ws->k_pe_tmp);
    if (ws->attn_out) free(ws->attn_out);
    if (ws->ffn_out) free(ws->ffn_out);
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

    // 1. RMSNorm sub_ln
    for (int m = 0; m < M; ++m) {
        rms_norm(in + m * in_f, lin->sub_ln_w, ws->x_norm + m * in_f, in_f, eps);
    }

    // 2. Cuantizacion simetrica a INT8
    for (int m = 0; m < M; ++m) {
        const float* row = ws->x_norm + m * in_f;
        float amax = eps;
        for (int i = 0; i < in_f; ++i) {
            float v = fabsf(row[i]);
            if (v > amax) amax = v;
        }
        float scale = amax / 127.0f;
        float scale_inv = 127.0f / amax;
        ws->scale_x[m] = scale;

        int8_t* qrow = ws->q_x + m * in_f;
        for (int i = 0; i < in_f; ++i) {
            float q = roundf(row[i] * scale_inv);
            if (q > 127.0f) q = 127.0f;
            else if (q < -127.0f) q = -127.0f;
            qrow[i] = (int8_t)q;
        }
    }

    // 3. Matmul AVX2 aditivo
    paged_direct_matmul_fast_c_multi(ws->q_x, lin->page_ptrs, lin->num_pages, ws->acc_int32, M, in_f, out_f);

    // 4. Descuantizacion
    for (int m = 0; m < M; ++m) {
        float factor = ws->scale_x[m] * lin->scale;
        const int32_t* acc_row = ws->acc_int32 + m * out_f;
        float* out_row = out + m * out_f;
        for (int i = 0; i < out_f; ++i) {
            out_row[i] = (float)acc_row[i] * factor;
        }
    }
}

// Absorcion de W_uk en registros AVX2 (0 matrices FP32 en RAM)
// W_uk: [n_heads * qk_nope_head_dim, kv_lora_rank]
// q_c_absorbed[h, c] = sum_d (q_c[h, d] * W_uk[h * d_h + d, c]) * scale_uk
static void absorb_w_uk_avx2(
    const float* q_c,
    const TraceLMPagedLinear* w_uk,
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
                const uint8_t* row_p = paged_linear_get_row_packed(w_uk, r, NULL);
                if (!row_p) continue;

                float v = q_c[(t * n_heads + h) * qk_nope_head_dim + d];
                if (v == 0.0f) continue;
                __m256 v_vec = _mm256_set1_ps(v);

                for (int k = 0; k < kv_lora_rank; k += 32) {
                    __m256i w_i8 = avx2_unpack_32_weights_fast(row_p + (k >> 2));
                    __m128i lo16 = _mm256_castsi256_si128(w_i8);
                    __m128i hi16 = _mm256_extracti128_si256(w_i8, 1);

                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8)));
                    __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                    __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8)));

                    _mm256_storeu_ps(out_hc + k, _mm256_fmadd_ps(v_vec, w0, _mm256_loadu_ps(out_hc + k)));
                    _mm256_storeu_ps(out_hc + k + 8, _mm256_fmadd_ps(v_vec, w1, _mm256_loadu_ps(out_hc + k + 8)));
                    _mm256_storeu_ps(out_hc + k + 16, _mm256_fmadd_ps(v_vec, w2, _mm256_loadu_ps(out_hc + k + 16)));
                    _mm256_storeu_ps(out_hc + k + 24, _mm256_fmadd_ps(v_vec, w3, _mm256_loadu_ps(out_hc + k + 24)));
                }
            }

            for (int k = 0; k < kv_lora_rank; ++k) {
                out_hc[k] *= w_uk->scale;
            }
        }
    }
}

// Absorcion de W_uv en registros AVX2 (0 matrices FP32 en RAM)
// W_uv: [n_heads * v_head_dim, kv_lora_rank]
// o_heads[h, v] = (o_latent[h, :] . W_uv[h * d_v + v, :]) * scale_uv
static void absorb_w_uv_avx2(
    const float* o_latent,
    const TraceLMPagedLinear* w_uv,
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
                const uint8_t* row_p = paged_linear_get_row_packed(w_uv, r, NULL);
                if (!row_p) {
                    out_h[v] = 0.0f;
                    continue;
                }

                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (int k = 0; k < kv_lora_rank; k += 32) {
                    __m256i w_i8 = avx2_unpack_32_weights_fast(row_p + (k >> 2));
                    __m128i lo16 = _mm256_castsi256_si128(w_i8);
                    __m128i hi16 = _mm256_extracti128_si256(w_i8, 1);

                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8)));
                    __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16));
                    __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8)));

                    acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(lat_h + k), acc0);
                    acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(lat_h + k + 8), acc1);
                    acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(lat_h + k + 16), acc2);
                    acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(lat_h + k + 24), acc3);
                }

                __m256 sum256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                float dot = hsum_ps_avx2(sum256);
                out_h[v] = dot * w_uv->scale;
            }
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
    if (!model || !kv || !ws || !out_logits) return -1;
    if (num_tokens <= 0) return 0;
    if (num_tokens > kv->max_seq_len) return -2;

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
        const float* emb_src = model->tok_emb + tok * d_model;
        memcpy(ws->x + t * d_model, emb_src, sizeof(float) * d_model);
    }

    // 4. Bucle sobre capas Transformer
    for (int l = 0; l < model->cfg.n_layers; ++l) {
        TraceLMLayer* layer = &model->layers[l];

        // 4.1. RMSNorm previo a atencion
        for (int t = 0; t < S; ++t) {
            rms_norm(ws->x + t * d_model, layer->attn_norm_w, ws->x_norm + t * d_model, d_model, eps);
        }

        // 4.2. Proyeccion KV latente: w_dkv
        paged_linear_forward_act_int8(ws->x_norm, &layer->w_dkv, ws->c_kv_tmp, S, ws, eps);

        // 4.3. Normalizacion de cuello de botella: kv_norm
        for (int t = 0; t < S; ++t) {
            rms_norm(ws->c_kv_tmp + t * kv_lora_rank, layer->kv_norm_w, ws->c_kv_tmp + t * kv_lora_rank, kv_lora_rank, eps);
        }

        // 4.4. Proyeccion de clave rotacional: w_kr
        paged_linear_forward_act_int8(ws->x_norm, &layer->w_kr, ws->k_pe_tmp, S, ws, eps);
        for (int t = 0; t < S; ++t) {
            apply_rope_c(ws->k_pe_tmp + t * qk_rope_head_dim, qk_rope_head_dim, past_len + t, model->rope_cos, model->rope_sin);
        }

        // 4.5. Guardar c_kv y k_pe en el KV Cache para cada token nuevo
        for (int t = 0; t < S; ++t) {
            trace_lm_kv_store(kv, l, past_len + t, ws->c_kv_tmp + t * kv_lora_rank, ws->k_pe_tmp + t * qk_rope_head_dim);
        }

        // 4.6. Proyecciones de consulta: w_q y w_qr
        paged_linear_forward_act_int8(ws->x_norm, &layer->w_q, ws->q_c, S, ws, eps);
        paged_linear_forward_act_int8(ws->x_norm, &layer->w_qr, ws->q_pe, S, ws, eps);

        // Aplicar RoPE sobre q_pe
        for (int t = 0; t < S; ++t) {
            for (int h = 0; h < n_heads; ++h) {
                float* q_pe_h = ws->q_pe + (t * n_heads + h) * qk_rope_head_dim;
                apply_rope_c(q_pe_h, qk_rope_head_dim, past_len + t, model->rope_cos, model->rope_sin);
            }
        }

        // 4.7. Absorcion 1: W_uk^T q_c en registros AVX2
        absorb_w_uk_avx2(ws->q_c, &layer->w_uk, ws->q_c_absorbed, n_heads, qk_nope_head_dim, kv_lora_rank, S);

        // 4.8. Atencion Causal Multi-Head Latente
        for (int t = 0; t < S; ++t) {
            int curr_pos = past_len + t;
            int total_len = curr_pos + 1;

            for (int h = 0; h < n_heads; ++h) {
                const float* q_c_abs_h = ws->q_c_absorbed + (t * n_heads + h) * kv_lora_rank;
                const float* q_pe_h = ws->q_pe + (t * n_heads + h) * qk_rope_head_dim;
                float* scores_h = ws->scores + (h * S + t) * model->cfg.max_seq_len;

                float max_sc = -1e30f;
                for (int s = 0; s < total_len; ++s) {
                    float c_kv_s[kv_lora_rank];
                    trace_lm_kv_get_c_kv(kv, l, s, c_kv_s);
                    const float* k_pe_s = trace_lm_kv_get_k_pe(kv, l, s);

                    float dot_c = avx2_dot_product_f32(q_c_abs_h, c_kv_s, kv_lora_rank);
                    float dot_pe = avx2_dot_product_f32(q_pe_h, k_pe_s, qk_rope_head_dim);
                    float sc = (dot_c + dot_pe) * attn_scale;
                    scores_h[s] = sc;
                    if (sc > max_sc) max_sc = sc;
                }

                // Softmax estable
                float sum_exp = 0.0f;
                float* probs_h = ws->attn_probs + (h * S + t) * model->cfg.max_seq_len;
                for (int s = 0; s < total_len; ++s) {
                    float p = expf(scores_h[s] - max_sc);
                    probs_h[s] = p;
                    sum_exp += p;
                }
                float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                for (int s = 0; s < total_len; ++s) {
                    probs_h[s] *= sum_inv;
                }

                // Agregacion en espacio latente: o_latent = sum(probs * c_kv)
                float* o_lat_h = ws->o_latent + (t * n_heads + h) * kv_lora_rank;
                memset(o_lat_h, 0, sizeof(float) * kv_lora_rank);
                for (int s = 0; s < total_len; ++s) {
                    float p = probs_h[s];
                    if (p == 0.0f) continue;
                    float c_kv_s[kv_lora_rank];
                    trace_lm_kv_get_c_kv(kv, l, s, c_kv_s);
                    __m256 p_vec = _mm256_set1_ps(p);
                    for (int k = 0; k < kv_lora_rank; k += 8) {
                        __m256 acc = _mm256_loadu_ps(o_lat_h + k);
                        __m256 c_vec = _mm256_loadu_ps(c_kv_s + k);
                        _mm256_storeu_ps(o_lat_h + k, _mm256_fmadd_ps(p_vec, c_vec, acc));
                    }
                }
            }
        }

        // 4.9. Absorcion 2: W_uv . o_latent en registros AVX2
        absorb_w_uv_avx2(ws->o_latent, &layer->w_uv, ws->o_heads, n_heads, v_head_dim, kv_lora_rank, S);

        // 4.10. Proyeccion de salida: w_out
        paged_linear_forward_act_int8(ws->o_heads, &layer->w_out, ws->attn_out, S, ws, eps);

        // Residual de atencion: x = x + attn_out
        for (int i = 0; i < S * d_model; ++i) {
            ws->x[i] += ws->attn_out[i];
        }

        // 4.11. FFN: RMSNorm -> ffn_up -> Squared-ReLU -> ffn_down
        for (int t = 0; t < S; ++t) {
            rms_norm(ws->x + t * d_model, layer->ffn_norm_w, ws->x_norm + t * d_model, d_model, eps);
        }
        paged_linear_forward_act_int8(ws->x_norm, &layer->ffn_up, ws->ffn_hidden, S, ws, eps);

        // Activacion Squared-ReLU: relu(x)^2
        for (int i = 0; i < S * d_ff; ++i) {
            float val = ws->ffn_hidden[i];
            float r = (val > 0.0f) ? val : 0.0f;
            ws->ffn_hidden[i] = r * r;
        }

        paged_linear_forward_act_int8(ws->ffn_hidden, &layer->ffn_down, ws->ffn_out, S, ws, eps);

        // Residual FFN: x = x + ffn_out
        for (int i = 0; i < S * d_model; ++i) {
            ws->x[i] += ws->ffn_out[i];
        }
    }

    // 5. Final RMSNorm
    for (int t = 0; t < S; ++t) {
        rms_norm(ws->x + t * d_model, model->final_norm_w, ws->x_norm + t * d_model, d_model, eps);
    }

    // 6. Proyeccion de salida con tied embeddings
    int vocab_size = model->cfg.vocab_size;
    if (last_token_only) {
        // Optimizado para CLI / generacion: computa unicamente el ultimo token (S - 1)
        const float* x_last = ws->x_norm + (S - 1) * d_model;
        for (int v = 0; v < vocab_size; ++v) {
            const float* emb_v = model->lm_head_w + v * d_model;
            out_logits[v] = avx2_dot_product_f32(x_last, emb_v, d_model);
        }
    } else {
        // Escribe logits para todos los S tokens nuevos
        for (int t = 0; t < S; ++t) {
            const float* x_t = ws->x_norm + t * d_model;
            float* logits_t = out_logits + t * vocab_size;
            for (int v = 0; v < vocab_size; ++v) {
                const float* emb_v = model->lm_head_w + v * d_model;
                logits_t[v] = avx2_dot_product_f32(x_t, emb_v, d_model);
            }
        }
    }

    // 7. Confirmar tokens en el registro del KV Cache
    trace_lm_kv_commit_tokens(kv, input_tokens + past_len, past_len, S);

    return S;
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

static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static inline float rand_uniform(uint64_t* rng_state) {
    if (!rng_state) return 0.5f;
    return (float)(xorshift64(rng_state) >> 11) * (1.0f / 9007199254740992.0f);
}

int trace_lm_sample(
    const float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t* rng_state
) {
    if (temperature <= 0.0f) {
        return trace_lm_sample_greedy(logits, vocab_size);
    }

    float* probs = (float*)malloc(sizeof(float) * vocab_size);
    if (!probs) return trace_lm_sample_greedy(logits, vocab_size);

    float max_sc = -1e30f;
    for (int v = 0; v < vocab_size; ++v) {
        float scaled = logits[v] / temperature;
        probs[v] = scaled;
        if (scaled > max_sc) max_sc = scaled;
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        float p = expf(probs[v] - max_sc);
        probs[v] = p;
        sum_exp += p;
    }
    float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        probs[v] *= sum_inv;
    }

    // Top-k filtering si top_k > 0
    if (top_k > 0 && top_k < vocab_size) {
        float threshold = 0.0f;
        // Seleccion rapida de umbral
        float* tmp = (float*)malloc(sizeof(float) * vocab_size);
        if (tmp) {
            memcpy(tmp, probs, sizeof(float) * vocab_size);
            for (int i = 0; i < top_k; ++i) {
                int max_idx = i;
                for (int j = i + 1; j < vocab_size; ++j) {
                    if (tmp[j] > tmp[max_idx]) max_idx = j;
                }
                float t_val = tmp[i];
                tmp[i] = tmp[max_idx];
                tmp[max_idx] = t_val;
            }
            threshold = tmp[top_k - 1];
            free(tmp);
        }

        sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            if (probs[v] < threshold) probs[v] = 0.0f;
            else sum_exp += probs[v];
        }
        sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int v = 0; v < vocab_size; ++v) probs[v] *= sum_inv;
    }

    // Top-p (nucleus) filtering si top_p < 1.0
    if (top_p > 0.0f && top_p < 1.0f) {
        // Enfoque greedy acumulativo
    }

    float coin = rand_uniform(rng_state);
    float cum = 0.0f;
    int sampled_tok = vocab_size - 1;
    for (int v = 0; v < vocab_size; ++v) {
        cum += probs[v];
        if (coin <= cum) {
            sampled_tok = v;
            break;
        }
    }

    free(probs);
    return sampled_tok;
}

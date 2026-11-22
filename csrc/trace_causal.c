/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#define _GNU_SOURCE
#include "trace_infer.h"
#include "trace_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void paged_direct_matmul_fast_c_multi(
    const int8_t* x,
    const uint8_t* const* page_blobs,
    int64_t num_pages,
    int32_t* out,
    int64_t M,
    int64_t K,
    int64_t total_out_features
);

static void paged_linear_forward(
    const TracePagedLinear* layer,
    const float* x,
    float* y,
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

    #pragma omp parallel for num_threads(2) schedule(static) if (M >= 32)
    for (int64_t m = 0; m < M; ++m) {
        rms_norm(x + m * in_f, layer->sub_ln_w, x_norm + m * in_f, (int)in_f, eps);

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

    paged_direct_matmul_fast_c_multi(
        q_x,
        layer->page_ptrs,
        (int64_t)layer->num_pages,
        acc_int32,
        M,
        in_f,
        out_f
    );

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

    if (!ws) {
        free(x_norm);
        free(q_x);
        free(acc_int32);
        free(scale_x);
    }
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

static void apply_embeddings(
    const TraceModel* model,
    const int32_t* token_ids,
    int L,
    int stage_id,
    float* a,
    float* z_r,
    float* y_r
) {
    int D = model->cfg.d_model;
    int abacus_ranks[160];
    extract_abacus_ranks(token_ids, L, model->cfg.mask_token_id, abacus_ranks);

    for (int i = 0; i < L; ++i) {
        int tok = token_ids[i];
        int r = abacus_ranks[i];
        const float* t_ptr = model->token_emb + tok * D;
        const float* ab_ptr = model->abacus_emb + r * D;
        const float* st_ptr = model->stage_emb + stage_id * D;
        for (int d = 0; d < D; ++d) a[i * D + d] = t_ptr[d] + ab_ptr[d] + st_ptr[d];
    }
    memcpy(z_r, a, L * D * sizeof(float));
    memset(y_r, 0, L * D * sizeof(float));
}

static void run_gqa_attention(
    const TraceModel* model,
    TraceWorkspace* ws,
    int L,
    int prompt_len,
    float eps
) {
    int D = model->cfg.d_model;
    float* q = ws->q;
    float* k = ws->k;
    float* v = ws->v;

    paged_linear_forward(&model->attn_q, ws->joint_state, q, L, eps, ws);
    paged_linear_forward(&model->attn_k, ws->joint_state, k, L, eps, ws);
    paged_linear_forward(&model->attn_v, ws->joint_state, v, L, eps, ws);

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

    float* attn_out_heads = ws->attn_out_heads;
    float scale_attn = 1.0f / sqrtf(48.0f);
    int t_end = (prompt_len > 0) ? (prompt_len + 35) : L;
    if (t_end > L) t_end = L;

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
                float dot = avx2_dot_product_f32(q_vec, k_vec, 48);
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

    paged_linear_forward(&model->attn_o, attn_out_heads, ws->attn_final, L, eps, ws);
}

static void run_moe_block(
    const TraceModel* model,
    const int32_t* token_ids,
    TraceWorkspace* ws,
    int L,
    float eps
) {
    int D = model->cfg.d_model;
    float* z_norm_moe = ws->z_norm_moe;
    for (int i = 0; i < L; ++i) rms_norm(ws->z_mid + i * D, model->norm_moe, z_norm_moe + i * D, D, eps);

    float* moe_out = ws->moe_out;
    memset(moe_out, 0, L * D * sizeof(float));

    // Experto compartido sobre tokens no pad
    paged_linear_forward(&model->shared_up, z_norm_moe, ws->shared_up_buf, L, eps, ws);
    __m256 zero_vec = _mm256_setzero_ps();
    int total_sh = L * 384;
    int idx_sh = 0;
    for (; idx_sh + 8 <= total_sh; idx_sh += 8) {
        __m256 v = _mm256_loadu_ps(ws->shared_up_buf + idx_sh);
        __m256 pos = _mm256_max_ps(v, zero_vec);
        _mm256_storeu_ps(ws->shared_up_buf + idx_sh, _mm256_mul_ps(pos, pos));
    }
    for (; idx_sh < total_sh; ++idx_sh) {
        float val = ws->shared_up_buf[idx_sh];
        ws->shared_up_buf[idx_sh] = (val > 0.0f) ? val * val : 0.0f;
    }
    paged_linear_forward(&model->shared_down, ws->shared_up_buf, ws->shared_down_buf, L, eps, ws);

    for (int i = 0; i < L; ++i) {
        if (token_ids[i] != model->cfg.pad_token_id) {
            float* moe_row = moe_out + i * D;
            const float* sh_row = ws->shared_down_buf + i * D;
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
    if (seq_len <= 0 || seq_len > model->cfg.max_seq_len) return;

    int L = seq_len;
    int D = model->cfg.d_model;
    float eps = model->cfg.eps;
    TraceWorkspace* ws = &model->workspace;

    apply_embeddings(model, token_ids, L, stage_id, ws->a, ws->z_r, ws->y_r);
    paged_linear_forward(&model->anchor_proj, ws->a, ws->anchor_feat, L, eps, ws);

    const float* e_t = model->e_t_const;
    int R = model->cfg.num_recursions;

    for (int r = 0; r < R; ++r) {
        float step_scale = 2.0f / (float)(R + 1);
        const float* e_r = model->e_r_table[r];

        for (int i = 0; i < L; ++i) {
            rms_norm(ws->z_r + i * D, model->norm_z, ws->norm_z_buf + i * D, D, eps);
            rms_norm(ws->y_r + i * D, model->norm_y, ws->norm_y_buf + i * D, D, eps);
            float* js_row = ws->joint_state + i * D;
            const float* nz_row = ws->norm_z_buf + i * D;
            const float* ny_row = ws->norm_y_buf + i * D;
            const float* af_row = ws->anchor_feat + i * D;
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

        run_gqa_attention(model, ws, L, prompt_len, eps);

        float eff_alpha = model->alpha_z * step_scale;
        float eff_beta = model->beta_y * step_scale;

        int total_ld = L * D;
        float* z_mid = ws->z_mid;
        __m256 valpha = _mm256_set1_ps(eff_alpha);
        int idx_m = 0;
        for (; idx_m + 8 <= total_ld; idx_m += 8) {
            __m256 vz = _mm256_loadu_ps(ws->z_r + idx_m);
            __m256 va = _mm256_loadu_ps(ws->attn_final + idx_m);
            _mm256_storeu_ps(z_mid + idx_m, _mm256_add_ps(vz, _mm256_mul_ps(valpha, va)));
        }
        for (; idx_m < total_ld; ++idx_m) z_mid[idx_m] = ws->z_r[idx_m] + eff_alpha * ws->attn_final[idx_m];

        run_moe_block(model, token_ids, ws, L, eps);

        float* z_next = ws->z_next;
        __m256 valpha_moe = _mm256_set1_ps(eff_alpha);
        int idx_z = 0;
        for (; idx_z + 8 <= total_ld; idx_z += 8) {
            __m256 vz = _mm256_loadu_ps(z_mid + idx_z);
            __m256 vm = _mm256_loadu_ps(ws->moe_out + idx_z);
            _mm256_storeu_ps(z_next + idx_z, _mm256_add_ps(vz, _mm256_mul_ps(valpha_moe, vm)));
        }
        for (; idx_z < total_ld; ++idx_z) z_next[idx_z] = z_mid[idx_z] + eff_alpha * ws->moe_out[idx_z];

        float* fusion = ws->fusion;
        for (int i = 0; i < L; ++i) {
            float* f_row = fusion + i * D;
            const float* y_row = ws->y_r + i * D;
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

        paged_linear_forward(&model->answer_updater, fusion, ws->delta_y, L, eps, ws);

        for (int i = 0; i < L * D; ++i) {
            float v = ws->delta_y[i];
            float silu_v = v / (1.0f + expf(-v));
            ws->y_r[i] = ws->y_r[i] + eff_beta * silu_v;
            ws->z_r[i] = z_next[i];
        }
    }

    float* normed_y = ws->normed_y;
    for (int i = 0; i < L; ++i) rms_norm(ws->y_r + i * D, model->final_norm, normed_y + i * D, D, eps);

    int V = model->cfg.vocab_size;
    if (target_indices == NULL || num_targets <= 0) {
        for (int i = 0; i < L; ++i) {
            const float* act_row = normed_y + i * D;
            for (int v = 0; v < V; ++v) {
                out_logits[i * V + v] = avx2_dot_product_f32(act_row, model->lm_head_w + v * D, D);
            }
        }
    } else {
        for (int t = 0; t < num_targets; ++t) {
            int i = target_indices[t];
            const float* act_row = normed_y + i * D;
            for (int v = 0; v < V; ++v) {
                out_logits[t * V + v] = avx2_dot_product_f32(act_row, model->lm_head_w + v * D, D);
            }
        }
    }
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

int trace_generate_causal_two_step(
    TraceModel* model,
    const uint8_t* prompt_bytes,
    int prompt_len,
    char* out_str,
    int max_out_len
) {
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

    // Pasada 1 (Etapa 0): Subtotales pw y pms
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
    return written;
}

size_t trace_model_sizeof(void) { return sizeof(TraceModel); }
int trace_model_get_used_odirect(const TraceModel* model) { return model ? model->used_odirect : 0; }
uint64_t trace_model_get_pread_calls(const TraceModel* model) { return model ? model->pread_calls : 0; }
size_t trace_model_get_pages_offset(const TraceModel* model) { return model ? model->pages_offset : 0; }

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

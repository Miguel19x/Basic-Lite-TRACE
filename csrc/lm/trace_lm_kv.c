/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "trace_lm_kv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

TraceLMKVCache* trace_lm_kv_create(const TraceLMConfig* cfg, int max_seq_len, int use_int8_kv) {
    if (!cfg || max_seq_len <= 0) return NULL;

    TraceLMKVCache* kv = (TraceLMKVCache*)calloc(1, sizeof(TraceLMKVCache));
    if (!kv) return NULL;

    kv->max_seq_len = max_seq_len;
    kv->current_seq_len = 0;
    kv->last_common_prefix = 0;
    kv->use_int8_kv = use_int8_kv;
    kv->n_layers = cfg->n_layers;
    kv->kv_lora_rank = cfg->kv_lora_rank;
    kv->qk_rope_head_dim = cfg->qk_rope_head_dim;
    kv->attn_type = cfg->attn_type;
    kv->n_kv_heads = cfg->n_kv_heads;
    kv->head_dim = cfg->head_dim;

    kv->cached_tokens = (int*)malloc(sizeof(int) * (size_t)max_seq_len);
    if (!kv->cached_tokens) {
        free(kv);
        return NULL;
    }

    if (cfg->attn_type == TRACE_ATTN_GQA) {
        size_t kv_dim = (size_t)cfg->n_kv_heads * (size_t)cfg->head_dim;
        size_t total_elems = (size_t)cfg->n_layers * (size_t)max_seq_len * kv_dim;
        if (use_int8_kv) {
            size_t total_scales = (size_t)cfg->n_layers * (size_t)max_seq_len * (size_t)cfg->n_kv_heads;
            kv->gqa_k_i8 = (int8_t*)malloc(sizeof(int8_t) * total_elems);
            kv->gqa_k_scales = (float*)malloc(sizeof(float) * total_scales);
            kv->gqa_v_i8 = (int8_t*)malloc(sizeof(int8_t) * total_elems);
            kv->gqa_v_scales = (float*)malloc(sizeof(float) * total_scales);
            if (!kv->gqa_k_i8 || !kv->gqa_k_scales || !kv->gqa_v_i8 || !kv->gqa_v_scales) {
                trace_lm_kv_free(kv);
                return NULL;
            }
        } else {
            kv->gqa_k_fp32 = (float*)malloc(sizeof(float) * total_elems);
            kv->gqa_v_fp32 = (float*)malloc(sizeof(float) * total_elems);
            if (!kv->gqa_k_fp32 || !kv->gqa_v_fp32) {
                trace_lm_kv_free(kv);
                return NULL;
            }
        }
        return kv;
    }

    size_t k_pe_total = (size_t)cfg->n_layers * (size_t)max_seq_len * (size_t)cfg->qk_rope_head_dim;
    kv->k_pe = (float*)malloc(sizeof(float) * k_pe_total);
    if (!kv->k_pe) {
        free(kv->cached_tokens);
        free(kv);
        return NULL;
    }

    // Ahorro real de RAM: solo se reserva el buffer correspondiente a la precision activa
    if (use_int8_kv) {
        size_t c_kv_total = (size_t)cfg->n_layers * (size_t)max_seq_len * (size_t)cfg->kv_lora_rank;
        kv->c_kv_i8 = (int8_t*)malloc(sizeof(int8_t) * c_kv_total);
        kv->c_kv_scales = (float*)malloc(sizeof(float) * (size_t)cfg->n_layers * (size_t)max_seq_len);
        kv->c_kv_fp32 = NULL;
        if (!kv->c_kv_i8 || !kv->c_kv_scales) {
            trace_lm_kv_free(kv);
            return NULL;
        }
    } else {
        size_t c_kv_total = (size_t)cfg->n_layers * (size_t)max_seq_len * (size_t)cfg->kv_lora_rank;
        kv->c_kv_fp32 = (float*)malloc(sizeof(float) * c_kv_total);
        kv->c_kv_i8 = NULL;
        kv->c_kv_scales = NULL;
        if (!kv->c_kv_fp32) {
            trace_lm_kv_free(kv);
            return NULL;
        }
    }

    return kv;
}

void trace_lm_kv_free(TraceLMKVCache* kv) {
    if (!kv) return;
    if (kv->cached_tokens) free(kv->cached_tokens);
    if (kv->k_pe) free(kv->k_pe);
    if (kv->c_kv_i8) free(kv->c_kv_i8);
    if (kv->c_kv_scales) free(kv->c_kv_scales);
    if (kv->c_kv_fp32) free(kv->c_kv_fp32);
    if (kv->gqa_k_i8) free(kv->gqa_k_i8);
    if (kv->gqa_k_scales) free(kv->gqa_k_scales);
    if (kv->gqa_v_i8) free(kv->gqa_v_i8);
    if (kv->gqa_v_scales) free(kv->gqa_v_scales);
    if (kv->gqa_k_fp32) free(kv->gqa_k_fp32);
    if (kv->gqa_v_fp32) free(kv->gqa_v_fp32);
    free(kv);
}

void trace_lm_kv_reset(TraceLMKVCache* kv) {
    if (!kv) return;
    kv->current_seq_len = 0;
    kv->last_common_prefix = 0;
}

int trace_lm_kv_find_common_prefix(TraceLMKVCache* kv, const int* tokens, int num_tokens) {
    if (!kv || !tokens || num_tokens <= 0) return 0;
    int pfx = 0;
    while (pfx < kv->current_seq_len && pfx < num_tokens && kv->cached_tokens[pfx] == tokens[pfx]) {
        pfx++;
    }
    kv->last_common_prefix = pfx;
    return pfx;
}

void trace_lm_kv_commit_tokens(TraceLMKVCache* kv, const int* tokens, int start_pos, int count) {
    if (!kv || !tokens || start_pos < 0 || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        if (start_pos + i < kv->max_seq_len) {
            kv->cached_tokens[start_pos + i] = tokens[i];
        }
    }
    // Trunca explicitamente la longitud para descartar ramas divergentes previas
    int new_len = start_pos + count;
    if (new_len > kv->max_seq_len) new_len = kv->max_seq_len;
    kv->current_seq_len = new_len;
}

void trace_lm_kv_store(
    TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    const float* c_kv_token,
    const float* k_pe_token
) {
    if (!kv || !c_kv_token || !k_pe_token ||
        kv->attn_type != TRACE_ATTN_MLA || !kv->k_pe ||
        layer_idx < 0 || layer_idx >= kv->n_layers ||
        pos < 0 || pos >= kv->max_seq_len) {
        return;
    }

    // 1. Guardar k_pe en float32
    size_t k_pe_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->qk_rope_head_dim;
    memcpy(kv->k_pe + k_pe_offset, k_pe_token, sizeof(float) * (size_t)kv->qk_rope_head_dim);

    // 2. Guardar c_kv (INT8 con escala o FP32 continuo)
    if (kv->use_int8_kv) {
        float amax = 1e-6f;
        for (int c = 0; c < kv->kv_lora_rank; ++c) {
            float v = fabsf(c_kv_token[c]);
            if (v > amax) amax = v;
        }
        float scale = amax / 127.0f;
        float scale_inv = 127.0f / amax;

        size_t scale_idx = (size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos;
        kv->c_kv_scales[scale_idx] = scale;

        size_t i8_offset = scale_idx * (size_t)kv->kv_lora_rank;
        for (int c = 0; c < kv->kv_lora_rank; ++c) {
            float q = nearbyintf(c_kv_token[c] * scale_inv);
            if (!isfinite(q)) q = 0.0f;
            else if (q > 127.0f) q = 127.0f;
            else if (q < -127.0f) q = -127.0f;
            kv->c_kv_i8[i8_offset + (size_t)c] = (int8_t)q;
        }
    } else {
        size_t fp32_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->kv_lora_rank;
        memcpy(kv->c_kv_fp32 + fp32_offset, c_kv_token, sizeof(float) * (size_t)kv->kv_lora_rank);
    }
}

void trace_lm_kv_get_c_kv(
    const TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    float* c_kv_out
) {
    if (!kv || !c_kv_out ||
        layer_idx < 0 || layer_idx >= kv->n_layers ||
        pos < 0 || pos >= kv->max_seq_len) {
        return;
    }

    if (kv->use_int8_kv) {
        size_t scale_idx = (size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos;
        float scale = kv->c_kv_scales[scale_idx];
        size_t i8_offset = scale_idx * (size_t)kv->kv_lora_rank;
        const int8_t* src = kv->c_kv_i8 + i8_offset;
        for (int c = 0; c < kv->kv_lora_rank; ++c) {
            c_kv_out[c] = (float)src[c] * scale;
        }
    } else {
        size_t fp32_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->kv_lora_rank;
        memcpy(c_kv_out, kv->c_kv_fp32 + fp32_offset, sizeof(float) * (size_t)kv->kv_lora_rank);
    }
}

const float* trace_lm_kv_get_k_pe(
    const TraceLMKVCache* kv,
    int layer_idx,
    int pos
) {
    if (!kv || layer_idx < 0 || layer_idx >= kv->n_layers || pos < 0 || pos >= kv->max_seq_len) {
        return NULL;
    }
    size_t k_pe_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->qk_rope_head_dim;
    return kv->k_pe + k_pe_offset;
}

int trace_lm_kv_get_last_common_prefix(const TraceLMKVCache* kv) {
    return kv ? kv->last_common_prefix : 0;
}

static inline void quantize_vec_int8_symmetric(
    const float* src,
    int8_t* dst,
    float* out_scale,
    int n
) {
    float amax = 1e-6f;
    for (int i = 0; i < n; ++i) {
        float v = fabsf(src[i]);
        if (v > amax) amax = v;
    }
    float scale = amax / 127.0f;
    float scale_inv = 127.0f / amax;
    *out_scale = scale;
    for (int i = 0; i < n; ++i) {
        float q = nearbyintf(src[i] * scale_inv);
        if (!isfinite(q)) q = 0.0f;
        else if (q > 127.0f) q = 127.0f;
        else if (q < -127.0f) q = -127.0f;
        dst[i] = (int8_t)q;
    }
}

void trace_lm_kv_store_gqa(
    TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    const float* k_token,
    const float* v_token
) {
    if (!kv || !k_token || !v_token ||
        kv->attn_type != TRACE_ATTN_GQA ||
        layer_idx < 0 || layer_idx >= kv->n_layers ||
        pos < 0 || pos >= kv->max_seq_len) {
        return;
    }

    size_t kv_dim = (size_t)kv->n_kv_heads * (size_t)kv->head_dim;
    size_t base_slot = (size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos;

    if (kv->use_int8_kv) {
        for (int kh = 0; kh < kv->n_kv_heads; ++kh) {
            size_t scale_idx = base_slot * (size_t)kv->n_kv_heads + (size_t)kh;
            size_t elem_off = scale_idx * (size_t)kv->head_dim;
            const float* k_src = k_token + (size_t)kh * (size_t)kv->head_dim;
            const float* v_src = v_token + (size_t)kh * (size_t)kv->head_dim;
            quantize_vec_int8_symmetric(k_src, kv->gqa_k_i8 + elem_off, &kv->gqa_k_scales[scale_idx], kv->head_dim);
            quantize_vec_int8_symmetric(v_src, kv->gqa_v_i8 + elem_off, &kv->gqa_v_scales[scale_idx], kv->head_dim);
        }
    } else {
        size_t elem_off = base_slot * kv_dim;
        memcpy(kv->gqa_k_fp32 + elem_off, k_token, sizeof(float) * kv_dim);
        memcpy(kv->gqa_v_fp32 + elem_off, v_token, sizeof(float) * kv_dim);
    }
}

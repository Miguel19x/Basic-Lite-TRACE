/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
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

    kv->cached_tokens = (int*)malloc(sizeof(int) * max_seq_len);
    if (!kv->cached_tokens) {
        free(kv);
        return NULL;
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
    if (!kv || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        if (start_pos + i < kv->max_seq_len) {
            kv->cached_tokens[start_pos + i] = tokens[i];
        }
    }
    // Trunca explicitamente la longitud para descartar ramas divergentes previas
    kv->current_seq_len = start_pos + count;
}

void trace_lm_kv_store(
    TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    const float* c_kv_token,
    const float* k_pe_token
) {
    if (!kv || pos >= kv->max_seq_len) return;

    // 1. Guardar k_pe en float32
    size_t k_pe_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->qk_rope_head_dim;
    memcpy(kv->k_pe + k_pe_offset, k_pe_token, sizeof(float) * kv->qk_rope_head_dim);

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
            float q = roundf(c_kv_token[c] * scale_inv);
            if (q > 127.0f) q = 127.0f;
            else if (q < -127.0f) q = -127.0f;
            kv->c_kv_i8[i8_offset + c] = (int8_t)q;
        }
    } else {
        size_t fp32_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->kv_lora_rank;
        memcpy(kv->c_kv_fp32 + fp32_offset, c_kv_token, sizeof(float) * kv->kv_lora_rank);
    }
}

void trace_lm_kv_get_c_kv(
    const TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    float* c_kv_out
) {
    if (!kv || pos >= kv->max_seq_len) return;

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
        memcpy(c_kv_out, kv->c_kv_fp32 + fp32_offset, sizeof(float) * kv->kv_lora_rank);
    }
}

const float* trace_lm_kv_get_k_pe(
    const TraceLMKVCache* kv,
    int layer_idx,
    int pos
) {
    if (!kv || pos >= kv->max_seq_len) return NULL;
    size_t k_pe_offset = ((size_t)layer_idx * (size_t)kv->max_seq_len + (size_t)pos) * (size_t)kv->qk_rope_head_dim;
    return kv->k_pe + k_pe_offset;
}

int trace_lm_kv_get_last_common_prefix(const TraceLMKVCache* kv) {
    return kv ? kv->last_common_prefix : 0;
}

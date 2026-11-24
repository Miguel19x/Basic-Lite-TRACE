/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#ifndef TRACE_LM_KV_H
#define TRACE_LM_KV_H

#include "trace_lm.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TraceLMKVCache {
    int max_seq_len;
    int current_seq_len;
    int last_common_prefix; // Registro del prefijo comuncacheado en la ultima consulta
    int use_int8_kv;        // 1: INT8 [-127, 127] + escala, 0: FP32 continuo
    int n_layers;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int* cached_tokens;     // [max_seq_len]
    // Memoria contigua por capa
    int8_t* c_kv_i8;        // [n_layers * max_seq_len * kv_lora_rank] (solo si use_int8_kv == 1)
    float* c_kv_scales;     // [n_layers * max_seq_len] (solo si use_int8_kv == 1)
    float* c_kv_fp32;       // [n_layers * max_seq_len * kv_lora_rank] (solo si use_int8_kv == 0)
    float* k_pe;            // [n_layers * max_seq_len * qk_rope_head_dim] (FP32)
} TraceLMKVCache;

TraceLMKVCache* trace_lm_kv_create(const TraceLMConfig* cfg, int max_seq_len, int use_int8_kv);
void trace_lm_kv_free(TraceLMKVCache* kv);
void trace_lm_kv_reset(TraceLMKVCache* kv);

// Prefix caching: halla la longitud del prefijo comun sin recomputo
int trace_lm_kv_find_common_prefix(TraceLMKVCache* kv, const int* tokens, int num_tokens);
void trace_lm_kv_commit_tokens(TraceLMKVCache* kv, const int* tokens, int start_pos, int count);

// Almacenamiento y recuperacion de tokens
void trace_lm_kv_store(
    TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    const float* c_kv_token,
    const float* k_pe_token
);

void trace_lm_kv_get_c_kv(
    const TraceLMKVCache* kv,
    int layer_idx,
    int pos,
    float* c_kv_out
);

const float* trace_lm_kv_get_k_pe(
    const TraceLMKVCache* kv,
    int layer_idx,
    int pos
);

// Getter para ctypes
int trace_lm_kv_get_last_common_prefix(const TraceLMKVCache* kv);

#ifdef __cplusplus
}
#endif

#endif // TRACE_LM_KV_H

/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#ifndef TRACE_LM_H
#define TRACE_LM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Declaración opaca de KV Cache
struct TraceLMKVCache;

#define TRACE_ATTN_MLA 0
#define TRACE_ATTN_GQA 1

#define TRACE_EMB_FP32 0
#define TRACE_EMB_FP16 1
#define TRACE_EMB_INT8 2

typedef struct {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
    int d_ff;
    int max_seq_len;
    float eps;
    int kv_lora_rank;      // d_c (MLA)
    int qk_nope_head_dim;  // d_h (MLA)
    int qk_rope_head_dim;  // d_pe (MLA) o head_dim (GQA)
    int v_head_dim;        // d_v (MLA) o head_dim (GQA)
    int tie_word_embeddings;
    int pad_token_id;
    int mask_token_id;
    int bos_token_id;
    int eos_token_id;
    // Extensiones GQA / BitNet-2B-4T
    int attn_type;         // 0 = MLA, 1 = GQA
    int n_kv_heads;        // Cabezas KV en GQA
    int head_dim;          // Dimension por cabeza en GQA
    float rope_theta;      // Base RoPE (10000.0 por defecto)
    int rope_interleaved;  // 1 = intercalado, 0 = split-half
    int use_gated_ffn;     // 0 = ReLU^2(up), 1 = ReLU^2(gate) * up
} TraceLMConfig;

typedef struct {
    int in_features;
    int out_features;
    int num_pages;
    const uint8_t* const* page_ptrs;
    float scale;
    float* sub_ln_w;       // NULL si use_sub_ln == 0 (w_uk, w_uv)
} TraceLMPagedLinear;

typedef struct {
    float* attn_norm_w;    // [d_model]
    float* ffn_norm_w;     // [d_model]
    float* kv_norm_w;      // [kv_lora_rank]
    TraceLMPagedLinear w_dkv;
    TraceLMPagedLinear w_uk;
    TraceLMPagedLinear w_uv;
    TraceLMPagedLinear w_kr;
    TraceLMPagedLinear w_q;
    TraceLMPagedLinear w_qr;
    TraceLMPagedLinear w_k;    // GQA
    TraceLMPagedLinear w_v;    // GQA
    TraceLMPagedLinear w_out;
    TraceLMPagedLinear ffn_gate; // Gated FFN (BitNet-2B-4T)
    TraceLMPagedLinear ffn_up;
    TraceLMPagedLinear ffn_down;
} TraceLMLayer;

typedef struct {
    TraceLMConfig cfg;
    float* tok_emb;        // [vocab_size * d_model] si tok_emb_dtype == TRACE_EMB_FP32
    float* lm_head_w;      // [vocab_size * d_model] si lm_head_dtype == TRACE_EMB_FP32
    float* final_norm_w;   // [d_model]
    TraceLMLayer* layers;  // [n_layers]
    float* rope_cos;       // [max_seq_len * qk_rope_head_dim] o [max_seq_len * (head_dim/2)]
    float* rope_sin;       // [max_seq_len * qk_rope_head_dim] o [max_seq_len * (head_dim/2)]
    void* page_pool;       // Buffer alineado a 4096 bytes cargado en 1 pread
    size_t page_pool_bytes;
    int pread_calls;       // Syscalls pread (cabecera + pool contiguo = 2)
    int used_odirect;      // 1: O_DIRECT activo, 0: lectura estandar
    int fd;
    // Embeddings compactos (FP16 / INT8 por fila)
    int tok_emb_dtype;
    uint16_t* tok_emb_f16;
    int8_t* tok_emb_i8;
    float* tok_emb_scales;
    int lm_head_dtype;
    uint16_t* lm_head_f16;
    int8_t* lm_head_i8;
    float* lm_head_scales;
} TraceLMModel;

typedef struct {
    int max_batch_seq;
    int max_in_dim;
    int max_out_dim;
    float* x;              // [max_batch_seq * d_model]
    float* h_norm;         // [max_batch_seq * d_model] (salida de attn_norm / ffn_norm)
    float* x_norm;         // [max_batch_seq * max_in_dim] (exclusivo para sub_ln)
    int8_t* q_x;           // [max_batch_seq * max_in_dim]
    float* scale_x;        // [max_batch_seq]
    int32_t* acc_int32;    // [max_batch_seq * max_out_dim]
    float* q_c;            // [max_batch_seq * n_heads * qk_nope_head_dim]
    float* q_pe;           // [max_batch_seq * n_heads * qk_rope_head_dim]
    float* q_c_absorbed;   // [max_batch_seq * n_heads * kv_lora_rank]
    float* scores;         // [n_heads * max_batch_seq * max_seq_len]
    float* attn_probs;     // [n_heads * max_batch_seq * max_seq_len]
    float* o_latent;       // [n_heads * max_batch_seq * kv_lora_rank]
    float* o_heads;        // [max_batch_seq * n_heads * v_head_dim]
    float* ffn_hidden;     // [max_batch_seq * d_ff]
    // Buffers scratchpad dedicados
    float* c_kv_tmp;       // [max_batch_seq * kv_lora_rank]
    float* c_kv_tmp_single;// [kv_lora_rank] (scratchpad sin VLA para atencion)
    float* k_pe_tmp;       // [max_batch_seq * qk_rope_head_dim]
    float* attn_out;       // [max_batch_seq * d_model]
    float* ffn_out;        // [max_batch_seq * d_model]
    float* c_kv_dequant;   // [max_seq_len * kv_lora_rank] (descuantizacion unica por capa MLA)
    int max_threads;       // Hilos maximos reservados en ws->scores
    // Buffers opcionales GQA / Gated FFN (NULL si MLA sin compuerta)
    float* ffn_gate_buf;   // [max_batch_seq * d_ff]
    float* gqa_k_tmp;      // [max_batch_seq * n_kv_heads * head_dim]
    float* gqa_v_tmp;      // [max_batch_seq * n_kv_heads * head_dim]
    float* gqa_k_dequant;  // [max_seq_len * n_kv_heads * head_dim]
    float* gqa_v_dequant;  // [max_seq_len * n_kv_heads * head_dim]
} TraceLMWorkspace;

// Carga y destrucción de modelo
TraceLMModel* trace_lm_load(const char* bin_path);
void trace_lm_free(TraceLMModel* model);

// Workspace
TraceLMWorkspace* trace_lm_workspace_create(const TraceLMConfig* cfg, int max_batch_seq);
void trace_lm_workspace_free(TraceLMWorkspace* ws);

// Inferencia autorregresiva
int trace_lm_forward(
    TraceLMModel* model,
    struct TraceLMKVCache* kv,
    const int* input_tokens,
    int num_tokens,
    float* out_logits,
    int last_token_only,
    TraceLMWorkspace* ws
);

// Muestreo y penalizacion
void trace_lm_apply_repetition_penalty(
    float* logits,
    int vocab_size,
    const int* tokens,
    int num_tokens,
    float penalty
);
int trace_lm_sample_greedy(const float* logits, int vocab_size);
int trace_lm_sample(
    const float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    uint64_t* rng_state
);

// Getters ABI para ctypes
int trace_lm_get_pread_calls(const TraceLMModel* model);
int trace_lm_get_used_odirect(const TraceLMModel* model);
const TraceLMConfig* trace_lm_get_config(const TraceLMModel* model);
int trace_lm_get_vocab_size(const TraceLMModel* model);
int trace_lm_get_tok_emb_dtype(const TraceLMModel* model);
int trace_lm_get_attn_type(const TraceLMModel* model);

#ifdef __cplusplus
}
#endif

#endif // TRACE_LM_H

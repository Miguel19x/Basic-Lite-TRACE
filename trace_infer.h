#ifndef TRACE_INFER_H
#define TRACE_INFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int d_model;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int d_ff_expert;
    int num_experts;
    int num_recursions;
    int block_size;
    int max_seq_len;
    float eps;
    int pad_token_id;
    int mask_token_id;
    int bos_token_id;
    int eos_token_id;
} TraceConfig;

typedef struct {
    int in_features;
    int out_features;
    int num_pages;
    const uint8_t* const* page_ptrs; // Punteros a las páginas dentro de model->page_pool
    float scale;                     // Leída dinámicamente desde PagedHeader
    float* sub_ln_w;                 // RMSNorm sub_ln [in_features] (0 pesos FP32 en W)
} TracePagedLinear;

// Búferes prealocados para inferencia sin asignación dinámica en el heap (Zero-Malloc)
typedef struct {
    float* x_norm;
    int8_t* q_x;
    int32_t* acc_int32;
    float* scale_x;

    float* a;
    float* z_r;
    float* y_r;
    float* anchor_feat;
    float* joint_state;
    float* norm_z_buf;
    float* norm_y_buf;
    float* e_r;
    float* e_t;
    float* q;
    float* k;
    float* v;
    float* attn_out_heads;
    float* attn_final;
    float* z_mid;
    float* z_norm_moe;
    float* moe_out;
    float* shared_up_buf;
    float* shared_down_buf;
    float* fusion;
    float* delta_y;
    float* z_next;
    float* normed_y;

    float* exp_tokens_in;
    float* exp_up_buf;
    float* exp_down_buf;
    int*   exp_tok_idx;
    float* exp_tok_gate;
} TraceWorkspace;

typedef struct {
    TraceConfig cfg;
    int fd;
    size_t num_pages;
    size_t page_size;
    size_t pages_offset;             // pages_offset % 4096 == 0
    uint8_t* page_pool;              // Buffer posix_memalign alineado a 4096 bytes
    int used_odirect;                // 1 si O_DIRECT estuvo activo, 0 en fallback
    uint64_t pread_calls;            // Contador estricto de pread

    // Tabla RoPE precomputada [max_seq_len, head_dim / 2] (160 * 24)
    float rope_cos[160 * 24];
    float rope_sin[160 * 24];

    // Tensores residentes FP32 (cobertura 100% dinámica desde JSON)
    float* token_emb;                // [vocab_size, d_model]
    float* abacus_emb;               // [16, d_model]
    float* stage_emb;                // [4, d_model]
    float* step_proj;                // [d_model, d_model]
    float* noise_mlp_w0;             // [d_model, 2]
    float* noise_mlp_b0;             // [d_model]
    float* noise_mlp_w2;             // [d_model, d_model]
    float* noise_mlp_b2;             // [d_model]
    float* norm_z;                   // [d_model]
    float* norm_y;                   // [d_model]
    float* norm_moe;                 // [d_model]
    float* norm_refine;              // [d_model]
    float* attn_q_norm;              // [head_dim]
    float* attn_k_norm;              // [head_dim]
    float* router_w;                 // [num_experts, d_model]
    float* halt_gate_w;              // [1, d_model]
    float  halt_gate_b;
    float  alpha_z;
    float  beta_y;
    float  carry_gate;               // Parámetro residente mapeado
    float* final_norm;               // [d_model]
    float* lm_head_w;                // [vocab_size, d_model]

    // 16 Capas Ternarias BitLinear (0 pesos FP32 en RAM)
    TracePagedLinear anchor_proj;
    TracePagedLinear attn_q;
    TracePagedLinear attn_k;
    TracePagedLinear attn_v;
    TracePagedLinear attn_o;
    TracePagedLinear exp_up[4];
    TracePagedLinear exp_down[4];
    TracePagedLinear shared_up;
    TracePagedLinear shared_down;
    TracePagedLinear answer_updater;

    // Workspace reutilizable
    TraceWorkspace workspace;
} TraceModel;

// Inspección de ABI y memoria para Python ctypes
size_t trace_model_sizeof(void);

// Ciclo de vida
int trace_model_load_odirect(const char* bin_path, TraceModel* model);
void trace_model_free(TraceModel* model);

// Perfilado temporal estricto (CLOCK_MONOTONIC)
void trace_model_reset_profile(void);
void trace_model_print_profile(void);
void trace_model_print_profile_report(void);

// Getters de auditoría O_DIRECT
int trace_model_get_used_odirect(const TraceModel* model);
uint64_t trace_model_get_pread_calls(const TraceModel* model);
size_t trace_model_get_pages_offset(const TraceModel* model);

// Inferencia
void trace_forward_backbone(
    TraceModel* model,
    const int32_t* token_ids,
    int seq_len,
    int stage_id,
    int prompt_len,                  // <= 0: block-causal estándar; > 0: target-unified
    float* out_logits                // [seq_len, vocab_size]
);

void trace_forward_backbone_lazy(
    TraceModel* model,
    const int32_t* token_ids,
    int seq_len,
    int stage_id,
    int prompt_len,
    const int* target_indices,
    int num_targets,
    float* out_logits
);

int trace_generate_causal_two_step(
    TraceModel* model,
    const uint8_t* prompt_bytes,
    int prompt_len,
    char* out_str,
    int max_out_len
);

#ifdef __cplusplus
}
#endif

#endif // TRACE_INFER_H

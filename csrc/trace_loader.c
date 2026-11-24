/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#define _GNU_SOURCE
#include "trace_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>

typedef struct {
    const char* key;
    size_t offset_in_model;
} ResidentTensorDesc;

typedef struct {
    const char* base_name;
    size_t offset_in_model;
} PagedLayerDesc;

static const ResidentTensorDesc g_resident_tensors[] = {
    {"token_emb.weight", offsetof(TraceModel, token_emb)},
    {"abacus_emb.embed.weight", offsetof(TraceModel, abacus_emb)},
    {"stage_emb.weight", offsetof(TraceModel, stage_emb)},
    {"resident_core.step_encoder.proj.weight", offsetof(TraceModel, step_proj)},
    {"resident_core.noise_mlp.0.weight", offsetof(TraceModel, noise_mlp_w0)},
    {"resident_core.noise_mlp.0.bias", offsetof(TraceModel, noise_mlp_b0)},
    {"resident_core.noise_mlp.2.weight", offsetof(TraceModel, noise_mlp_w2)},
    {"resident_core.noise_mlp.2.bias", offsetof(TraceModel, noise_mlp_b2)},
    {"resident_core.norm_z.weight", offsetof(TraceModel, norm_z)},
    {"resident_core.norm_y.weight", offsetof(TraceModel, norm_y)},
    {"resident_core.norm_moe.weight", offsetof(TraceModel, norm_moe)},
    {"resident_core.norm_refine.weight", offsetof(TraceModel, norm_refine)},
    {"resident_core.attn.q_norm.weight", offsetof(TraceModel, attn_q_norm)},
    {"resident_core.attn.k_norm.weight", offsetof(TraceModel, attn_k_norm)},
    {"resident_core.paged_moe.router.weight", offsetof(TraceModel, router_w)},
    {"resident_core.halt_gate.weight", offsetof(TraceModel, halt_gate_w)},
    {"final_norm.weight", offsetof(TraceModel, final_norm)},
    {"lm_head.weight", offsetof(TraceModel, lm_head_w)},
    {NULL, 0}
};

static const PagedLayerDesc g_paged_layers[] = {
    {"resident_core.anchor_proj", offsetof(TraceModel, anchor_proj)},
    {"resident_core.attn.q_proj", offsetof(TraceModel, attn_q)},
    {"resident_core.attn.k_proj", offsetof(TraceModel, attn_k)},
    {"resident_core.attn.v_proj", offsetof(TraceModel, attn_v)},
    {"resident_core.attn.o_proj", offsetof(TraceModel, attn_o)},
    {"resident_core.paged_moe.expert_bank.0.up", offsetof(TraceModel, exp_up[0])},
    {"resident_core.paged_moe.expert_bank.0.down", offsetof(TraceModel, exp_down[0])},
    {"resident_core.paged_moe.expert_bank.1.up", offsetof(TraceModel, exp_up[1])},
    {"resident_core.paged_moe.expert_bank.1.down", offsetof(TraceModel, exp_down[1])},
    {"resident_core.paged_moe.expert_bank.2.up", offsetof(TraceModel, exp_up[2])},
    {"resident_core.paged_moe.expert_bank.2.down", offsetof(TraceModel, exp_down[2])},
    {"resident_core.paged_moe.expert_bank.3.up", offsetof(TraceModel, exp_up[3])},
    {"resident_core.paged_moe.expert_bank.3.down", offsetof(TraceModel, exp_down[3])},
    {"resident_core.paged_moe.shared_expert.up", offsetof(TraceModel, shared_up)},
    {"resident_core.paged_moe.shared_expert.down", offsetof(TraceModel, shared_down)},
    {"resident_core.answer_updater", offsetof(TraceModel, answer_updater)},
    {NULL, 0}
};

static void free_workspace(TraceWorkspace* ws);

static int init_workspace(TraceWorkspace* ws, int max_seq_len, int d_model, int d_ff_expert) {
    int max_dim = (d_ff_expert > d_model) ? d_ff_expert : d_model;
    size_t seq = (size_t)max_seq_len;
    size_t dm = (size_t)d_model;
    size_t dff = (size_t)d_ff_expert;
    size_t md = (size_t)max_dim;
    int err = 0;
    #define ALIGN_ALLOC(ptr, sz) do { \
        if (posix_memalign((void**)&(ptr), 64, (sz)) != 0) err = -1; \
    } while(0)

    ALIGN_ALLOC(ws->x_norm, seq * md * sizeof(float));
    ALIGN_ALLOC(ws->q_x, seq * md * sizeof(int8_t));
    ALIGN_ALLOC(ws->acc_int32, seq * md * sizeof(int32_t));
    ALIGN_ALLOC(ws->scale_x, seq * sizeof(float));

    ALIGN_ALLOC(ws->a, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->z_r, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->y_r, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->anchor_feat, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->joint_state, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->norm_z_buf, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->norm_y_buf, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->q, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->k, seq * (dm / 2u) * sizeof(float));
    ALIGN_ALLOC(ws->v, seq * (dm / 2u) * sizeof(float));
    ALIGN_ALLOC(ws->attn_out_heads, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->attn_final, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->z_mid, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->z_norm_moe, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->moe_out, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->shared_up_buf, seq * dff * sizeof(float));
    ALIGN_ALLOC(ws->shared_down_buf, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->fusion, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->delta_y, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->z_next, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->normed_y, seq * dm * sizeof(float));

    ALIGN_ALLOC(ws->exp_tokens_in, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->exp_up_buf, seq * dff * sizeof(float));
    ALIGN_ALLOC(ws->exp_down_buf, seq * dm * sizeof(float));
    ALIGN_ALLOC(ws->exp_tok_idx, seq * sizeof(int));
    ALIGN_ALLOC(ws->exp_tok_gate, seq * sizeof(float));
    #undef ALIGN_ALLOC
    if (err != 0) {
        free_workspace(ws);
    }
    return err;
}

static void free_workspace(TraceWorkspace* ws) {
    if (!ws) return;
    free(ws->x_norm); free(ws->q_x); free(ws->acc_int32); free(ws->scale_x);
    free(ws->a); free(ws->z_r); free(ws->y_r); free(ws->anchor_feat);
    free(ws->joint_state); free(ws->norm_z_buf); free(ws->norm_y_buf);
    free(ws->q); free(ws->k); free(ws->v);
    free(ws->attn_out_heads); free(ws->attn_final); free(ws->z_mid);
    free(ws->z_norm_moe); free(ws->moe_out); free(ws->shared_up_buf);
    free(ws->shared_down_buf); free(ws->fusion); free(ws->delta_y);
    free(ws->z_next); free(ws->normed_y);
    free(ws->exp_tokens_in); free(ws->exp_up_buf); free(ws->exp_down_buf);
    free(ws->exp_tok_idx); free(ws->exp_tok_gate);
    memset(ws, 0, sizeof(TraceWorkspace));
}

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

static int pread_exact(int fd, void* buf, size_t len, off_t off) {
    uint8_t* p = (uint8_t*)buf;
    size_t total = 0;
    while (total < len) {
        size_t remain = len - total;
        size_t chunk = (remain > 0x7ffff000U) ? 0x7ffff000U : remain;
        ssize_t n = pread(fd, p + total, chunk, off + (off_t)total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        total += (size_t)n;
    }
    return 0;
}

int trace_model_load_odirect(const char* bin_path, TraceModel* model) {
    memset(model, 0, sizeof(TraceModel));
    model->fd = -1; // Inicialización estricta tras memset

    int open_flags = O_RDONLY;
#ifdef O_DIRECT
    open_flags |= O_DIRECT;
#endif
    int fd = open(bin_path, open_flags);
    int used_odirect = 0;
#ifdef O_DIRECT
    if (fd >= 0) used_odirect = 1;
#endif
#ifdef F_NOCACHE
    if (fd >= 0) {
        fcntl(fd, F_NOCACHE, 1);
        used_odirect = 1;
    }
#endif
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
        model->fd = -1;
        return -2;
    }

    ssize_t r;
    do {
        r = pread(fd, scratch, 4096, 0);
    } while (r < 0 && errno == EINTR);
    if (r < 0 && errno == EINVAL && used_odirect) {
        close(fd);
        model->fd = -1;
        fd = open(bin_path, O_RDONLY);
        if (fd < 0) {
            free(scratch);
            return -1;
        }
        used_odirect = 0;
        model->fd = fd;
        model->used_odirect = 0;
        do {
            r = pread(fd, scratch, 4096, 0);
        } while (r < 0 && errno == EINTR);
    }
    model->pread_calls++;
    if (r < 64 || memcmp(scratch, "PBITCOS1", 8) != 0) {
        free(scratch);
        close(fd);
        model->fd = -1;
        return -4;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 64) {
        free(scratch);
        close(fd);
        model->fd = -1;
        return -4;
    }
    uint64_t file_size = (uint64_t)st.st_size;

    const uint8_t* p = (const uint8_t*)scratch;
    uint32_t page_sz, fmt_code, total_pages;
    uint64_t idx_off, exp_off, res_off, pages_off, total_sz;
    memcpy(&page_sz, p + 8, 4);
    memcpy(&fmt_code, p + 12, 4);
    memcpy(&total_pages, p + 16, 4);
    memcpy(&idx_off, p + 20, 8);
    memcpy(&exp_off, p + 28, 8);
    memcpy(&res_off, p + 36, 8);
    memcpy(&pages_off, p + 44, 8);
    memcpy(&total_sz, p + 52, 8);
    free(scratch);

    if (fmt_code != 1 ||
        page_sz < sizeof(PagedHeader) ||
        idx_off > exp_off || exp_off > file_size ||
        res_off > pages_off || pages_off > total_sz || total_sz > file_size) {
        close(fd);
        model->fd = -1;
        return -4;
    }

    uint64_t json_len_u64 = exp_off - idx_off;
    uint64_t pool_size_u64 = (uint64_t)total_pages * (uint64_t)page_sz;
    if (json_len_u64 == 0 || json_len_u64 > (uint64_t)(64u * 1024u * 1024u) ||
        total_pages == 0 ||
        (uint64_t)total_pages > (total_sz - pages_off) / (uint64_t)page_sz) {
        close(fd);
        model->fd = -1;
        return -4;
    }

    model->num_pages = total_pages;
    model->page_size = page_sz;
    model->pages_offset = pages_off;

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

    // Precomputación estática de RoPE
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

    size_t pool_size = (size_t)pool_size_u64;
    if (posix_memalign((void**)&model->page_pool, 4096, pool_size) != 0) {
        trace_model_free(model);
        return -5;
    }

    // Lectura contigua O_DIRECT para todo el page_pool con reintento EINTR
    size_t pool_read = 0;
    while (pool_read < pool_size) {
        size_t remain = pool_size - pool_read;
        size_t chunk = (remain > 0x7ffff000U) ? 0x7ffff000U : remain;
        ssize_t n_read = pread(fd, model->page_pool + pool_read, chunk, (off_t)(pages_off + pool_read));
        if (n_read < 0) {
            if (errno == EINTR) continue;
            trace_model_free(model);
            return -6;
        }
        if (n_read == 0) {
            trace_model_free(model);
            return -6;
        }
        model->pread_calls++;
        pool_read += (size_t)n_read;
    }

    close(fd);
    model->fd = -1;

    int fd_meta = open(bin_path, O_RDONLY);
    if (fd_meta < 0) {
        trace_model_free(model);
        return -7;
    }

    size_t json_len = (size_t)json_len_u64;
    char* json_buf = (char*)malloc(json_len + 1);
    if (!json_buf) {
        close(fd_meta);
        trace_model_free(model);
        return -7;
    }
    if (pread_exact(fd_meta, json_buf, json_len, (off_t)idx_off) != 0) {
        free(json_buf);
        close(fd_meta);
        trace_model_free(model);
        return -8;
    }
    json_buf[json_len] = '\0';

    int mapped_count = 0;
    const char* curr = json_buf;
    const char* json_end = json_buf + json_len;

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
            for (int i = 0; g_resident_tensors[i].key != NULL; ++i) {
                if (strcmp(name, g_resident_tensors[i].key) == 0) {
                    dest_ptr = (float**)((char*)model + g_resident_tensors[i].offset_in_model);
                    break;
                }
            }

            if (!dest_ptr) {
                for (int i = 0; g_paged_layers[i].base_name != NULL; ++i) {
                    char sub_key[160];
                    snprintf(sub_key, sizeof(sub_key), "%s.sub_ln.weight", g_paged_layers[i].base_name);
                    if (strcmp(name, sub_key) == 0) {
                        TracePagedLinear* layer = (TracePagedLinear*)((char*)model + g_paged_layers[i].offset_in_model);
                        dest_ptr = &layer->sub_ln_w;
                        break;
                    }
                }
            }

            // Suma estricta de res_off en todas las lecturas de tensores y escalares
            if (dest_ptr) {
                *dest_ptr = (float*)malloc(t_length);
                if (*dest_ptr && pread_exact(fd_meta, *dest_ptr, t_length, (off_t)(res_off + t_offset)) == 0) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.alpha_z") == 0) {
                if (pread_exact(fd_meta, &model->alpha_z, sizeof(float), (off_t)(res_off + t_offset)) == 0) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.beta_y") == 0) {
                if (pread_exact(fd_meta, &model->beta_y, sizeof(float), (off_t)(res_off + t_offset)) == 0) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.carry_gate") == 0) {
                if (pread_exact(fd_meta, &model->carry_gate, sizeof(float), (off_t)(res_off + t_offset)) == 0) {
                    mapped_count++;
                }
            } else if (strcmp(name, "resident_core.halt_gate.bias") == 0) {
                if (pread_exact(fd_meta, &model->halt_gate_b, sizeof(float), (off_t)(res_off + t_offset)) == 0) {
                    mapped_count++;
                }
            }
        } else {
            TracePagedLinear* target_layer = NULL;
            for (int i = 0; g_paged_layers[i].base_name != NULL; ++i) {
                if (strcmp(name, g_paged_layers[i].base_name) == 0) {
                    target_layer = (TracePagedLinear*)((char*)model + g_paged_layers[i].offset_in_model);
                    break;
                }
            }

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
                    if (pids_ptr < obj_end) pids_ptr++;
                    int pids[32];
                    int n_pids = 0;
                    int pid_err = 0;
                    while (pids_ptr < obj_end && *pids_ptr != ']' && n_pids < 32) {
                        while (pids_ptr < obj_end && (*pids_ptr == ' ' || *pids_ptr == ',')) pids_ptr++;
                        if (pids_ptr >= obj_end || *pids_ptr == ']') break;
                        char* endptr = NULL;
                        long pid = strtol(pids_ptr, &endptr, 10);
                        if (endptr == pids_ptr) {
                            pid_err = 1;
                            break;
                        }
                        pids_ptr = endptr;
                        if (pid < 0 || (size_t)pid >= (size_t)total_pages) {
                            pid_err = 1;
                            break;
                        }
                        pids[n_pids++] = (int)pid;
                    }
                    if (pid_err || n_pids <= 0) {
                        free(json_buf);
                        close(fd_meta);
                        trace_model_free(model);
                        return -8;
                    }
                    uint8_t** ptrs = (uint8_t**)malloc((size_t)n_pids * sizeof(uint8_t*));
                    if (!ptrs) {
                        free(json_buf);
                        close(fd_meta);
                        trace_model_free(model);
                        return -8;
                    }
                    for (int pi = 0; pi < n_pids; ++pi) {
                        ptrs[pi] = model->page_pool + (size_t)pids[pi] * (size_t)page_sz;
                    }
                    target_layer->num_pages = n_pids;
                    target_layer->page_ptrs = (const uint8_t* const*)ptrs;
                    const PagedHeader* ph0 = (const PagedHeader*)ptrs[0];
                    target_layer->scale = ph0->scale_w;
                    mapped_count++;
                }
            }
        }
        curr = obj_end;
    }

    free(json_buf);
    close(fd_meta);

    // Validación contractual exacta de los 54 componentes mapeados
    if (mapped_count != 54) {
        trace_model_free(model);
        return -8;
    }

    // Precomputación en carga del vector constante e_t (noise_mlp con noise_in={1.0f, 0.25f})
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
        model->e_t_const[i] = acc;
    }

    // Precomputación en carga de la tabla e_r_table[0..R-1]
    int R = model->cfg.num_recursions;
    for (int r = 0; r < R && r < 8; ++r) {
        float u_norm = (float)(r + 1) / (float)R;
        float angle_cat[192];
        for (int i = 0; i < 96; ++i) {
            float freq = expf((float)i * (-logf(100.0f) / 95.0f));
            float ang = u_norm * 3.141592653589793f * freq;
            angle_cat[i] = sinf(ang);
            angle_cat[i + 96] = cosf(ang);
        }
        for (int i = 0; i < model->cfg.d_model; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < 192; ++k) acc += model->step_proj[i * 192 + k] * angle_cat[k];
            model->e_r_table[r][i] = acc;
        }
    }

    return 0;
}

void trace_model_free(TraceModel* model) {
    if (!model) return;
    free_workspace(&model->workspace);

    if (model->page_pool) {
        free(model->page_pool);
        model->page_pool = NULL;
    }
    if (model->fd >= 0) {
        close(model->fd);
        model->fd = -1;
    }

    for (int i = 0; g_resident_tensors[i].key != NULL; ++i) {
        float** ptr = (float**)((char*)model + g_resident_tensors[i].offset_in_model);
        if (*ptr) {
            free(*ptr);
            *ptr = NULL;
        }
    }

    for (int i = 0; g_paged_layers[i].base_name != NULL; ++i) {
        TracePagedLinear* layer = (TracePagedLinear*)((char*)model + g_paged_layers[i].offset_in_model);
        if (layer->sub_ln_w) {
            free(layer->sub_ln_w);
            layer->sub_ln_w = NULL;
        }
        if (layer->page_ptrs) {
            free((void*)layer->page_ptrs);
            layer->page_ptrs = NULL;
        }
    }
}

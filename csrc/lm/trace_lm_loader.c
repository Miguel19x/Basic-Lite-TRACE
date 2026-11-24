/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#define _GNU_SOURCE
#include "trace_lm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

static const char* scan_json_key(const char* start, const char* end, const char* key) {
    size_t klen = strlen(key);
    const char* p = start;
    while (p + klen + 3 <= end) {
        if (*p == '"' && memcmp(p + 1, key, klen) == 0 && p[klen + 1] == '"') {
            const char* val = p + klen + 2;
            while (val < end && (*val == ' ' || *val == ':' || *val == '\t' || *val == '\n' || *val == '\r')) {
                val++;
            }
            return val;
        }
        p++;
    }
    return NULL;
}

static int scan_json_int(const char* start, const char* end, const char* key, int default_val) {
    const char* p = scan_json_key(start, end, key);
    if (!p) return default_val;
    return atoi(p);
}

static int scan_json_bool(const char* start, const char* end, const char* key, int default_val) {
    const char* p = scan_json_key(start, end, key);
    if (!p) return default_val;
    if (strncmp(p, "true", 4) == 0 || strncmp(p, "True", 4) == 0 || *p == '1') return 1;
    if (strncmp(p, "false", 5) == 0 || strncmp(p, "False", 5) == 0 || *p == '0') return 0;
    return default_val;
}

static float scan_json_float(const char* start, const char* end, const char* key, float default_val) {
    const char* p = scan_json_key(start, end, key);
    if (!p) return default_val;
    return (float)atof(p);
}

static int scan_json_int_array(const char* start, const char* end, const char* key, int* out_arr, int max_items) {
    const char* p = scan_json_key(start, end, key);
    if (!p || *p != '[') return 0;
    p++;
    int count = 0;
    while (p < end && *p != ']' && count < max_items) {
        while (p < end && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        if (*p == ']') break;
        out_arr[count++] = atoi(p);
        while (p < end && *p != ',' && *p != ']') p++;
        if (*p == ',') p++;
    }
    return count;
}

static int pread_exact(int fd, void* buf, size_t len, off_t off) {
    uint8_t* p = (uint8_t*)buf;
    size_t total = 0;
    while (total < len) {
        ssize_t n = pread(fd, p + total, len - total, off + (off_t)total);
        if (n <= 0) {
            return -1;
        }
        total += (size_t)n;
    }
    return 0;
}

TraceLMModel* trace_lm_load(const char* bin_path) {
    if (!bin_path) return NULL;

    TraceLMModel* model = (TraceLMModel*)calloc(1, sizeof(TraceLMModel));
    if (!model) return NULL;
    model->fd = -1;

    // 1. Intentar O_DIRECT portable (o F_NOCACHE en Darwin)
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
    if (fd < 0) {
        free(model);
        return NULL;
    }

    model->fd = fd;
    model->used_odirect = used_odirect;
    model->pread_calls = 0;

    // 2. Leer cabecera de 64 bytes alineada a 4096
    void* scratch = NULL;
    if (posix_memalign(&scratch, 4096, 4096) != 0) {
        trace_lm_free(model);
        return NULL;
    }

    ssize_t r = pread(fd, scratch, 4096, 0);
    if (r < 0 && errno == EINVAL && used_odirect) {
        close(fd);
        fd = open(bin_path, O_RDONLY);
        if (fd < 0) {
            free(scratch);
            trace_lm_free(model);
            return NULL;
        }
        used_odirect = 0;
        if (model) {
            model->fd = fd;
            model->used_odirect = 0;
        }
        r = pread(fd, scratch, 4096, 0);
    }
    model->pread_calls++;
    if (r < 64 || memcmp(scratch, "PBITCOS1", 8) != 0) {
        free(scratch);
        trace_lm_free(model);
        return NULL;
    }

    const uint8_t* hdr = (const uint8_t*)scratch;
    uint32_t page_sz, total_pages;
    uint64_t idx_off, exp_off, res_off, pages_off, total_sz;
    memcpy(&page_sz, hdr + 8, 4);
    memcpy(&total_pages, hdr + 16, 4);
    memcpy(&idx_off, hdr + 20, 8);
    memcpy(&exp_off, hdr + 28, 8);
    memcpy(&res_off, hdr + 36, 8);
    memcpy(&pages_off, hdr + 44, 8);
    memcpy(&total_sz, hdr + 52, 8);
    free(scratch);

    // 3. Cargar todo el pool de páginas en 1 única syscall pread alineada
    size_t pool_size = total_sz - pages_off;
    model->page_pool_bytes = pool_size;
    if (posix_memalign(&model->page_pool, 4096, pool_size) != 0) {
        trace_lm_free(model);
        return NULL;
    }

    ssize_t n_read = pread(fd, model->page_pool, pool_size, (off_t)pages_off);
    model->pread_calls++;
    if (n_read != (ssize_t)pool_size) {
        trace_lm_free(model);
        return NULL;
    }

    // 4. Leer metadatos JSON e índice
    int fd_meta = open(bin_path, O_RDONLY);
    if (fd_meta < 0) {
        trace_lm_free(model);
        return NULL;
    }

    size_t json_len = exp_off - idx_off;
    char* json_buf = (char*)malloc(json_len + 1);
    if (!json_buf) {
        close(fd_meta);
        trace_lm_free(model);
        return NULL;
    }

    if (pread_exact(fd_meta, json_buf, json_len, (off_t)idx_off) != 0) {
        free(json_buf);
        close(fd_meta);
        trace_lm_free(model);
        return NULL;
    }
    json_buf[json_len] = '\0';
    const char* json_end = json_buf + json_len;

    // Validar arquitectura
    const char* arch_p = scan_json_key(json_buf, json_end, "architecture");
    if (!arch_p || strncmp(arch_p, "\"trace-lm-v1\"", 13) != 0) {
        free(json_buf);
        close(fd_meta);
        trace_lm_free(model);
        return NULL;
    }

    // Parsear configuracion
    const char* cfg_p = scan_json_key(json_buf, json_end, "config");
    if (!cfg_p || *cfg_p != '{') {
        free(json_buf);
        close(fd_meta);
        trace_lm_free(model);
        return NULL;
    }
    const char* cfg_end = cfg_p;
    while (cfg_end < json_end && *cfg_end != '}') cfg_end++;

    model->cfg.vocab_size = scan_json_int(cfg_p, cfg_end, "vocab_size", 4096);
    model->cfg.d_model = scan_json_int(cfg_p, cfg_end, "d_model", 512);
    model->cfg.n_layers = scan_json_int(cfg_p, cfg_end, "n_layers", 12);
    model->cfg.n_heads = scan_json_int(cfg_p, cfg_end, "n_heads", 8);
    model->cfg.d_ff = scan_json_int(cfg_p, cfg_end, "d_ff", 1536);
    model->cfg.max_seq_len = scan_json_int(cfg_p, cfg_end, "max_seq_len", 4096);
    model->cfg.eps = scan_json_float(cfg_p, cfg_end, "eps", 1e-6f);
    model->cfg.kv_lora_rank = scan_json_int(cfg_p, cfg_end, "kv_lora_rank", 128);
    model->cfg.qk_nope_head_dim = scan_json_int(cfg_p, cfg_end, "qk_nope_head_dim", 64);
    model->cfg.qk_rope_head_dim = scan_json_int(cfg_p, cfg_end, "qk_rope_head_dim", 32);
    model->cfg.v_head_dim = scan_json_int(cfg_p, cfg_end, "v_head_dim", 64);
    model->cfg.tie_word_embeddings = scan_json_bool(cfg_p, cfg_end, "tie_word_embeddings", 1);
    model->cfg.pad_token_id = scan_json_int(cfg_p, cfg_end, "pad_token_id", 256);
    model->cfg.mask_token_id = scan_json_int(cfg_p, cfg_end, "mask_token_id", 257);
    model->cfg.bos_token_id = scan_json_int(cfg_p, cfg_end, "bos_token_id", 258);
    model->cfg.eos_token_id = scan_json_int(cfg_p, cfg_end, "eos_token_id", 259);

    // Alojar capas
    model->layers = (TraceLMLayer*)calloc(model->cfg.n_layers, sizeof(TraceLMLayer));
    if (!model->layers) {
        free(json_buf);
        close(fd_meta);
        trace_lm_free(model);
        return NULL;
    }

    // Precomputar tablas RoPE intercaladas
    size_t rope_sz = (size_t)model->cfg.max_seq_len * (size_t)model->cfg.qk_rope_head_dim;
    model->rope_cos = (float*)malloc(sizeof(float) * rope_sz);
    model->rope_sin = (float*)malloc(sizeof(float) * rope_sz);
    if (!model->rope_cos || !model->rope_sin) {
        free(json_buf);
        close(fd_meta);
        trace_lm_free(model);
        return NULL;
    }

    for (int p = 0; p < model->cfg.max_seq_len; ++p) {
        for (int j = 0; j < model->cfg.qk_rope_head_dim; j += 2) {
            float freq = 1.0f / powf(10000.0f, (float)j / (float)model->cfg.qk_rope_head_dim);
            float angle = (float)p * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            model->rope_cos[p * model->cfg.qk_rope_head_dim + j] = c;
            model->rope_cos[p * model->cfg.qk_rope_head_dim + j + 1] = c;
            model->rope_sin[p * model->cfg.qk_rope_head_dim + j] = s;
            model->rope_sin[p * model->cfg.qk_rope_head_dim + j + 1] = s;
        }
    }

    // 5. Iterar sobre tensores del indice
    const char* curr = json_buf;
    while ((curr = scan_json_key(curr, json_end, "name")) != NULL) {
        const char* name_start = curr;
        while (name_start < json_end && *name_start != '"') name_start++;
        name_start++;
        const char* name_end = name_start;
        while (name_end < json_end && *name_end != '"') name_end++;

        char name[128];
        size_t nlen = name_end - name_start;
        if (nlen >= sizeof(name)) nlen = sizeof(name) - 1;
        memcpy(name, name_start, nlen);
        name[nlen] = '\0';

        const char* obj_end = name_end;
        while (obj_end < json_end && *obj_end != '}') obj_end++;

        const char* res_ptr = scan_json_key(name_end, obj_end, "is_resident");
        int is_resident = (res_ptr && strncmp(res_ptr, "true", 4) == 0);

        if (is_resident) {
            const char* off_ptr = scan_json_key(name_end, obj_end, "offset");
            const char* len_ptr = scan_json_key(name_end, obj_end, "length");
            uint64_t t_off = off_ptr ? strtoull(off_ptr, NULL, 10) : 0;
            uint64_t t_len = len_ptr ? strtoull(len_ptr, NULL, 10) : 0;

            if (strcmp(name, "tok_emb.weight") == 0) {
                model->tok_emb = (float*)malloc(t_len);
                if (model->tok_emb) {
                    if (pread_exact(fd_meta, model->tok_emb, t_len, (off_t)(res_off + t_off)) != 0) {
                        free(model->tok_emb);
                        model->tok_emb = NULL;
                    }
                }
            } else if (strcmp(name, "final_norm.weight") == 0) {
                model->final_norm_w = (float*)malloc(t_len);
                if (model->final_norm_w) {
                    if (pread_exact(fd_meta, model->final_norm_w, t_len, (off_t)(res_off + t_off)) != 0) {
                        free(model->final_norm_w);
                        model->final_norm_w = NULL;
                    }
                }
            } else if (strcmp(name, "lm_head.weight") == 0) {
                if (!model->cfg.tie_word_embeddings) {
                    model->lm_head_w = (float*)malloc(t_len);
                    if (model->lm_head_w) {
                        if (pread_exact(fd_meta, model->lm_head_w, t_len, (off_t)(res_off + t_off)) != 0) {
                            free(model->lm_head_w);
                            model->lm_head_w = NULL;
                        }
                    }
                }
            } else if (strncmp(name, "layers.", 7) == 0) {
                int layer_idx = atoi(name + 7);
                const char* dot = strchr(name + 7, '.');
                if (dot && layer_idx >= 0 && layer_idx < model->cfg.n_layers) {
                    const char* field = dot + 1;
                    if (strcmp(field, "attn_norm.weight") == 0) {
                        model->layers[layer_idx].attn_norm_w = (float*)malloc(t_len);
                        if (model->layers[layer_idx].attn_norm_w) {
                            if (pread_exact(fd_meta, model->layers[layer_idx].attn_norm_w, t_len, (off_t)(res_off + t_off)) != 0) {
                                free(model->layers[layer_idx].attn_norm_w);
                                model->layers[layer_idx].attn_norm_w = NULL;
                            }
                        }
                    } else if (strcmp(field, "ffn_norm.weight") == 0) {
                        model->layers[layer_idx].ffn_norm_w = (float*)malloc(t_len);
                        if (model->layers[layer_idx].ffn_norm_w) {
                            if (pread_exact(fd_meta, model->layers[layer_idx].ffn_norm_w, t_len, (off_t)(res_off + t_off)) != 0) {
                                free(model->layers[layer_idx].ffn_norm_w);
                                model->layers[layer_idx].ffn_norm_w = NULL;
                            }
                        }
                    } else if (strcmp(field, "attn.kv_norm.weight") == 0) {
                        model->layers[layer_idx].kv_norm_w = (float*)malloc(t_len);
                        if (model->layers[layer_idx].kv_norm_w) {
                            if (pread_exact(fd_meta, model->layers[layer_idx].kv_norm_w, t_len, (off_t)(res_off + t_off)) != 0) {
                                free(model->layers[layer_idx].kv_norm_w);
                                model->layers[layer_idx].kv_norm_w = NULL;
                            }
                        }
                    } else if (strstr(field, "sub_ln.weight") != NULL) {
                        TraceLMPagedLinear* target_lin = NULL;
                        if (strcmp(field, "attn.w_dkv.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].w_dkv;
                        else if (strcmp(field, "attn.w_kr.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].w_kr;
                        else if (strcmp(field, "attn.w_q.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].w_q;
                        else if (strcmp(field, "attn.w_qr.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].w_qr;
                        else if (strcmp(field, "attn.w_out.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].w_out;
                        else if (strcmp(field, "ffn.up.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].ffn_up;
                        else if (strcmp(field, "ffn.down.sub_ln.weight") == 0) target_lin = &model->layers[layer_idx].ffn_down;

                        if (target_lin) {
                            target_lin->sub_ln_w = (float*)malloc(t_len);
                            if (target_lin->sub_ln_w) {
                                if (pread_exact(fd_meta, target_lin->sub_ln_w, t_len, (off_t)(res_off + t_off)) != 0) {
                                    free(target_lin->sub_ln_w);
                                    target_lin->sub_ln_w = NULL;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Capa paginada
            if (strncmp(name, "layers.", 7) == 0) {
                int layer_idx = atoi(name + 7);
                const char* dot = strchr(name + 7, '.');
                if (dot && layer_idx >= 0 && layer_idx < model->cfg.n_layers) {
                    const char* field = dot + 1;
                    TraceLMPagedLinear* lin = NULL;
                    if (strcmp(field, "attn.w_dkv") == 0) lin = &model->layers[layer_idx].w_dkv;
                    else if (strcmp(field, "attn.w_uk") == 0) lin = &model->layers[layer_idx].w_uk;
                    else if (strcmp(field, "attn.w_uv") == 0) lin = &model->layers[layer_idx].w_uv;
                    else if (strcmp(field, "attn.w_kr") == 0) lin = &model->layers[layer_idx].w_kr;
                    else if (strcmp(field, "attn.w_q") == 0) lin = &model->layers[layer_idx].w_q;
                    else if (strcmp(field, "attn.w_qr") == 0) lin = &model->layers[layer_idx].w_qr;
                    else if (strcmp(field, "attn.w_out") == 0) lin = &model->layers[layer_idx].w_out;
                    else if (strcmp(field, "ffn.up") == 0) lin = &model->layers[layer_idx].ffn_up;
                    else if (strcmp(field, "ffn.down") == 0) lin = &model->layers[layer_idx].ffn_down;

                    if (lin) {
                        int shapes[4];
                        int n_dims = scan_json_int_array(name_end, obj_end, "shape", shapes, 4);
                        if (n_dims >= 2) {
                            lin->out_features = shapes[0];
                            lin->in_features = shapes[1];
                        }
                        lin->scale = scan_json_float(name_end, obj_end, "scale", 1.0f);
                        int num_p = scan_json_int(name_end, obj_end, "num_pages", 0);
                        lin->num_pages = num_p;

                        if (num_p > 0) {
                            int* pids = (int*)malloc(sizeof(int) * num_p);
                            int scanned_p = scan_json_int_array(name_end, obj_end, "page_ids", pids, num_p);
                            const uint8_t** ptrs = (const uint8_t**)malloc(sizeof(uint8_t*) * scanned_p);
                            for (int k = 0; k < scanned_p; ++k) {
                                ptrs[k] = (const uint8_t*)model->page_pool + ((size_t)pids[k] * page_sz);
                            }
                            lin->page_ptrs = ptrs;
                            free(pids);
                        }
                    }
                }
            }
        }
        curr = obj_end;
    }

    // Tied embeddings: si esta activo o si lm_head_w no se serializo por separado, enlazar con tok_emb
    if (model->cfg.tie_word_embeddings || model->lm_head_w == NULL) {
        model->lm_head_w = model->tok_emb;
    }

    free(json_buf);
    close(fd_meta);
    return model;
}

void trace_lm_free(TraceLMModel* model) {
    if (!model) return;

    if (model->tok_emb) free(model->tok_emb);
    // Solo liberar lm_head_w si no esta ligado a tok_emb (evitar double free)
    if (!model->cfg.tie_word_embeddings && model->lm_head_w && model->lm_head_w != model->tok_emb) {
        free(model->lm_head_w);
    }
    model->lm_head_w = NULL;

    if (model->final_norm_w) free(model->final_norm_w);

    if (model->layers) {
        for (int i = 0; i < model->cfg.n_layers; ++i) {
            TraceLMLayer* l = &model->layers[i];
            if (l->attn_norm_w) free(l->attn_norm_w);
            if (l->ffn_norm_w) free(l->ffn_norm_w);
            if (l->kv_norm_w) free(l->kv_norm_w);

            TraceLMPagedLinear* lins[9] = {
                &l->w_dkv, &l->w_uk, &l->w_uv, &l->w_kr,
                &l->w_q, &l->w_qr, &l->w_out, &l->ffn_up, &l->ffn_down
            };
            for (int k = 0; k < 9; ++k) {
                if (lins[k]->page_ptrs) free((void*)lins[k]->page_ptrs);
                if (lins[k]->sub_ln_w) free(lins[k]->sub_ln_w);
            }
        }
        free(model->layers);
    }

    if (model->rope_cos) free(model->rope_cos);
    if (model->rope_sin) free(model->rope_sin);

    if (model->page_pool) free(model->page_pool);
    if (model->fd >= 0) close(model->fd);

    free(model);
}

int trace_lm_get_pread_calls(const TraceLMModel* model) {
    return model ? model->pread_calls : 0;
}

int trace_lm_get_used_odirect(const TraceLMModel* model) {
    return model ? model->used_odirect : 0;
}

/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "trace_lm.h"
#include "trace_lm_kv.h"
#include "trace_tokenizer.h"
#include "paged_dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    const char* model_path = "checkpoints/trace_lm_trained_paged_bitcos.bin";
    const char* tokenizer_path = "checkpoints/trace_tokenizer_8k_bi.bin";
    const char* prompt = "Basic-Lite";
    const char* system_prompt = NULL;
    int use_chat = 0;
    int max_tokens = 32;
    float temperature = 0.0f;
    int top_k = 50;
    float top_p = -1.0f;
    float rep_pen = 1.15f;
    int use_int8_kv = 1;
    uint64_t seed = 0;
    int seed_provided = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) system_prompt = argv[++i];
        else if (strcmp(argv[i], "--chat") == 0) use_chat = 1;
        else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rep_pen") == 0 && i + 1 < argc) rep_pen = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--int8_kv") == 0 && i + 1 < argc) use_int8_kv = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no_int8_kv") == 0 || strcmp(argv[i], "--fp32_kv") == 0) use_int8_kv = 0;
        else if (strcmp(argv[i], "--isa") == 0 && i + 1 < argc) trace_cpu_set_isa(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
            seed_provided = 1;
        }
    }

    if (top_p < 0.0f) {
        top_p = (temperature > 0.0f) ? 0.9f : 1.0f;
    }

    uint64_t rng_state = 0;
    if (seed_provided) {
        rng_state = (seed == 0) ? 0x853c49e6748fea9bULL : seed;
    } else {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        rng_state = ((uint64_t)time(NULL) ^ ((uint64_t)ts.tv_nsec << 16) ^ (uintptr_t)&argc);
        if (rng_state == 0) rng_state = 0x853c49e6748fea9bULL;
        seed = rng_state;
    }

    printf("================================================================================\n");
    printf("Basic-Lite TRACE Language Model CLI — Motor Nativo C Standalone\n");
    printf("ISA CPU activa: %s (Despacho Dinamico Universal)\n", trace_cpu_active_isa());
    printf("Modelo:      %s\n", model_path);
    printf("Tokenizador: %s\n", tokenizer_path);
    printf("Prompt:      \"%s\"\n", prompt);
    printf("Max tokens:  %d | Temp: %.2f | Top-K: %d | Top-P: %.2f | RepPen: %.2f | INT8 KV: %d | Seed: %" PRIu64 "\n",
           max_tokens, temperature, top_k, top_p, rep_pen, use_int8_kv, seed);
    printf("================================================================================\n");

    // 1. Cargar modelo
    TraceLMModel* model = trace_lm_load(model_path);
    if (!model) {
        fprintf(stderr, "Error: no se pudo cargar el modelo desde %s\n", model_path);
        return 1;
    }
    printf("Modelo cargado: %d capas, d_model=%d, vocab=%d (O_DIRECT: %d, pread_calls: %d)\n",
           model->cfg.n_layers, model->cfg.d_model, model->cfg.vocab_size,
           trace_lm_get_used_odirect(model), trace_lm_get_pread_calls(model));

    // 2. Cargar tokenizador si existe y preparar buffer de tokens con longitud max_seq_len
    TraceTokenizer* tok = trace_tok_load(tokenizer_path);
    int max_seq = model->cfg.max_seq_len;
    int* all_tokens = (int*)malloc(sizeof(int) * (size_t)max_seq);
    if (!all_tokens) {
        fprintf(stderr, "Error: no se pudo reservar memoria para historial de tokens.\n");
        trace_lm_free(model);
        trace_tok_free(tok);
        return 1;
    }

    int prompt_len = 0;
    if (tok) {
        if (use_chat) {
            prompt_len = trace_tokenizer_format_chat_llama3(tok, system_prompt, prompt, 1, all_tokens, max_seq);
            printf("Chat LLaMA-3 tokenizado (%d tokens):\n", prompt_len);
        } else {
            prompt_len = trace_tokenizer_encode(tok, (const uint8_t*)prompt, (int)strlen(prompt), all_tokens, max_seq);
            printf("Prompt tokenizado (%d tokens):\n", prompt_len);
        }
    } else {
        if (max_seq < 3) {
            fprintf(stderr, "Error: max_seq_len (%d) insuficiente para tokens sinteticos.\n", max_seq);
            free(all_tokens);
            trace_lm_free(model);
            return 1;
        }
        printf("Aviso: tokenizador no encontrado en %s, usando tokens iniciales sinteticos.\n", tokenizer_path);
        all_tokens[0] = 10;
        all_tokens[1] = 20;
        all_tokens[2] = 30;
        prompt_len = 3;
    }

    if (prompt_len <= 0) {
        fprintf(stderr, "Error: prompt vacio o fallo en codificacion.\n");
        free(all_tokens);
        trace_lm_free(model);
        trace_tok_free(tok);
        return 1;
    }

    // 3. Crear KV Cache y Workspace
    const int prefill_chunk = 512;
    TraceLMKVCache* kv = trace_lm_kv_create(&model->cfg, model->cfg.max_seq_len, use_int8_kv);
    TraceLMWorkspace* ws = trace_lm_workspace_create(&model->cfg, prefill_chunk);
    if (!kv || !ws) {
        fprintf(stderr, "Error al inicializar KV Cache o Workspace.\n");
        trace_lm_workspace_free(ws);
        trace_lm_kv_free(kv);
        free(all_tokens);
        trace_lm_free(model);
        trace_tok_free(tok);
        return 1;
    }

    float* logits = (float*)malloc(sizeof(float) * (size_t)model->cfg.vocab_size);
    if (!logits) {
        fprintf(stderr, "Error al reservar buffer de logits.\n");
        free(all_tokens);
        trace_lm_workspace_free(ws);
        trace_lm_kv_free(kv);
        trace_lm_free(model);
        trace_tok_free(tok);
        return 1;
    }

    // 4. Prefill en bloques <= 512 tokens para soportar prompts largos
    double t0 = get_time_sec();
    int s_proc = 0;
    for (int processed = 0; processed < prompt_len; ) {
        int next_end = processed + prefill_chunk;
        if (next_end > prompt_len) next_end = prompt_len;
        int rc = trace_lm_forward(model, kv, all_tokens, next_end, logits, 1, ws);
        if (rc < 0) {
            fprintf(stderr, "Error en prefill trace_lm_forward (rc=%d).\n", rc);
            free(all_tokens);
            free(logits);
            trace_lm_workspace_free(ws);
            trace_lm_kv_free(kv);
            trace_lm_free(model);
            trace_tok_free(tok);
            return 1;
        }
        s_proc += rc;
        processed = next_end;
    }
    double ttft = get_time_sec() - t0;
    printf("\nPrefill procesado (%d tokens nuevos): TTFT = %.2f ms\n", s_proc, ttft * 1000.0);

    if (rep_pen > 1.0f) {
        trace_lm_apply_repetition_penalty(logits, model->cfg.vocab_size, all_tokens, prompt_len, rep_pen);
    }

    int next_tok = trace_lm_sample(logits, model->cfg.vocab_size, temperature, top_k, top_p, &rng_state);

    // 5. Decodificacion autorregresiva token a token (S = 1 via Prefix Caching)
    printf("\n--- GENERACION ---\n");
    if (tok) {
        printf("%s", prompt);
        fflush(stdout);
    }

    int current_len = prompt_len;
    double t_gen_start = get_time_sec();
    int gen_count = 0;

    for (int step = 0; step < max_tokens; ++step) {
        if (tok) {
            uint8_t piece[128];
            int32_t t_id = (int32_t)next_tok;
            int plen = trace_tokenizer_decode(tok, &t_id, 1, piece, (int)sizeof(piece) - 1);
            if (plen > 0) {
                fwrite(piece, 1, (size_t)plen, stdout);
                fflush(stdout);
            }
        } else {
            printf("[%d] ", next_tok);
            fflush(stdout);
        }
        gen_count++;

        if (next_tok == model->cfg.eos_token_id || (tok && tok->eot_id >= 0 && next_tok == tok->eot_id)) {
            break;
        }

        if (current_len >= model->cfg.max_seq_len) break;

        // Registrar token generado en la secuencia acumulada
        all_tokens[current_len++] = next_tok;

        // Evaluar siguiente token (S = 1 preservando contexto en KV Cache)
        int rc = trace_lm_forward(model, kv, all_tokens, current_len, logits, 1, ws);
        if (rc < 0) {
            fprintf(stderr, "\nError en decodificacion trace_lm_forward (rc=%d).\n", rc);
            break;
        }

        if (rep_pen > 1.0f) {
            trace_lm_apply_repetition_penalty(logits, model->cfg.vocab_size, all_tokens, current_len, rep_pen);
        }

        next_tok = trace_lm_sample(logits, model->cfg.vocab_size, temperature, top_k, top_p, &rng_state);
    }

    double t_gen_total = get_time_sec() - t_gen_start;
    double tps = (gen_count > 0 && t_gen_total > 0.0) ? ((double)gen_count / t_gen_total) : 0.0;

    printf("\n\n================================================================================\n");
    printf("Tokens generados: %d | Tiempo: %.3f s | Throughput: %.2f tokens/seg\n", gen_count, t_gen_total, tps);
    printf("================================================================================\n");

    free(all_tokens);
    free(logits);
    trace_lm_workspace_free(ws);
    trace_lm_kv_free(kv);
    trace_lm_free(model);
    trace_tok_free(tok);

    return 0;
}

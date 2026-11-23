/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#include "trace_lm.h"
#include "trace_lm_kv.h"
#include "trace_tokenizer.h"
#include "paged_dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    const char* model_path = "checkpoints/trace_lm_25m_paged_bitcos.bin";
    const char* tokenizer_path = "checkpoints/tokenizer.bin";
    const char* prompt = "Basic-Lite";
    int max_tokens = 32;
    float temperature = 0.0f;
    int top_k = 50;
    float top_p = 0.95f;
    int use_int8_kv = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--int8_kv") == 0 && i + 1 < argc) use_int8_kv = atoi(argv[++i]);
    }

    printf("================================================================================\n");
    printf("Basic-Lite TRACE — Motor Nativo C Standalone (TRACELanguageModel)\n");
    printf("Motor ISA:  %s (Despacho Dinamico Universal)\n", trace_cpu_active_isa());
    printf("Modelo:     %s\n", model_path);
    printf("Tokenizador:%s\n", tokenizer_path);
    printf("Prompt:     \"%s\"\n", prompt);
    printf("Max tokens: %d | Temp: %.2f | INT8 KV: %d\n", max_tokens, temperature, use_int8_kv);
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

    // 2. Cargar tokenizador si existe
    TraceTokenizer* tok = (TraceTokenizer*)malloc(trace_tokenizer_sizeof());
    int prompt_tokens[512];
    int prompt_len = 0;

    if (tok && trace_tokenizer_load(tok, tokenizer_path) == 0) {
        prompt_len = trace_tokenizer_encode(tok, (const uint8_t*)prompt, (int)strlen(prompt), prompt_tokens, 512);
        printf("Prompt tokenizado (%d tokens):\n", prompt_len);
    } else {
        if (tok) {
            free(tok);
            tok = NULL;
        }
        printf("Aviso: tokenizador no encontrado en %s, usando tokens iniciales sinteticos.\n", tokenizer_path);
        prompt_tokens[0] = 10;
        prompt_tokens[1] = 20;
        prompt_tokens[2] = 30;
        prompt_len = 3;
    }

    // 3. Crear KV Cache y Workspace
    TraceLMKVCache* kv = trace_lm_kv_create(&model->cfg, model->cfg.max_seq_len, use_int8_kv);
    TraceLMWorkspace* ws = trace_lm_workspace_create(&model->cfg, 512);
    if (!kv || !ws) {
        fprintf(stderr, "Error al inicializar KV Cache o Workspace.\n");
        trace_lm_free(model);
        if (tok) trace_tokenizer_free(tok);
        return 1;
    }

    float* logits = (float*)malloc(sizeof(float) * model->cfg.vocab_size);
    uint64_t rng_state = 123456789ULL;

    // 4. Prefill
    double t0 = get_time_sec();
    int s_proc = trace_lm_forward(model, kv, prompt_tokens, prompt_len, logits, 1, ws);
    double ttft = get_time_sec() - t0;
    printf("\nPrefill procesado (%d tokens nuevos): TTFT = %.2f ms\n", s_proc, ttft * 1000.0);

    // 5. Decodificacion autorregresiva token a token
    printf("\n--- GENERACION ---\n");
    if (tok) {
        printf("%s", prompt);
        fflush(stdout);
    }

    int current_len = prompt_len;
    int next_tok = trace_lm_sample(logits, model->cfg.vocab_size, temperature, top_k, top_p, &rng_state);

    double t_gen_start = get_time_sec();
    int gen_count = 0;

    for (int step = 0; step < max_tokens; ++step) {
        if (tok) {
            uint8_t piece[128];
            int32_t t_id = (int32_t)next_tok;
            int plen = trace_tokenizer_decode(tok, &t_id, 1, piece, sizeof(piece) - 1);
            if (plen > 0) {
                piece[plen] = '\0';
                printf("%s", (char*)piece);
                fflush(stdout);
            }
        } else {
            printf("[%d] ", next_tok);
            fflush(stdout);
        }
        gen_count++;

        if (next_tok == model->cfg.eos_token_id) {
            break;
        }

        current_len++;
        if (current_len >= model->cfg.max_seq_len) break;

        // Evaluar siguiente token (S = 1)
        int token_buf[1] = { next_tok };
        trace_lm_forward(model, kv, token_buf, 1, logits, 1, ws);
        next_tok = trace_lm_sample(logits, model->cfg.vocab_size, temperature, top_k, top_p, &rng_state);
    }

    double t_gen_total = get_time_sec() - t_gen_start;
    double tps = (gen_count > 0 && t_gen_total > 0.0) ? ((double)gen_count / t_gen_total) : 0.0;

    printf("\n\n================================================================================\n");
    printf("Tokens generados: %d | Tiempo: %.3f s | Throughput: %.2f tokens/seg\n", gen_count, t_gen_total, tps);
    printf("================================================================================\n");

    free(logits);
    trace_lm_workspace_free(ws);
    trace_lm_kv_free(kv);
    trace_lm_free(model);
    if (tok) trace_tokenizer_free(tok);

    return 0;
}

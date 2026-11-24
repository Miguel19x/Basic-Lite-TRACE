/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#ifndef TRACE_TOKENIZER_H
#define TRACE_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRACE_DEFAULT_VOCAB_SIZE 4096
#define TRACE_SPECIAL_PAD 256
#define TRACE_SPECIAL_MASK 257
#define TRACE_SPECIAL_BOS 258
#define TRACE_SPECIAL_EOS 259
#define TRACE_FIRST_MERGE 260

typedef struct {
    uint32_t left;
    uint32_t right;
} TraceMerge;

typedef struct {
    uint32_t left;
    uint32_t right;
    int32_t rank; // -1 indica slot vacío
} TraceMergeHashEntry;

typedef struct {
    uint8_t* bytes;
    uint16_t len;
} TraceTokenPiece;

typedef struct {
    int vocab_size;
    int num_merges;
    int hash_size;
    TraceMerge* merges;
    TraceMergeHashEntry* hash_table;
    TraceTokenPiece* pieces;
    int pad_id;
    int mask_id;
    int bos_id;
    int eos_id;
} TraceTokenizer;

size_t trace_tokenizer_sizeof(void);
int trace_tokenizer_load(TraceTokenizer* tok, const char* bin_path);
void trace_tokenizer_free(TraceTokenizer* tok);

int trace_tokenizer_encode(
    const TraceTokenizer* tok,
    const uint8_t* text,
    int text_len,
    int32_t* out_tokens,
    int max_tokens
);

int trace_tokenizer_decode(
    const TraceTokenizer* tok,
    const int32_t* tokens,
    int num_tokens,
    uint8_t* out_text,
    int max_text_len
);

// Funciones de conveniencia de asignación dinámica
TraceTokenizer* trace_tok_load(const char* bin_path);
void trace_tok_free(TraceTokenizer* tok);
int trace_tok_encode(
    const TraceTokenizer* tok,
    const char* text,
    size_t text_len,
    int32_t* out_tokens,
    int max_tokens
);

#ifdef __cplusplus
}
#endif

#endif // TRACE_TOKENIZER_H

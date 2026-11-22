/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#ifndef TRACE_TOKENIZER_H
#define TRACE_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRACE_VOCAB_SIZE 4096
#define TRACE_SPECIAL_PAD 256
#define TRACE_SPECIAL_MASK 257
#define TRACE_SPECIAL_BOS 258
#define TRACE_SPECIAL_EOS 259
#define TRACE_FIRST_MERGE 260
#define TRACE_NUM_MERGES (TRACE_VOCAB_SIZE - TRACE_FIRST_MERGE)
#define TRACE_HASH_SIZE 16384 // Potencia de 2 para lookup O(1) con máscara

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
    TraceMerge merges[TRACE_NUM_MERGES];
    TraceMergeHashEntry hash_table[TRACE_HASH_SIZE];
    TraceTokenPiece pieces[TRACE_VOCAB_SIZE];
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

#ifdef __cplusplus
}
#endif

#endif // TRACE_TOKENIZER_H

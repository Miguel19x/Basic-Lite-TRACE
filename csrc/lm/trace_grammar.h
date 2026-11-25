/*
 * Basic-Lite TRACE
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#ifndef TRACE_GRAMMAR_H
#define TRACE_GRAMMAR_H

#include "trace_tokenizer.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRACE_JSON_MAX_DEPTH 16

typedef enum {
    TRACE_JSON_EXPECT_ROOT_OBJ = 0,
    TRACE_JSON_EXPECT_KEY_OR_END,
    TRACE_JSON_EXPECT_KEY,
    TRACE_JSON_EXPECT_COLON,
    TRACE_JSON_EXPECT_VALUE,
    TRACE_JSON_EXPECT_VAL_OR_ARR_END,
    TRACE_JSON_EXPECT_COMMA_OR_END,
    TRACE_JSON_IN_STRING,
    TRACE_JSON_IN_STRING_ESC,
    TRACE_JSON_IN_NUMBER_SIGN,
    TRACE_JSON_IN_NUMBER_INT,
    TRACE_JSON_IN_NUMBER_DOT,
    TRACE_JSON_IN_NUMBER_FRAC,
    TRACE_JSON_IN_LITERAL,
    TRACE_JSON_DONE
} TraceJsonState;

typedef struct {
    int state;
    int depth;
    uint8_t container_stack[TRACE_JSON_MAX_DEPTH]; // '{' o '['
    int string_is_key;
    char lit_target[6];
    int lit_pos;
} TraceJsonGrammar;

size_t trace_json_grammar_sizeof(void);
void trace_json_grammar_init(TraceJsonGrammar* g);
int trace_json_grammar_is_done(const TraceJsonGrammar* g);
int trace_json_grammar_advance_bytes(TraceJsonGrammar* g, const uint8_t* bytes, int len);
int trace_json_grammar_accept_token(TraceJsonGrammar* g, const TraceTokenizer* tok, int32_t token_id);
int trace_json_grammar_mask_logits(const TraceJsonGrammar* g, const TraceTokenizer* tok, float* logits);

#ifdef __cplusplus
}
#endif

#endif // TRACE_GRAMMAR_H

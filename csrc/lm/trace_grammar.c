/*
 * Basic-Lite TRACE
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "trace_grammar.h"
#include <math.h>
#include <string.h>

size_t trace_json_grammar_sizeof(void) {
    return sizeof(TraceJsonGrammar);
}

void trace_json_grammar_init(TraceJsonGrammar* g) {
    if (!g) return;
    memset(g, 0, sizeof(TraceJsonGrammar));
    g->state = TRACE_JSON_EXPECT_ROOT_OBJ;
    g->depth = 0;
}

int trace_json_grammar_is_done(const TraceJsonGrammar* g) {
    return (g && g->state == TRACE_JSON_DONE) ? 1 : 0;
}

static inline int is_json_ws(uint8_t c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
}

static inline int is_digit(uint8_t c) {
    return (c >= '0' && c <= '9');
}

static int close_container(TraceJsonGrammar* g, uint8_t close_ch) {
    if (g->depth <= 0) return 0;
    uint8_t top = g->container_stack[g->depth - 1];
    if ((top == '{' && close_ch != '}') || (top == '[' && close_ch != ']')) return 0;
    g->depth--;
    g->state = (g->depth == 0) ? TRACE_JSON_DONE : TRACE_JSON_EXPECT_COMMA_OR_END;
    return 1;
}

static int start_value(TraceJsonGrammar* g, uint8_t c) {
    if (c == '"') {
        g->string_is_key = 0;
        g->state = TRACE_JSON_IN_STRING;
        return 1;
    }
    if (c == '{') {
        if (g->depth >= TRACE_JSON_MAX_DEPTH) return 0;
        g->container_stack[g->depth++] = '{';
        g->state = TRACE_JSON_EXPECT_KEY_OR_END;
        return 1;
    }
    if (c == '[') {
        if (g->depth >= TRACE_JSON_MAX_DEPTH) return 0;
        g->container_stack[g->depth++] = '[';
        g->state = TRACE_JSON_EXPECT_VAL_OR_ARR_END;
        return 1;
    }
    if (c == '-') {
        g->state = TRACE_JSON_IN_NUMBER_SIGN;
        return 1;
    }
    if (is_digit(c)) {
        g->state = TRACE_JSON_IN_NUMBER_INT;
        return 1;
    }
    if (c == 't') {
        memcpy(g->lit_target, "true", 5);
        g->lit_pos = 1;
        g->state = TRACE_JSON_IN_LITERAL;
        return 1;
    }
    if (c == 'f') {
        memcpy(g->lit_target, "false", 6);
        g->lit_pos = 1;
        g->state = TRACE_JSON_IN_LITERAL;
        return 1;
    }
    if (c == 'n') {
        memcpy(g->lit_target, "null", 5);
        g->lit_pos = 1;
        g->state = TRACE_JSON_IN_LITERAL;
        return 1;
    }
    return 0;
}

static int step_byte(TraceJsonGrammar* g, uint8_t c) {
    switch (g->state) {
        case TRACE_JSON_EXPECT_ROOT_OBJ:
            if (is_json_ws(c)) return 1;
            if (c == '{') {
                g->container_stack[g->depth++] = '{';
                g->state = TRACE_JSON_EXPECT_KEY_OR_END;
                return 1;
            }
            return 0;

        case TRACE_JSON_EXPECT_KEY_OR_END:
            if (is_json_ws(c)) return 1;
            if (c == '}') return close_container(g, '}');
            if (c == '"') {
                g->string_is_key = 1;
                g->state = TRACE_JSON_IN_STRING;
                return 1;
            }
            return 0;

        case TRACE_JSON_EXPECT_KEY:
            if (is_json_ws(c)) return 1;
            if (c == '"') {
                g->string_is_key = 1;
                g->state = TRACE_JSON_IN_STRING;
                return 1;
            }
            return 0;

        case TRACE_JSON_EXPECT_COLON:
            if (is_json_ws(c)) return 1;
            if (c == ':') {
                g->state = TRACE_JSON_EXPECT_VALUE;
                return 1;
            }
            return 0;

        case TRACE_JSON_EXPECT_VALUE:
            if (is_json_ws(c)) return 1;
            return start_value(g, c);

        case TRACE_JSON_EXPECT_VAL_OR_ARR_END:
            if (is_json_ws(c)) return 1;
            if (c == ']') return close_container(g, ']');
            return start_value(g, c);

        case TRACE_JSON_EXPECT_COMMA_OR_END:
            if (is_json_ws(c)) return 1;
            if (c == ',') {
                if (g->depth <= 0) return 0;
                uint8_t top = g->container_stack[g->depth - 1];
                g->state = (top == '{') ? TRACE_JSON_EXPECT_KEY : TRACE_JSON_EXPECT_VALUE;
                return 1;
            }
            if (c == '}' || c == ']') return close_container(g, c);
            return 0;

        case TRACE_JSON_IN_STRING:
            if (c < 0x20) return 0; // Caracteres de control sin escapar prohibidos en JSON
            if (c == '\\') {
                g->state = TRACE_JSON_IN_STRING_ESC;
                return 1;
            }
            if (c == '"') {
                g->state = g->string_is_key ? TRACE_JSON_EXPECT_COLON : TRACE_JSON_EXPECT_COMMA_OR_END;
                return 1;
            }
            return 1;

        case TRACE_JSON_IN_STRING_ESC:
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
                g->state = TRACE_JSON_IN_STRING;
                return 1;
            }
            return 0;

        case TRACE_JSON_IN_NUMBER_SIGN:
            if (is_digit(c)) {
                g->state = TRACE_JSON_IN_NUMBER_INT;
                return 1;
            }
            return 0;

        case TRACE_JSON_IN_NUMBER_INT:
            if (is_digit(c)) return 1;
            if (c == '.') {
                g->state = TRACE_JSON_IN_NUMBER_DOT;
                return 1;
            }
            if (is_json_ws(c)) {
                g->state = TRACE_JSON_EXPECT_COMMA_OR_END;
                return 1;
            }
            if (c == ',' || c == '}' || c == ']') {
                g->state = TRACE_JSON_EXPECT_COMMA_OR_END;
                return step_byte(g, c);
            }
            return 0;

        case TRACE_JSON_IN_NUMBER_DOT:
            if (is_digit(c)) {
                g->state = TRACE_JSON_IN_NUMBER_FRAC;
                return 1;
            }
            return 0;

        case TRACE_JSON_IN_NUMBER_FRAC:
            if (is_digit(c)) return 1;
            if (is_json_ws(c)) {
                g->state = TRACE_JSON_EXPECT_COMMA_OR_END;
                return 1;
            }
            if (c == ',' || c == '}' || c == ']') {
                g->state = TRACE_JSON_EXPECT_COMMA_OR_END;
                return step_byte(g, c);
            }
            return 0;

        case TRACE_JSON_IN_LITERAL:
            if (g->lit_target[g->lit_pos] == (char)c) {
                g->lit_pos++;
                if (g->lit_target[g->lit_pos] == '\0') {
                    g->state = TRACE_JSON_EXPECT_COMMA_OR_END;
                }
                return 1;
            }
            return 0;

        case TRACE_JSON_DONE:
            return 0;

        default:
            return 0;
    }
}

int trace_json_grammar_advance_bytes(TraceJsonGrammar* g, const uint8_t* bytes, int len) {
    if (!g || !bytes || len <= 0) return 0;
    TraceJsonGrammar tmp = *g;
    for (int i = 0; i < len; ++i) {
        if (!step_byte(&tmp, bytes[i])) return 0;
    }
    *g = tmp;
    return 1;
}

static inline int is_stop_token(const TraceTokenizer* tok, int32_t tid) {
    return (tid == tok->eos_id) || (tok->eot_id >= 0 && tid == tok->eot_id);
}

static inline int is_control_token(const TraceTokenizer* tok, int32_t tid) {
    return (tid == tok->pad_id || tid == tok->mask_id || tid == tok->bos_id || tid == tok->eos_id ||
            (tok->start_header_id >= 0 && tid == tok->start_header_id) ||
            (tok->end_header_id >= 0 && tid == tok->end_header_id) ||
            (tok->eot_id >= 0 && tid == tok->eot_id));
}

int trace_json_grammar_accept_token(TraceJsonGrammar* g, const TraceTokenizer* tok, int32_t token_id) {
    if (!g || !tok || token_id < 0 || token_id >= tok->vocab_size) return 0;
    if (g->state == TRACE_JSON_DONE) {
        return is_stop_token(tok, token_id) ? 1 : 0;
    }
    if (is_control_token(tok, token_id)) return 0;
    const TraceTokenPiece* p = &tok->pieces[token_id];
    if (!p->bytes || p->len == 0) return 0;
    return trace_json_grammar_advance_bytes(g, p->bytes, (int)p->len);
}

int trace_json_grammar_mask_logits(const TraceJsonGrammar* g, const TraceTokenizer* tok, float* logits) {
    if (!g || !tok || !logits || tok->vocab_size <= 0) return 0;
    int vocab_size = tok->vocab_size;
    int allowed = 0;

    if (g->state == TRACE_JSON_DONE) {
        for (int v = 0; v < vocab_size; ++v) {
            if (is_stop_token(tok, v)) {
                allowed++;
            } else {
                logits[v] = -INFINITY;
            }
        }
        return allowed;
    }

    #pragma omp parallel for schedule(static) reduction(+:allowed) if (vocab_size >= 1024)
    for (int v = 0; v < vocab_size; ++v) {
        if (is_control_token(tok, v)) {
            logits[v] = -INFINITY;
            continue;
        }
        const TraceTokenPiece* p = &tok->pieces[v];
        if (!p->bytes || p->len == 0) {
            logits[v] = -INFINITY;
            continue;
        }
        TraceJsonGrammar tmp = *g;
        if (trace_json_grammar_advance_bytes(&tmp, p->bytes, (int)p->len)) {
            allowed++;
        } else {
            logits[v] = -INFINITY;
        }
    }
    return allowed;
}

/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 * SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
 */

#include "trace_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;      // 0x4B545254 ("TRTK")
    uint32_t version;    // 1
    uint32_t vocab_size; // Tamaño dinámico (e.g. 4096, 8192)
    uint32_t num_merges; // Número de merges BPE
    uint32_t pad_id;
    uint32_t mask_id;
    uint32_t bos_id;
    uint32_t eos_id;
} TokenizerHeader;
#pragma pack(pop)

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(TokenizerHeader) == 32, "TokenizerHeader debe medir exactamente 32 bytes");
_Static_assert(sizeof(TraceMerge) == 8, "TraceMerge debe medir exactamente 8 bytes");
#endif

typedef enum {
    CAT_DIGIT,
    CAT_WORD,
    CAT_SPACE,
    CAT_NEWLINE,
    CAT_PUNCT,
    CAT_OTHER
} CodepointCat;

size_t trace_tokenizer_sizeof(void) {
    return sizeof(TraceTokenizer);
}

static inline uint32_t hash_pair(uint32_t left, uint32_t right, int hash_size) {
    uint64_t key = ((uint64_t)left << 32) | right;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)key & (uint32_t)(hash_size - 1);
}

static inline int utf8_decode(const uint8_t* s, int len, uint32_t* cp, int* cp_len) {
    if (len <= 0) return 0;
    uint8_t b0 = s[0];
    if (b0 < 0x80) {
        *cp = b0;
        *cp_len = 1;
        return 1;
    }
    if ((b0 & 0xE0) == 0xC0 && len >= 2 && (s[1] & 0xC0) == 0x80) {
        *cp = ((b0 & 0x1F) << 6) | (s[1] & 0x3F);
        *cp_len = 2;
        return 1;
    }
    if ((b0 & 0xF0) == 0xE0 && len >= 3 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((b0 & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        *cp_len = 3;
        return 1;
    }
    if ((b0 & 0xF8) == 0xF0 && len >= 4 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *cp = ((b0 & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        *cp_len = 4;
        return 1;
    }
    *cp = b0;
    *cp_len = 1;
    return 1;
}

static inline CodepointCat classify_cp(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return CAT_DIGIT;
    if (cp == ' ' || cp == '\t') return CAT_SPACE;
    if (cp == '\r' || cp == '\n') return CAT_NEWLINE;
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp == '_') return CAT_WORD;
    if ((cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0x024F)) return CAT_WORD;
    if (cp >= 33 && cp <= 126) return CAT_PUNCT;
    return CAT_OTHER;
}

void trace_tokenizer_free(TraceTokenizer* tok) {
    if (!tok) return;
    if (tok->pieces) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->pieces[i].bytes) {
                free(tok->pieces[i].bytes);
                tok->pieces[i].bytes = NULL;
            }
            tok->pieces[i].len = 0;
        }
        free(tok->pieces);
        tok->pieces = NULL;
    }
    if (tok->hash_table) {
        free(tok->hash_table);
        tok->hash_table = NULL;
    }
    if (tok->merges) {
        free(tok->merges);
        tok->merges = NULL;
    }
    tok->vocab_size = 0;
    tok->num_merges = 0;
    tok->hash_size = 0;
}

int trace_tokenizer_load(TraceTokenizer* tok, const char* bin_path) {
    if (!tok || !bin_path) return -1;
    memset(tok, 0, sizeof(TraceTokenizer));

    FILE* f = fopen(bin_path, "rb");
    if (!f) return -2;

    uint32_t magic = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x4B545254) {
        fclose(f);
        return -3;
    }

    uint32_t w1 = 0, w2 = 0;
    if (fread(&w1, sizeof(uint32_t), 1, f) != 1 || fread(&w2, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return -3;
    }

    int is_compact = 0;
    uint32_t vocab_u32 = 0;
    uint32_t merges_u32 = 0;
    if (w1 == 1) {
        // Formato estándar de 32 bytes (versión 1)
        uint32_t rest[5];
        if (fread(rest, sizeof(uint32_t), 5, f) != 5) {
            fclose(f);
            return -4;
        }
        vocab_u32 = w2;
        merges_u32 = rest[0];
        tok->pad_id = (int)rest[1];
        tok->mask_id = (int)rest[2];
        tok->bos_id = (int)rest[3];
        tok->eos_id = (int)rest[4];
    } else if (w1 >= TRACE_FIRST_MERGE) {
        // Formato compacto (w1 = vocab_size, w2 = num_merges)
        is_compact = 1;
        vocab_u32 = w1;
        merges_u32 = w2;
        tok->pad_id = TRACE_SPECIAL_PAD;
        tok->mask_id = TRACE_SPECIAL_MASK;
        tok->bos_id = TRACE_SPECIAL_BOS;
        tok->eos_id = TRACE_SPECIAL_EOS;
    } else {
        fclose(f);
        return -4;
    }

    if (vocab_u32 < (uint32_t)TRACE_FIRST_MERGE ||
        vocab_u32 > (1u << 20) ||
        merges_u32 > vocab_u32 - (uint32_t)TRACE_FIRST_MERGE) {
        fclose(f);
        return -4;
    }

    tok->vocab_size = (int)vocab_u32;
    tok->num_merges = (int)merges_u32;

    if (tok->num_merges > 0) {
        tok->merges = (TraceMerge*)malloc(sizeof(TraceMerge) * (size_t)tok->num_merges);
        if (!tok->merges) {
            fclose(f);
            return -5;
        }

        if (is_compact) {
            for (int i = 0; i < tok->num_merges; i++) {
                uint16_t pair[2];
                if (fread(pair, sizeof(uint16_t), 2, f) != 2) {
                    trace_tokenizer_free(tok);
                    fclose(f);
                    return -5;
                }
                tok->merges[i].left = pair[0];
                tok->merges[i].right = pair[1];
            }
        } else {
            if (fread(tok->merges, sizeof(TraceMerge), (size_t)tok->num_merges, f) != (size_t)tok->num_merges) {
                trace_tokenizer_free(tok);
                fclose(f);
                return -5;
            }
        }
    }

    // Tabla hash O(1) con tamaño potencia de 2 >= num_merges * 4 (en size_t sin desbordamiento)
    size_t h_sz = 1024;
    size_t min_h_sz = (size_t)tok->num_merges * 4u;
    while (h_sz < min_h_sz && h_sz < ((size_t)1u << 24)) {
        h_sz <<= 1;
    }
    tok->hash_size = (int)h_sz;
    tok->hash_table = (TraceMergeHashEntry*)malloc(sizeof(TraceMergeHashEntry) * h_sz);
    if (!tok->hash_table) {
        trace_tokenizer_free(tok);
        fclose(f);
        return -5;
    }

    for (size_t i = 0; i < h_sz; i++) {
        tok->hash_table[i].rank = -1;
    }
    for (int i = 0; i < tok->num_merges; i++) {
        uint32_t l = tok->merges[i].left;
        uint32_t r = tok->merges[i].right;
        uint32_t h = hash_pair(l, r, tok->hash_size);
        int already_present = 0;
        uint32_t orig_h = h;
        while (tok->hash_table[h].rank != -1) {
            if (tok->hash_table[h].left == l && tok->hash_table[h].right == r) {
                already_present = 1;
                break;
            }
            h = (h + 1) & (uint32_t)(tok->hash_size - 1);
            if (h == orig_h) break;
        }
        if (!already_present) {
            tok->hash_table[h].left = l;
            tok->hash_table[h].right = r;
            tok->hash_table[h].rank = i;
        }
    }

    tok->pieces = (TraceTokenPiece*)calloc((size_t)tok->vocab_size, sizeof(TraceTokenPiece));
    if (!tok->pieces) {
        trace_tokenizer_free(tok);
        fclose(f);
        return -6;
    }

    if (is_compact) {
        // Reconstrucción determinista de piezas a partir de bytes base y merges
        for (int i = 0; i < 256; i++) {
            tok->pieces[i].len = 1;
            tok->pieces[i].bytes = (uint8_t*)malloc(2);
            if (!tok->pieces[i].bytes) {
                trace_tokenizer_free(tok);
                fclose(f);
                return -6;
            }
            tok->pieces[i].bytes[0] = (uint8_t)i;
            tok->pieces[i].bytes[1] = 0;
        }
        const char* specials[4] = {"<PAD>", "<MASK>", "<BOS>", "<EOS>"};
        for (int i = 0; i < 4; i++) {
            int idx = 256 + i;
            if (idx < tok->vocab_size) {
                int slen = (int)strlen(specials[i]);
                tok->pieces[idx].len = (uint16_t)slen;
                tok->pieces[idx].bytes = (uint8_t*)malloc((size_t)slen + 1);
                if (!tok->pieces[idx].bytes) {
                    trace_tokenizer_free(tok);
                    fclose(f);
                    return -6;
                }
                memcpy(tok->pieces[idx].bytes, specials[i], (size_t)slen + 1);
            }
        }
        for (int i = 0; i < tok->num_merges; i++) {
            int idx = TRACE_FIRST_MERGE + i;
            if (idx >= tok->vocab_size) break;
            uint32_t l = tok->merges[i].left;
            uint32_t r = tok->merges[i].right;
            int l_len = (l < (uint32_t)idx && tok->pieces[l].bytes) ? (int)tok->pieces[l].len : 0;
            int r_len = (r < (uint32_t)idx && tok->pieces[r].bytes) ? (int)tok->pieces[r].len : 0;
            tok->pieces[idx].len = (uint16_t)(l_len + r_len);
            tok->pieces[idx].bytes = (uint8_t*)malloc((size_t)l_len + (size_t)r_len + 1);
            if (!tok->pieces[idx].bytes) {
                trace_tokenizer_free(tok);
                fclose(f);
                return -6;
            }
            if (l_len > 0) memcpy(tok->pieces[idx].bytes, tok->pieces[l].bytes, (size_t)l_len);
            if (r_len > 0) memcpy(tok->pieces[idx].bytes + l_len, tok->pieces[r].bytes, (size_t)r_len);
            tok->pieces[idx].bytes[l_len + r_len] = 0;
        }
    } else {
        for (int i = 0; i < tok->vocab_size; i++) {
            uint16_t len = 0;
            if (fread(&len, sizeof(uint16_t), 1, f) != 1) {
                trace_tokenizer_free(tok);
                fclose(f);
                return -7;
            }
            tok->pieces[i].len = len;
            tok->pieces[i].bytes = (uint8_t*)malloc((size_t)len + 1);
            if (!tok->pieces[i].bytes) {
                trace_tokenizer_free(tok);
                fclose(f);
                return -7;
            }
            if (len > 0) {
                if (fread(tok->pieces[i].bytes, 1, len, f) != (size_t)len) {
                    trace_tokenizer_free(tok);
                    fclose(f);
                    return -7;
                }
            }
            tok->pieces[i].bytes[len] = '\0';
        }
    }

    fclose(f);
    return 0;
}

static inline int find_merge_rank(const TraceTokenizer* tok, int32_t left, int32_t right) {
    if (!tok || !tok->hash_table || tok->hash_size <= 0) return -1;
    uint32_t h = hash_pair((uint32_t)left, (uint32_t)right, tok->hash_size);
    uint32_t mask = (uint32_t)(tok->hash_size - 1);
    for (int probes = 0; probes < tok->hash_size; ++probes) {
        if (tok->hash_table[h].rank == -1) return -1;
        if ((int32_t)tok->hash_table[h].left == left && (int32_t)tok->hash_table[h].right == right) {
            return tok->hash_table[h].rank;
        }
        h = (h + 1) & mask;
    }
    return -1;
}

static int bpe_merge_segment(
    const TraceTokenizer* tok,
    const uint8_t* seg,
    int seg_len,
    int32_t* out_tokens,
    int max_tokens
) {
    if (seg_len <= 0) return 0;
    if (seg_len == 1) {
        if (max_tokens > 0) {
            out_tokens[0] = seg[0];
            return 1;
        }
        return 0;
    }

    int32_t stack_tokens[256];
    int32_t* tokens = stack_tokens;
    int is_alloc = 0;

    if (seg_len > 256) {
        tokens = (int32_t*)malloc((size_t)seg_len * sizeof(int32_t));
        if (!tokens) return 0;
        is_alloc = 1;
    }

    int n = seg_len;
    for (int i = 0; i < n; i++) {
        tokens[i] = seg[i];
    }

    while (n >= 2) {
        int best_rank = tok->num_merges;
        int best_idx = -1;

        for (int i = 0; i < n - 1; i++) {
            int rank = find_merge_rank(tok, tokens[i], tokens[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = TRACE_FIRST_MERGE + best_rank;
        for (int i = best_idx + 1; i < n - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n--;
    }

    int count = n < max_tokens ? n : max_tokens;
    for (int i = 0; i < count; i++) {
        out_tokens[i] = tokens[i];
    }

    if (is_alloc) {
        free(tokens);
    }

    return count;
}

int trace_tokenizer_encode(
    const TraceTokenizer* tok,
    const uint8_t* text,
    int text_len,
    int32_t* out_tokens,
    int max_tokens
) {
    if (!tok || !text || text_len <= 0 || !out_tokens || max_tokens <= 0) return 0;

    int total_tokens = 0;
    int i = 0;

    while (i < text_len && total_tokens < max_tokens) {
        uint32_t cp = 0;
        int cp_len = 0;
        utf8_decode(text + i, text_len - i, &cp, &cp_len);
        CodepointCat cat = classify_cp(cp);

        // Regla 1: Dígito ASCII estrictamente aislado en 1 byte
        if (cat == CAT_DIGIT) {
            int written = bpe_merge_segment(tok, text + i, 1, out_tokens + total_tokens, max_tokens - total_tokens);
            total_tokens += written;
            i += 1;
            continue;
        }

        // Regla 2: Emojis y multibyte no-palabra aislados individualmente
        if (cat == CAT_OTHER) {
            int written = bpe_merge_segment(tok, text + i, cp_len, out_tokens + total_tokens, max_tokens - total_tokens);
            total_tokens += written;
            i += cp_len;
            continue;
        }

        // Prefijo de espacio único en palabras (' ' + WORD)
        if (cat == CAT_SPACE && cp == ' ' && cp_len == 1) {
            if (i + 1 < text_len) {
                uint32_t next_cp = 0;
                int next_len = 0;
                utf8_decode(text + i + 1, text_len - (i + 1), &next_cp, &next_len);
                if (classify_cp(next_cp) == CAT_WORD) {
                    int j = i + 1 + next_len;
                    while (j < text_len) {
                        uint32_t w_cp = 0;
                        int w_len = 0;
                        utf8_decode(text + j, text_len - j, &w_cp, &w_len);
                        if (classify_cp(w_cp) != CAT_WORD) break;
                        j += w_len;
                    }
                    int seg_len = j - i;
                    int written = bpe_merge_segment(tok, text + i, seg_len, out_tokens + total_tokens, max_tokens - total_tokens);
                    total_tokens += written;
                    i = j;
                    continue;
                }
            }
        }

        // Corridas regulares de la misma categoría
        int j = i + cp_len;
        while (j < text_len) {
            uint32_t next_cp = 0;
            int next_len = 0;
            utf8_decode(text + j, text_len - j, &next_cp, &next_len);
            if (classify_cp(next_cp) != cat) break;
            j += next_len;
        }

        int seg_len = j - i;
        int written = bpe_merge_segment(tok, text + i, seg_len, out_tokens + total_tokens, max_tokens - total_tokens);
        total_tokens += written;
        i = j;
    }

    return total_tokens;
}

int trace_tokenizer_decode(
    const TraceTokenizer* tok,
    const int32_t* tokens,
    int num_tokens,
    uint8_t* out_text,
    int max_text_len
) {
    if (!out_text || max_text_len <= 0) return 0;
    out_text[0] = '\0';
    if (!tok || !tokens || num_tokens <= 0) return 0;

    int total_bytes = 0;
    for (int i = 0; i < num_tokens; i++) {
        int32_t tid = tokens[i];
        if (tid < 0 || tid >= tok->vocab_size) continue;
        if (tid == tok->pad_id || tid == tok->mask_id || tid == tok->bos_id || tid == tok->eos_id) continue;

        int len = (int)tok->pieces[tid].len;
        if (total_bytes + len >= max_text_len) break;

        if (len > 0 && tok->pieces[tid].bytes) {
            memcpy(out_text + total_bytes, tok->pieces[tid].bytes, (size_t)len);
            total_bytes += len;
        }
    }

    out_text[total_bytes] = '\0';
    return total_bytes;
}

TraceTokenizer* trace_tok_load(const char* bin_path) {
    if (!bin_path) return NULL;
    TraceTokenizer* tok = (TraceTokenizer*)malloc(sizeof(TraceTokenizer));
    if (!tok) return NULL;
    if (trace_tokenizer_load(tok, bin_path) != 0) {
        free(tok);
        return NULL;
    }
    return tok;
}

void trace_tok_free(TraceTokenizer* tok) {
    if (tok) {
        trace_tokenizer_free(tok);
        free(tok);
    }
}

int trace_tok_encode(
    const TraceTokenizer* tok,
    const char* text,
    size_t text_len,
    int32_t* out_tokens,
    int max_tokens
) {
    if (!tok || !text || !out_tokens || max_tokens <= 0) return 0;
    return trace_tokenizer_encode(tok, (const uint8_t*)text, (int)text_len, out_tokens, max_tokens);
}

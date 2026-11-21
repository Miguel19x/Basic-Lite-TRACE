/*
 * Basic-Lite
 * Copyright © 2026 Miguel19x
 * Todos los derechos reservados.
 * Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
 */

#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#pragma pack(push, 1)
typedef struct {
    uint32_t row_start;
    uint32_t rows;
    uint16_t cols;
    uint16_t fmt_code; // 0: compact, 1: fast
    float scale_w;
    uint32_t len_p1;
    uint32_t len_p2;
} PagedHeader;
#pragma pack(pop)

/*
 * Kernel C optimizado para una página individual Paged BitCOS Fast Format.
 */
void paged_direct_matmul_fast_c(
    const int8_t* x,              // [M, K]
    const uint8_t* page_blob,     // Búfer completo de la página binaria
    int32_t* out,                 // [M, rows]
    int64_t M,
    int64_t K
) {
    const PagedHeader* hdr = (const PagedHeader*)page_blob;
    if (hdr->fmt_code != 1) {
        return; // Solo admite fast format (fmt_code == 1)
    }
    const uint32_t rows = hdr->rows;
    const uint16_t cols = hdr->cols;
    const uint8_t* payload = page_blob + sizeof(PagedHeader);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t r = 0; r < (int64_t)rows; ++r) {
            const int8_t* x_row = x + m * K;
            int64_t row_weight_offset = r * (int64_t)cols;
            int32_t acc = 0;

            int64_t k = 0;
            if ((row_weight_offset & 3) == 0) {
                const uint8_t* w_packed = payload + (row_weight_offset >> 2);

                for (; k + 16 <= cols; k += 16) {
                    uint8_t b0 = w_packed[(k >> 2) + 0];
                    int32_t w0 = ((int32_t)(b0 & 0x03)) - 1;
                    int32_t w1 = ((int32_t)((b0 >> 2) & 0x03)) - 1;
                    int32_t w2 = ((int32_t)((b0 >> 4) & 0x03)) - 1;
                    int32_t w3 = ((int32_t)((b0 >> 6) & 0x03)) - 1;
                    acc += (int32_t)x_row[k + 0] * w0;
                    acc += (int32_t)x_row[k + 1] * w1;
                    acc += (int32_t)x_row[k + 2] * w2;
                    acc += (int32_t)x_row[k + 3] * w3;

                    uint8_t b1 = w_packed[(k >> 2) + 1];
                    int32_t w4 = ((int32_t)(b1 & 0x03)) - 1;
                    int32_t w5 = ((int32_t)((b1 >> 2) & 0x03)) - 1;
                    int32_t w6 = ((int32_t)((b1 >> 4) & 0x03)) - 1;
                    int32_t w7 = ((int32_t)((b1 >> 6) & 0x03)) - 1;
                    acc += (int32_t)x_row[k + 4] * w4;
                    acc += (int32_t)x_row[k + 5] * w5;
                    acc += (int32_t)x_row[k + 6] * w6;
                    acc += (int32_t)x_row[k + 7] * w7;

                    uint8_t b2 = w_packed[(k >> 2) + 2];
                    int32_t w8 = ((int32_t)(b2 & 0x03)) - 1;
                    int32_t w9 = ((int32_t)((b2 >> 2) & 0x03)) - 1;
                    int32_t w10 = ((int32_t)((b2 >> 4) & 0x03)) - 1;
                    int32_t w11 = ((int32_t)((b2 >> 6) & 0x03)) - 1;
                    acc += (int32_t)x_row[k + 8] * w8;
                    acc += (int32_t)x_row[k + 9] * w9;
                    acc += (int32_t)x_row[k + 10] * w10;
                    acc += (int32_t)x_row[k + 11] * w11;

                    uint8_t b3 = w_packed[(k >> 2) + 3];
                    int32_t w12 = ((int32_t)(b3 & 0x03)) - 1;
                    int32_t w13 = ((int32_t)((b3 >> 2) & 0x03)) - 1;
                    int32_t w14 = ((int32_t)((b3 >> 4) & 0x03)) - 1;
                    int32_t w15 = ((int32_t)((b3 >> 6) & 0x03)) - 1;
                    acc += (int32_t)x_row[k + 12] * w12;
                    acc += (int32_t)x_row[k + 13] * w13;
                    acc += (int32_t)x_row[k + 14] * w14;
                    acc += (int32_t)x_row[k + 15] * w15;
                }

                for (; k + 4 <= cols; k += 4) {
                    uint8_t b = w_packed[k >> 2];
                    int32_t w0 = ((int32_t)(b & 0x03)) - 1;
                    int32_t w1 = ((int32_t)((b >> 2) & 0x03)) - 1;
                    int32_t w2 = ((int32_t)((b >> 4) & 0x03)) - 1;
                    int32_t w3 = ((int32_t)((b >> 6) & 0x03)) - 1;
                    acc += (int32_t)x_row[k + 0] * w0;
                    acc += (int32_t)x_row[k + 1] * w1;
                    acc += (int32_t)x_row[k + 2] * w2;
                    acc += (int32_t)x_row[k + 3] * w3;
                }
            }

            for (; k < cols; ++k) {
                int64_t global_idx = row_weight_offset + k;
                uint8_t byte_val = payload[global_idx >> 2];
                int shift = (global_idx & 3) * 2;
                int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                acc += (int32_t)x_row[k] * w;
            }

            out[m * rows + r] = acc;
        }
    }
}

/*
 * Kernel Fisionado Multi-Página: procesa todas las páginas del tensor en un único pase
 * escribiendo directamente en la matriz de salida [M, total_out_features].
 */
void paged_direct_matmul_fast_c_multi(
    const int8_t* x,                  // [M, K]
    const uint8_t* const* page_blobs, // Array de punteros a cada página
    int64_t num_pages,
    int32_t* out,                     // [M, total_out_features]
    int64_t M,
    int64_t K,
    int64_t total_out_features
) {
    #pragma omp parallel for schedule(dynamic)
    for (int64_t p = 0; p < num_pages; ++p) {
        const uint8_t* blob = page_blobs[p];
        const PagedHeader* hdr = (const PagedHeader*)blob;
        const uint32_t row_start = hdr->row_start;
        const uint32_t rows = hdr->rows;
        const uint16_t cols = hdr->cols;
        const uint8_t* payload = blob + sizeof(PagedHeader);

        for (int64_t m = 0; m < M; ++m) {
            const int8_t* x_row = x + m * K;
            int32_t* out_row = out + m * total_out_features + row_start;

            for (int64_t r = 0; r < (int64_t)rows; ++r) {
                int64_t row_weight_offset = r * (int64_t)cols;
                int32_t acc = 0;
                int64_t k = 0;

                if ((row_weight_offset & 3) == 0) {
                    const uint8_t* w_packed = payload + (row_weight_offset >> 2);
                    for (; k + 16 <= cols; k += 16) {
                        uint8_t b0 = w_packed[(k >> 2) + 0];
                        uint8_t b1 = w_packed[(k >> 2) + 1];
                        uint8_t b2 = w_packed[(k >> 2) + 2];
                        uint8_t b3 = w_packed[(k >> 2) + 3];

                        acc += (int32_t)x_row[k + 0] * (((int32_t)(b0 & 0x03)) - 1)
                             + (int32_t)x_row[k + 1] * (((int32_t)((b0 >> 2) & 0x03)) - 1)
                             + (int32_t)x_row[k + 2] * (((int32_t)((b0 >> 4) & 0x03)) - 1)
                             + (int32_t)x_row[k + 3] * (((int32_t)((b0 >> 6) & 0x03)) - 1)
                             + (int32_t)x_row[k + 4] * (((int32_t)(b1 & 0x03)) - 1)
                             + (int32_t)x_row[k + 5] * (((int32_t)((b1 >> 2) & 0x03)) - 1)
                             + (int32_t)x_row[k + 6] * (((int32_t)((b1 >> 4) & 0x03)) - 1)
                             + (int32_t)x_row[k + 7] * (((int32_t)((b1 >> 6) & 0x03)) - 1)
                             + (int32_t)x_row[k + 8] * (((int32_t)(b2 & 0x03)) - 1)
                             + (int32_t)x_row[k + 9] * (((int32_t)((b2 >> 2) & 0x03)) - 1)
                             + (int32_t)x_row[k + 10] * (((int32_t)((b2 >> 4) & 0x03)) - 1)
                             + (int32_t)x_row[k + 11] * (((int32_t)((b2 >> 6) & 0x03)) - 1)
                             + (int32_t)x_row[k + 12] * (((int32_t)(b3 & 0x03)) - 1)
                             + (int32_t)x_row[k + 13] * (((int32_t)((b3 >> 2) & 0x03)) - 1)
                             + (int32_t)x_row[k + 14] * (((int32_t)((b3 >> 4) & 0x03)) - 1)
                             + (int32_t)x_row[k + 15] * (((int32_t)((b3 >> 6) & 0x03)) - 1);
                    }
                    for (; k + 4 <= cols; k += 4) {
                        uint8_t b = w_packed[k >> 2];
                        acc += (int32_t)x_row[k + 0] * (((int32_t)(b & 0x03)) - 1)
                             + (int32_t)x_row[k + 1] * (((int32_t)((b >> 2) & 0x03)) - 1)
                             + (int32_t)x_row[k + 2] * (((int32_t)((b >> 4) & 0x03)) - 1)
                             + (int32_t)x_row[k + 3] * (((int32_t)((b >> 6) & 0x03)) - 1);
                    }
                }
                for (; k < cols; ++k) {
                    int64_t global_idx = row_weight_offset + k;
                    uint8_t byte_val = payload[global_idx >> 2];
                    int shift = (global_idx & 3) * 2;
                    int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
                    acc += (int32_t)x_row[k] * w;
                }
                out_row[r] = acc;
            }
        }
    }
}

/*
 * Multiplica un vector x de dimensión K por una matriz de pesos ternarios 2-bit
 * empaquetada (rows x K), acumulando en out [rows].
 */
void paged_lowrank_matmul_row_c(
    const int8_t* x,          // [K]
    const uint8_t* w_packed,  // [rows * (K / 4)]
    int32_t* out,             // [rows]
    int64_t rows,
    int64_t K
) {
    int64_t bytes_per_row = K >> 2;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_w = w_packed + r * bytes_per_row;
        int32_t acc = 0;
        int64_t k = 0;
        for (; k + 16 <= K; k += 16) {
            uint8_t b0 = row_w[(k >> 2) + 0];
            uint8_t b1 = row_w[(k >> 2) + 1];
            uint8_t b2 = row_w[(k >> 2) + 2];
            uint8_t b3 = row_w[(k >> 2) + 3];
            acc += (int32_t)x[k + 0] * (((int32_t)(b0 & 0x03)) - 1)
                 + (int32_t)x[k + 1] * (((int32_t)((b0 >> 2) & 0x03)) - 1)
                 + (int32_t)x[k + 2] * (((int32_t)((b0 >> 4) & 0x03)) - 1)
                 + (int32_t)x[k + 3] * (((int32_t)((b0 >> 6) & 0x03)) - 1)
                 + (int32_t)x[k + 4] * (((int32_t)(b1 & 0x03)) - 1)
                 + (int32_t)x[k + 5] * (((int32_t)((b1 >> 2) & 0x03)) - 1)
                 + (int32_t)x[k + 6] * (((int32_t)((b1 >> 4) & 0x03)) - 1)
                 + (int32_t)x[k + 7] * (((int32_t)((b1 >> 6) & 0x03)) - 1)
                 + (int32_t)x[k + 8] * (((int32_t)(b2 & 0x03)) - 1)
                 + (int32_t)x[k + 9] * (((int32_t)((b2 >> 2) & 0x03)) - 1)
                 + (int32_t)x[k + 10] * (((int32_t)((b2 >> 4) & 0x03)) - 1)
                 + (int32_t)x[k + 11] * (((int32_t)((b2 >> 6) & 0x03)) - 1)
                 + (int32_t)x[k + 12] * (((int32_t)(b3 & 0x03)) - 1)
                 + (int32_t)x[k + 13] * (((int32_t)((b3 >> 2) & 0x03)) - 1)
                 + (int32_t)x[k + 14] * (((int32_t)((b3 >> 4) & 0x03)) - 1)
                 + (int32_t)x[k + 15] * (((int32_t)((b3 >> 6) & 0x03)) - 1);
        }
        for (; k + 4 <= K; k += 4) {
            uint8_t b = row_w[k >> 2];
            acc += (int32_t)x[k + 0] * (((int32_t)(b & 0x03)) - 1)
                 + (int32_t)x[k + 1] * (((int32_t)((b >> 2) & 0x03)) - 1)
                 + (int32_t)x[k + 2] * (((int32_t)((b >> 4) & 0x03)) - 1)
                 + (int32_t)x[k + 3] * (((int32_t)((b >> 6) & 0x03)) - 1);
        }
        for (; k < K; ++k) {
            uint8_t byte_val = row_w[k >> 2];
            int shift = (k & 3) * 2;
            int32_t w = ((int32_t)((byte_val >> shift) & 0x03)) - 1;
            acc += (int32_t)x[k] * w;
        }
        out[r] = acc;
    }
}

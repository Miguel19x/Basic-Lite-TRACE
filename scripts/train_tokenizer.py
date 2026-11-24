#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Script de Entrenamiento del Vocabulario BPE 4096 para Basic-Lite TRACE
Entrena de forma 100% determinista sobre un corpus de texto crudo UTF-8,
garantizando cero fusiones con dígitos ASCII ('0'..'9') y reportando
la tasa de compresión (bytes/token).
"""

import argparse
import ctypes
import os
import time
from trace.lm.tokenizer import FIRST_MERGE, VOCAB_SIZE, TRACETokenizer

IN_DOMAIN_PROMPTS = [
    "La arquitectura Basic-Lite TRACE opera con cuantización ternaria W1.58A8 en CPU.",
    "El algoritmo de atención MLA comprime los estados KV hacia un cuello de botella latente.",
    "def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:\n    return self.sub_ln(x)\n",
    "#include <stdio.h>\nint main(int argc, char** argv) { return 0; }",
]

GENERAL_PROMPTS = [
    "El cielo de la tarde se teñía de tonos naranjas mientras caminaban junto al río en silencio.",
    "The quick brown fox jumps over the lazy dog near the old wooden bridge in autumn.",
    "La fotosíntesis permite a las plantas transformar la energía lumínica en glucosa y oxígeno.",
    "Modern operating systems manage concurrent threads through preemptive scheduling and memory paging.",
]


def load_c_tokenizer_lib():
    so_path = os.path.abspath("libtrace_tokenizer.so")
    assert os.path.exists(so_path), f"Falta {so_path}. Compila con make primero."
    lib = ctypes.CDLL(so_path)
    lib.trace_tokenizer_sizeof.restype = ctypes.c_size_t
    lib.trace_tokenizer_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.trace_tokenizer_load.restype = ctypes.c_int
    lib.trace_tokenizer_free.argtypes = [ctypes.c_void_p]
    lib.trace_tokenizer_free.restype = None
    lib.trace_tokenizer_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.trace_tokenizer_encode.restype = ctypes.c_int
    lib.trace_tokenizer_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_tokenizer_decode.restype = ctypes.c_int
    return lib


def load_corpus_files(file_paths: list[str]) -> list[str]:
    texts = []
    total_bytes = 0
    for p in file_paths:
        if not os.path.exists(p):
            print(f"[Aviso] Archivo no encontrado: {p}, omitiendo.")
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            content = f.read()
            texts.append(content)
            total_bytes += len(content.encode("utf-8"))
        print(f"  - Cargado: {p} ({len(content.encode('utf-8')) / 1024:.2f} KB)")

    if not texts:
        raise ValueError("No se pudo cargar ningún archivo de corpus válido.")

    print(f"Total corpus cargado: {total_bytes / (1024 * 1024):.2f} MB en {len(texts)} documentos.")
    return texts


def evaluate_compression(tok: TRACETokenizer, prompts: list[str], label: str) -> tuple[int, int]:
    print(f"\n--- {label} ---")
    total_raw = 0
    total_tok = 0
    for idx, prompt in enumerate(prompts, 1):
        raw_b = prompt.encode("utf-8")
        tokens = tok.encode(prompt)
        ratio = len(raw_b) / len(tokens) if tokens else 0.0
        total_raw += len(raw_b)
        total_tok += len(tokens)

        decoded = tok.decode(tokens)
        assert decoded == prompt, f"Fallo de round-trip en {label} [{idx}]"
        print(f"  [{idx}] {len(raw_b):3d} bytes -> {len(tokens):2d} tokens | {ratio:.2f} B/tok | '{prompt[:40]}...'")

    sub_ratio = total_raw / total_tok if total_tok else 0.0
    print(f"  Subtotal {label}: {sub_ratio:.2f} bytes/token ({1.0 / sub_ratio * 100:.1f}% ratio de longitud)")
    return total_raw, total_tok


def verify_c_parity(bin_path: str, tok: TRACETokenizer, prompts: list[str]) -> None:
    c_lib = load_c_tokenizer_lib()
    model_sz = c_lib.trace_tokenizer_sizeof()
    c_tok_buf = ctypes.create_string_buffer(model_sz)

    ret = c_lib.trace_tokenizer_load(c_tok_buf, bin_path.encode("utf-8"))
    assert ret == 0, f"Error cargando tokenizer recién entrenado en C: {ret}"

    out_c_buf = (ctypes.c_int32 * 2048)()
    out_text_buf = ctypes.create_string_buffer(8192)

    for p in prompts:
        raw_b = p.encode("utf-8")
        py_tokens = tok.encode(p)

        c_cnt = c_lib.trace_tokenizer_encode(c_tok_buf, raw_b, len(raw_b), out_c_buf, 2048)
        c_tokens = [out_c_buf[i] for i in range(c_cnt)]
        assert py_tokens == c_tokens, f"Discrepancia encode Py vs C en prompt: '{p[:30]}'"

        c_dec_len = c_lib.trace_tokenizer_decode(c_tok_buf, out_c_buf, c_cnt, out_text_buf, 8192)
        assert c_dec_len == len(raw_b)
        c_dec = out_text_buf.raw[:c_dec_len].decode("utf-8")
        assert c_dec == p, f"Discrepancia decode Py vs C en prompt: '{p[:30]}'"

    c_lib.trace_tokenizer_free(c_tok_buf)
    print("✓ Certificación de paridad 100% C vs Python superada sobre el binario generado.")


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del Tokenizador BPE 4096 para Basic-Lite TRACE"
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        required=True,
        help="Rutas a uno o más archivos de texto plano UTF-8 para el entrenamiento."
    )
    parser.add_argument(
        "--output",
        default="checkpoints/trace_tokenizer_4k.bin",
        help="Ruta de destino para el binario entrenado (default: checkpoints/trace_tokenizer_4k.bin)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=VOCAB_SIZE,
        help=f"Tamaño objetivo del vocabulario (default: {VOCAB_SIZE})"
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Frecuencia mínima requerida para autorizar una fusión de pares (default: 2)"
    )

    args = parser.parse_args()

    assert FIRST_MERGE < args.vocab_size <= VOCAB_SIZE, (
        f"vocab_size ({args.vocab_size}) debe estar en el rango ({FIRST_MERGE}, {VOCAB_SIZE}]"
    )

    print("=" * 80)
    print("Basic-Lite TRACE — Entrenamiento de Tokenizador BPE 4096")
    print(f"Vocabulario objetivo: {args.vocab_size} | Min freq: {args.min_freq}")
    print(f"Destino: {args.output}")
    print("=" * 80)

    # 1. Cargar documentos
    texts = load_corpus_files(args.corpus)

    # 2. Entrenar
    t0 = time.time()
    print("\nIniciando extracción y fusiones BPE deterministas...")
    tok = TRACETokenizer.train(
        texts=texts,
        target_vocab_size=args.vocab_size,
        min_freq=args.min_freq
    )
    dur = time.time() - t0

    num_merges = tok.num_merges
    target_merges = args.vocab_size - FIRST_MERGE
    print(f"\nEntrenamiento completado en {dur:.2f} segundos.")
    print(f"Merges alcanzados: {num_merges} / {target_merges} posibles ({(num_merges / target_merges) * 100:.1f}%)")

    # 3. Guardar binario
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    tok.save(args.output)
    bin_size_kb = os.path.getsize(args.output) / 1024
    print(f"Binario serializado exitosamente: {args.output} ({bin_size_kb:.2f} KB)")

    # 4. Auditoría de compresión: In-Domain vs General
    print("\n" + "=" * 80)
    print("REPORTE DE EFICIENCIA Y COMPRESIÓN (BYTES / TOKEN)")
    print("=" * 80)

    raw_in, tok_in = evaluate_compression(tok, IN_DOMAIN_PROMPTS, "PROMPTS IN-DOMAIN (TRACE)")
    raw_gen, tok_gen = evaluate_compression(tok, GENERAL_PROMPTS, "PROMPTS GENERALES FUERA DE DOMINIO")

    tot_raw = raw_in + raw_gen
    tot_tok = tok_in + tok_gen
    global_ratio = tot_raw / tot_tok if tot_tok else 0.0
    print("-" * 80)
    print(f"Tasa de compresión promedio global: {global_ratio:.2f} bytes/token ({1.0 / global_ratio * 100:.1f}% ratio de longitud)")
    print("=" * 80)

    # 5. Certificación de paridad nativa C recargando el binario
    print("\nCertificando paridad nativa en C recargando el archivo serializado...")
    verify_c_parity(args.output, tok, IN_DOMAIN_PROMPTS + GENERAL_PROMPTS)


if __name__ == "__main__":
    main()

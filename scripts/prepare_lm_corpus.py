#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Preparador de Corpus y Exportador Binario UINT16 para Basic-Lite TRACE.
Soporta:
1. Streaming / descarga desde Hugging Face (TinyStories, FineWeb-Edu).
2. Procesamiento de archivos de texto locales (--local-files).
3. Modo sintético (--synthetic) para desarrollo y smoke tests 100% offline.
4. Tokenización acelerada por C nativo (libtrace_tokenizer.so) con fallback en Python.
5. Inserción de SPECIAL_EOS (259) al final de cada documento.
6. Escritura en streaming por bloques a train.bin y val.bin (cero acumulación en RAM).
"""

import argparse
import ctypes
import os
import sys
from trace.lm.tokenizer import SPECIAL_EOS, TRACETokenizer

import numpy as np

SYNTHETIC_PROMPTS = [
    "La arquitectura Basic-Lite TRACE opera con cuantización ternaria W1.58A8 en CPU.\n",
    "El algoritmo de atención MLA comprime los estados KV hacia un cuello de botella latente.\n",
    "La memoria paginada Paged BitCOS v1.1 alinea cada bloque a 4096 bytes para I/O directa.\n",
    "Once upon a time in a digital forest, tiny algorithms learned to think with ternary weights.\n",
    "The CPU executes add-sub operations without floating-point multipliers, saving energy.\n",
    "En los sistemas modernos, el kernel gestiona páginas de memoria mediante llamadas pread.\n",
    "def train_step(model, x, y, optimizer):\n    optimizer.zero_grad()\n    loss = model(x, targets=y)[1]\n    loss.backward()\n    optimizer.step()\n    return loss\n",
    "#include <stdio.h>\nint main(void) {\n    printf(\"Basic-Lite TRACE 0.2\\n\");\n    return 0;\n}\n",
]


class FastTokenizerWrapper:
    """Envuelve TRACETokenizer con aceleración en C vía libtrace_tokenizer.so."""

    def __init__(self, tokenizer_path: str):
        self.tok_path = tokenizer_path
        self.py_tok = TRACETokenizer(tokenizer_path)
        self.c_lib = None
        self.c_buf = None
        self.c_max_out = 16384
        self.c_out_buf = None

        so_path = os.path.abspath("libtrace_tokenizer.so")
        if os.path.exists(so_path):
            try:
                c_lib = ctypes.CDLL(so_path)
                c_lib.trace_tokenizer_sizeof.restype = ctypes.c_size_t
                c_lib.trace_tokenizer_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                c_lib.trace_tokenizer_load.restype = ctypes.c_int
                c_lib.trace_tokenizer_free.argtypes = [ctypes.c_void_p]
                c_lib.trace_tokenizer_free.restype = None
                c_lib.trace_tokenizer_encode.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.c_int,
                ]
                c_lib.trace_tokenizer_encode.restype = ctypes.c_int

                buf = ctypes.create_string_buffer(c_lib.trace_tokenizer_sizeof())
                if c_lib.trace_tokenizer_load(buf, tokenizer_path.encode("utf-8")) == 0:
                    self.c_lib = c_lib
                    self.c_buf = buf
                    self.c_out_buf = (ctypes.c_int32 * self.c_max_out)()
                    print("✓ Motor de tokenización acelerado en C (libtrace_tokenizer.so) activo.")
            except Exception:
                self.c_lib = None
                self.c_buf = None

    def encode(self, text: str) -> list[int]:
        if self.c_lib is not None and self.c_buf is not None and self.c_out_buf is not None:
            raw_b = text.encode("utf-8")
            if len(raw_b) * 2 > self.c_max_out:
                self.c_max_out = max(self.c_max_out * 2, len(raw_b) * 2 + 1024)
                self.c_out_buf = (ctypes.c_int32 * self.c_max_out)()
            cnt = self.c_lib.trace_tokenizer_encode(
                self.c_buf, raw_b, len(raw_b), self.c_out_buf, self.c_max_out
            )
            if cnt >= 0:
                return [self.c_out_buf[i] for i in range(cnt)]
        return self.py_tok.encode(text)

    def close(self) -> None:
        if self.c_lib is not None and self.c_buf is not None:
            self.c_lib.trace_tokenizer_free(self.c_buf)
            self.c_lib = None
            self.c_buf = None


def generate_synthetic_corpus(target_tokens: int) -> list[str]:
    """Genera documentos sintéticos multilingües y técnicos repetidos hasta cubrir tokens aproximados."""
    docs = []
    accum = 0
    idx = 0
    while accum < target_tokens:
        text = SYNTHETIC_PROMPTS[idx % len(SYNTHETIC_PROMPTS)]
        docs.append(text)
        accum += len(text.split())
        idx += 1
    return docs


def load_from_local_files(paths: list[str]) -> list[str]:
    """Carga texto UTF-8 desde una lista de rutas locales."""
    docs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[Aviso] Archivo local no encontrado: {p}, omitiendo.")
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            docs.append(f.read())
    return docs


def iter_documents(args) -> list[str]:
    """Obtiene el conjunto o generador de documentos según la configuración."""
    if args.synthetic:
        print("[Modo Sintético] Generando documentos multilingües y técnicos...")
        return generate_synthetic_corpus(target_tokens=args.max_docs * 50)
    if args.local_files:
        print(f"[Modo Local] Cargando {len(args.local_files)} archivos locales...")
        return load_from_local_files(args.local_files)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: El paquete 'datasets' no está instalado en el entorno.")
        print("Ejecuta 'pip install datasets' o utiliza --local-files / --synthetic.")
        sys.exit(1)

    print(f"[Modo Hugging Face] Transmitiendo desde '{args.dataset}' (máx {args.max_docs} docs)...")
    ds = load_dataset(args.dataset, split="train", streaming=True)
    docs = []
    for idx, item in enumerate(ds):
        if idx >= args.max_docs:
            break
        text = item.get("text") or item.get("content") or item.get("story") or ""
        if text.strip():
            docs.append(text)
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Preparación de corpus y exportación binaria UINT16 para Basic-Lite TRACE."
    )
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Identificador del dataset en Hugging Face (default: roneneldan/TinyStories).")
    parser.add_argument("--local-files", nargs="+", default=None,
                        help="Rutas a archivos de texto locales para procesar offline.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generar corpus sintético determinista offline.")
    parser.add_argument("--max-docs", type=int, default=50000,
                        help="Número máximo de documentos a procesar (default: 50000).")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fracción de documentos reservados para validación (default: 0.05).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directorio de destino para train.bin y val.bin (default: data).")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/trace_tokenizer_4k.bin",
                        help="Ruta al binario del tokenizador BPE (default: checkpoints/trace_tokenizer_4k.bin).")
    parser.add_argument("--train-tokenizer", action="store_true",
                        help="Entrenar el tokenizador BPE si el archivo de salida no existe.")
    parser.add_argument("--target-vocab-size", type=int, default=4096,
                        help="Tamaño objetivo del vocabulario BPE (default: 4096).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok_dir = os.path.dirname(os.path.abspath(args.tokenizer))
    if tok_dir:
        os.makedirs(tok_dir, exist_ok=True)

    print("=" * 80)
    print("Basic-Lite TRACE — Preparación de Corpus de Lenguaje")
    print(f"Destino de datos:     {args.output_dir}")
    print(f"Tokenizador binario:  {args.tokenizer}")
    print("=" * 80)

    # 1. Obtención de documentos
    docs = iter_documents(args)
    if not docs:
        print("Error: No se obtuvieron documentos válidos.")
        sys.exit(1)

    total_chars = sum(len(d) for d in docs)
    print(f"Corpus cargado: {len(docs)} documentos ({total_chars / 1024 / 1024:.2f} MB de texto).")

    # 2. Entrenamiento opcional o carga del tokenizador
    if not os.path.exists(args.tokenizer):
        if args.train_tokenizer or args.synthetic:
            print(f"\nEntrenando nuevo tokenizador BPE en {args.tokenizer}...")
            sample_docs = docs[:5000]
            tok_trained = TRACETokenizer.train(sample_docs, target_vocab_size=args.target_vocab_size, min_freq=2)
            tok_trained.save(args.tokenizer)
            print("✓ Tokenizador entrenado y serializado exitosamente.")
        else:
            print(f"Error: No se encontró {args.tokenizer}. Entrénalo primero o usa --train-tokenizer.")
            sys.exit(1)

    tokenizer = FastTokenizerWrapper(args.tokenizer)

    # 3. Tokenización y serialización en streaming a disco (cero fuga de RAM)
    train_bin_path = os.path.join(args.output_dir, "train.bin")
    val_bin_path = os.path.join(args.output_dir, "val.bin")

    val_stride = max(2, int(1.0 / max(args.val_ratio, 1e-4)))

    train_tokens_count = 0
    val_tokens_count = 0
    train_buffer = []
    val_buffer = []
    chunk_flush_size = 65536

    with open(train_bin_path, "wb") as f_train, open(val_bin_path, "wb") as f_val:
        if len(docs) == 1:
            tokens = tokenizer.encode(docs[0])
            tokens.append(SPECIAL_EOS)
            val_count = max(1, int(len(tokens) * args.val_ratio))
            train_count = len(tokens) - val_count
            train_buffer.extend(tokens[:train_count])
            train_tokens_count += train_count
            val_buffer.extend(tokens[train_count:])
            val_tokens_count += val_count
        else:
            for idx, doc in enumerate(docs):
                tokens = tokenizer.encode(doc)
                tokens.append(SPECIAL_EOS)

                # Selección determinista por documento para streaming uniforme
                if (idx % val_stride) == (val_stride - 1):
                    val_buffer.extend(tokens)
                    val_tokens_count += len(tokens)
                else:
                    train_buffer.extend(tokens)
                    train_tokens_count += len(tokens)

                if len(train_buffer) >= chunk_flush_size:
                    np.array(train_buffer, dtype=np.uint16).tofile(f_train)
                    train_buffer.clear()

                if len(val_buffer) >= chunk_flush_size:
                    np.array(val_buffer, dtype=np.uint16).tofile(f_val)
                    val_buffer.clear()

            if val_tokens_count == 0 and len(train_buffer) > 0:
                val_slice = max(1, int(len(train_buffer) * args.val_ratio))
                val_buffer.extend(train_buffer[-val_slice:])
                val_tokens_count += val_slice
                train_buffer = train_buffer[:-val_slice]
                train_tokens_count -= val_slice

        # Vaciar buffers residuales
        if train_buffer:
            np.array(train_buffer, dtype=np.uint16).tofile(f_train)
            train_buffer.clear()
        if val_buffer:
            np.array(val_buffer, dtype=np.uint16).tofile(f_val)
            val_buffer.clear()

    tokenizer.close()

    total_tokens = train_tokens_count + val_tokens_count
    b_per_tok = (total_chars / total_tokens) if total_tokens > 0 else 0.0

    print("\n" + "=" * 80)
    print("✓ Preparación finalizada exitosamente:")
    print(f"  - Documentos procesados: {len(docs):,}")
    print(f"  - Tokens totales:        {total_tokens:,} ({b_per_tok:.2f} bytes/token)")
    print(f"  - Train: {train_bin_path} ({train_tokens_count:,} tokens, {os.path.getsize(train_bin_path) / 1024 / 1024:.2f} MB)")
    print(f"  - Val:   {val_bin_path} ({val_tokens_count:,} tokens, {os.path.getsize(val_bin_path) / 1024 / 1024:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()

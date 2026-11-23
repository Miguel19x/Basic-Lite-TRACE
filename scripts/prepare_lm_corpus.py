#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Preparador de Corpus y Exportador Binario UINT16 para Basic-Lite TRACE.
Soporta:
1. Streaming / descarga desde Hugging Face con interleaving ponderado (--dataset-mix / --dataset).
2. Procesamiento de archivos de texto locales (--local-files).
3. Modo sintético (--synthetic) para desarrollo y smoke tests 100% offline.
4. Muestreo estratificado (--tokenizer-sample-mb) para entrenamiento de BPE (4096 u 8192 tokens).
5. Filtro de higiene estricto (alpha/non_space >= 0.70 y len >= 80).
6. Tokenización acelerada por C nativo (libtrace_tokenizer.so) con fallback en Python.
7. Inserción de SPECIAL_EOS (259) al final de cada documento.
8. Escritura en streaming por bloques a train.bin y val.bin (cero acumulación en RAM).
"""

import argparse
import ctypes
import os
import random
import sys
from collections.abc import Generator
from trace.lm.tokenizer import SPECIAL_EOS, TRACETokenizer

import numpy as np

SYNTHETIC_PROMPTS = [
    "La arquitectura neuronal Basic-Lite TRACE opera con cuantización ternaria W1.58A8 sobre arquitecturas x86_64 y ARM64.\n",
    "El algoritmo de atención MLA comprime los estados clave y valor hacia un cuello de botella latente multidimensional.\n",
    "La memoria paginada Paged BitCOS v1.1 alinea cada bloque a 4096 bytes para permitir transferencias mediante lectura directa del kernel.\n",
    "Once upon a time in a digital forest, tiny algorithms learned to think deeply with ternary weights and low-rank latent projections.\n",
    "The CPU executes add-sub operations without floating-point multipliers, saving electrical energy and memory bandwidth.\n",
    "En los sistemas operativos modernos, el kernel gestiona páginas físicas de memoria mediante llamadas pread concurrentes y seguras.\n",
    "El entrenamiento bilingüe de modelos de lenguaje ligeros aprovecha la transferencia translingüística entre inglés y español formal.\n",
    "Cada capa del transformador aplica normalización cuadrática media sin sesgo para preservar la estabilidad numérica a gran escala.\n",
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


def passes_hygiene_filter(text: str) -> bool:
    """Aplica filtro de calidad e higiene sintáctica."""
    if len(text) < 80:
        return False
    non_space = sum(not c.isspace() for c in text)
    if non_space == 0:
        return False
    alpha = sum(c.isalpha() for c in text)
    return (alpha / non_space) >= 0.70


def parse_dataset_mix(mix_str: str) -> list[tuple[str, str | None, float]]:
    """Parsea cadena repo:config:weight,... normalizando la suma de pesos a 1.0."""
    entries = []
    total_w = 0.0
    for part in mix_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        if len(tokens) == 3:
            repo, config, weight_str = tokens
            config = config.strip() if config.strip() else None
        elif len(tokens) == 2:
            repo, weight_str = tokens
            config = None
        elif len(tokens) == 1:
            repo = tokens[0]
            config = None
            weight_str = "1.0"
        else:
            raise ValueError(f"Formato de dataset inválido: {part}")
        weight = float(weight_str)
        entries.append((repo.strip(), config, weight))
        total_w += weight

    if total_w <= 0:
        raise ValueError("La suma de pesos debe ser positiva.")
    return [(r, c, w / total_w) for r, c, w in entries]


def make_dataset_stream(repo: str, config: str | None) -> Generator[str, None, None]:
    """Generador streaming de documentos para un dataset individual de Hugging Face."""
    from datasets import load_dataset
    if config:
        ds = load_dataset(repo, name=config, split="train", streaming=True)
    else:
        ds = load_dataset(repo, split="train", streaming=True)

    for item in ds:
        text = item.get("text") or item.get("content") or item.get("story") or ""
        if text and passes_hygiene_filter(text):
            yield text


def iter_streaming_mix(mix_entries: list[tuple[str, str | None, float]], seed: int = 42) -> Generator[tuple[int, str], None, None]:
    """Interleaving ponderado en streaming entre múltiples generadores de datasets."""
    rng = random.Random(seed)
    streams = [make_dataset_stream(repo, cfg) for repo, cfg, _ in mix_entries]
    weights = [w for _, _, w in mix_entries]
    active_indices = list(range(len(streams)))

    while active_indices:
        active_weights = [weights[i] for i in active_indices]
        choice_idx = rng.choices(active_indices, weights=active_weights, k=1)[0]
        try:
            doc = next(streams[choice_idx])
            yield (choice_idx, doc)
        except StopIteration:
            active_indices.remove(choice_idx)


def sample_text_for_tokenizer(args, target_bytes: int) -> list[str]:
    """Extrae un muestreo estratificado de texto plano para entrenar BPE."""
    sample_docs = []
    collected_bytes = 0

    if args.synthetic:
        for idx in range(100):
            p = SYNTHETIC_PROMPTS[idx % len(SYNTHETIC_PROMPTS)]
            sample_docs.append(p)
            collected_bytes += len(p.encode("utf-8"))
            if collected_bytes >= target_bytes:
                break
        return sample_docs

    if args.local_files:
        for p in args.local_files:
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                content = f.read()
                if passes_hygiene_filter(content):
                    sample_docs.append(content)
                    collected_bytes += len(content.encode("utf-8"))
            if collected_bytes >= target_bytes:
                break
        return sample_docs

    mix_entries = parse_dataset_mix(args.dataset_mix) if args.dataset_mix else [(args.dataset, None, 1.0)]
    stream = iter_streaming_mix(mix_entries, seed=1337)
    for _, doc in stream:
        sample_docs.append(doc)
        collected_bytes += len(doc.encode("utf-8"))
        if collected_bytes >= target_bytes:
            break
    return sample_docs


def main():
    parser = argparse.ArgumentParser(
        description="Preparación de corpus y exportación binaria UINT16 para Basic-Lite TRACE."
    )
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset Hugging Face único (default: roneneldan/TinyStories).")
    parser.add_argument("--dataset-mix", type=str, default=None,
                        help="Mezcla ponderada de datasets repo:config:weight,... (ej: 'roneneldan/TinyStories::0.40,...').")
    parser.add_argument("--local-files", nargs="+", default=None,
                        help="Rutas a archivos de texto locales para procesar offline.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generar corpus sintético determinista offline.")
    parser.add_argument("--max-docs", type=int, default=50000,
                        help="Número máximo de documentos a procesar (default: 50000).")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fracción de documentos reservados para validación (default: 0.05).")
    parser.add_argument("--tokenizer-sample-mb", type=float, default=50.0,
                        help="Megabytes de texto plano a muestrear para entrenar el BPE (default: 50.0).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directorio de destino para train.bin y val.bin (default: data).")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/trace_tokenizer_4k.bin",
                        help="Ruta al binario del tokenizador BPE (default: checkpoints/trace_tokenizer_4k.bin).")
    parser.add_argument("--train-tokenizer", action="store_true",
                        help="Entrenar el tokenizador BPE si el archivo de salida no existe.")
    parser.add_argument("--target-vocab-size", type=int, default=4096,
                        help="Tamaño objetivo del vocabulario BPE (4096 monolingüe, 8192 bilingüe).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok_dir = os.path.dirname(os.path.abspath(args.tokenizer))
    if tok_dir:
        os.makedirs(tok_dir, exist_ok=True)

    print("=" * 80)
    print("Basic-Lite TRACE — Preparación de Corpus de Lenguaje")
    print(f"Destino de datos:     {args.output_dir}")
    print(f"Tokenizador binario:  {args.tokenizer} (Vocab objetivo: {args.target_vocab_size})")
    if args.dataset_mix:
        print(f"Mezcla de datasets:   {args.dataset_mix}")
    elif args.synthetic:
        print("Modo de operación:    Sintético determinista")
    elif args.local_files:
        print(f"Modo de operación:    Archivos locales ({len(args.local_files)} archivos)")
    else:
        print(f"Dataset único HF:     {args.dataset}")
    print("=" * 80)

    # 1. Entrenamiento opcional o carga del tokenizador
    if not os.path.exists(args.tokenizer):
        if args.train_tokenizer or args.synthetic:
            target_bytes = int(args.tokenizer_sample_mb * 1024 * 1024) if not args.synthetic else 50000
            print(f"\nMuestreando ~{target_bytes / 1024 / 1024:.2f} MB de texto para entrenar tokenizador...")
            sample_docs = sample_text_for_tokenizer(args, target_bytes)
            print(f"Documentos muestreados: {len(sample_docs)}. Entrenando BPE (V={args.target_vocab_size})...")
            tok_trained = TRACETokenizer.train(
                sample_docs,
                target_vocab_size=args.target_vocab_size,
                min_freq=2,
            )
            tok_trained.save(args.tokenizer)
            print(f"✓ Tokenizador serializado exitosamente en {args.tokenizer}.")
        else:
            print(f"Error: No se encontró {args.tokenizer}. Entrénalo primero o usa --train-tokenizer.")
            sys.exit(1)

    tokenizer = FastTokenizerWrapper(args.tokenizer)

    # 2. Configuración de partición de validación balanceada
    val_stride = max(2, int(round(1.0 / max(args.val_ratio, 1e-4))))
    train_bin_path = os.path.join(args.output_dir, "train.bin")
    val_bin_path = os.path.join(args.output_dir, "val.bin")

    train_tokens_count = 0
    val_tokens_count = 0
    train_buffer = []
    val_buffer = []
    chunk_flush_size = 65536
    doc_count = 0
    total_chars = 0
    source_counts: dict[int, int] = {}

    def get_document_generator() -> Generator[tuple[int, str], None, None]:
        if args.synthetic:
            for idx in range(args.max_docs):
                p = SYNTHETIC_PROMPTS[idx % len(SYNTHETIC_PROMPTS)]
                yield (0, p)
            return

        if args.local_files:
            s_idx = 0
            for p in args.local_files:
                if not os.path.exists(p):
                    continue
                with open(p, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    if passes_hygiene_filter(content):
                        yield (s_idx, content)
                        s_idx += 1
            return

        mix_entries = parse_dataset_mix(args.dataset_mix) if args.dataset_mix else [(args.dataset, None, 1.0)]
        yield from iter_streaming_mix(mix_entries, seed=42)

    # 3. Tokenización y serialización en streaming a disco (cero acumulación en RAM)
    with open(train_bin_path, "wb") as f_train, open(val_bin_path, "wb") as f_val:
        for src_id, doc in get_document_generator():
            if doc_count >= args.max_docs:
                break

            doc_count += 1
            total_chars += len(doc)
            src_seen = source_counts.get(src_id, 0)
            source_counts[src_id] = src_seen + 1

            tokens = tokenizer.encode(doc)
            tokens.append(SPECIAL_EOS)

            # Selección estratificada balanceada por fuente
            if (src_seen % val_stride) == (val_stride - 1):
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

        # Balanceo si la validación quedó vacía en corpus muy cortos
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
    print(f"  - Documentos procesados: {doc_count:,}")
    print(f"  - Tokens totales:        {total_tokens:,} ({b_per_tok:.2f} bytes/token)")
    print(f"  - Train: {train_bin_path} ({train_tokens_count:,} tokens, {os.path.getsize(train_bin_path) / 1024 / 1024:.2f} MB)")
    print(f"  - Val:   {val_bin_path} ({val_tokens_count:,} tokens, {os.path.getsize(val_bin_path) / 1024 / 1024:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    main()

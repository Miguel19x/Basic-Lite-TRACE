#!/usr/bin/env python3
# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Harness formal de Fase 4A.4: Perplejidad (PPL) y Benchmark de Inferencia 2B
sobre el motor nativo C (`libtrace_lm.so` + `libtrace_tokenizer.so`) en 2 hilos físicos.

Mide:
  1. Tiempo de carga, O_DIRECT, pread_calls, tipo de atención y dtype de embeddings.
  2. Perplejidad exacta por ventanas/bloques (con comparación opcional ±1% vs referencia `bitnet.cpp`).
  3. TTFT de cold prefill (ms y tok/s), decodificación autorregresiva (tok/s, mediana y p95 ms/tok) y RSS pico.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_native_libs() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    lm_so = REPO_ROOT / "libtrace_lm.so"
    tok_so = REPO_ROOT / "libtrace_tokenizer.so"
    if not lm_so.exists() or not tok_so.exists():
        raise FileNotFoundError("Faltan libtrace_lm.so o libtrace_tokenizer.so. Compila con 'make all'.")

    lm = ctypes.CDLL(str(lm_so))
    lm.trace_lm_load.argtypes = [ctypes.c_char_p]
    lm.trace_lm_load.restype = ctypes.c_void_p
    lm.trace_lm_free.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_config.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_config.restype = ctypes.c_void_p
    lm.trace_lm_get_vocab_size.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_vocab_size.restype = ctypes.c_int
    lm.trace_lm_get_pread_calls.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_pread_calls.restype = ctypes.c_int
    lm.trace_lm_get_used_odirect.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_used_odirect.restype = ctypes.c_int
    lm.trace_lm_get_tok_emb_dtype.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_tok_emb_dtype.restype = ctypes.c_int
    lm.trace_lm_get_attn_type.argtypes = [ctypes.c_void_p]
    lm.trace_lm_get_attn_type.restype = ctypes.c_int
    lm.trace_lm_workspace_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lm.trace_lm_workspace_create.restype = ctypes.c_void_p
    lm.trace_lm_workspace_free.argtypes = [ctypes.c_void_p]
    lm.trace_lm_kv_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lm.trace_lm_kv_create.restype = ctypes.c_void_p
    lm.trace_lm_kv_free.argtypes = [ctypes.c_void_p]
    lm.trace_lm_kv_reset.argtypes = [ctypes.c_void_p]
    lm.trace_lm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lm.trace_lm_forward.restype = ctypes.c_int
    lm.trace_lm_sample_greedy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lm.trace_lm_sample_greedy.restype = ctypes.c_int
    lm.trace_cpu_active_isa.argtypes = []
    lm.trace_cpu_active_isa.restype = ctypes.c_char_p

    tok = ctypes.CDLL(str(tok_so))
    tok.trace_tokenizer_sizeof.restype = ctypes.c_size_t
    tok.trace_tokenizer_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    tok.trace_tokenizer_load.restype = ctypes.c_int
    tok.trace_tokenizer_free.argtypes = [ctypes.c_void_p]
    tok.trace_tokenizer_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    tok.trace_tokenizer_encode.restype = ctypes.c_int
    return lm, tok


def compute_token_nll(logits_row: np.ndarray, target_id: int) -> float:
    """Negative log-likelihood numéricamente estable en FP64 para un vector de logits 1D."""
    x = logits_row.astype(np.float64, copy=False)
    m = float(np.max(x))
    log_z = m + float(np.log(np.sum(np.exp(x - m))))
    return float(log_z - float(x[target_id]))


def evaluate_perplexity_c(
    lm: ctypes.CDLL,
    c_model: ctypes.c_void_p,
    token_ids: list[int],
    context_len: int = 512,
    chunk_size: int = 32,
    use_int8_kv: int = 1,
) -> dict[str, Any]:
    """
    Evalúa perplejidad exacta (sin solapamiento entre ventanas de `context_len`, procesando
    en bloques de `chunk_size` para acotar la RAM del buffer de logits en vocabularios 128K).
    """
    if len(token_ids) < 2:
        raise ValueError("Se requieren al menos 2 tokens para calcular perplejidad.")

    vocab_size = int(lm.trace_lm_get_vocab_size(c_model))
    c_cfg = lm.trace_lm_get_config(c_model)
    eff_chunk = max(1, min(chunk_size, context_len))

    c_ws = lm.trace_lm_workspace_create(c_cfg, eff_chunk)
    c_kv = lm.trace_lm_kv_create(c_cfg, context_len, use_int8_kv)
    if not c_ws or not c_kv:
        if c_kv:
            lm.trace_lm_kv_free(c_kv)
        if c_ws:
            lm.trace_lm_workspace_free(c_ws)
        raise RuntimeError("No se pudo crear workspace/KV para evaluación de perplejidad.")

    logits_buf = (ctypes.c_float * (eff_chunk * vocab_size))()
    total_nll = 0.0
    total_pred = 0
    t0 = time.perf_counter()

    try:
        for win_start in range(0, len(token_ids) - 1, context_len):
            win_tokens = token_ids[win_start : min(win_start + context_len, len(token_ids))]
            if len(win_tokens) < 2:
                break
            lm.trace_lm_kv_reset(c_kv)

            # Evaluamos prefijos incrementales en bloques de eff_chunk tokens
            win_len = len(win_tokens)
            for pos in range(0, win_len - 1, eff_chunk):
                end = min(pos + eff_chunk, win_len - 1)
                sub = win_tokens[:end]
                c_sub = (ctypes.c_int * end)(*sub)
                n_proc = int(lm.trace_lm_forward(c_model, c_kv, c_sub, end, logits_buf, 0, c_ws))
                if n_proc <= 0:
                    raise RuntimeError(f"trace_lm_forward falló en PPL (pos={pos}, end={end}, rc={n_proc})")
                logits_mat = np.ctypeslib.as_array(logits_buf)[: n_proc * vocab_size].reshape(n_proc, vocab_size)
                # Los n_proc logits corresponden a las posiciones [end - n_proc .. end - 1], prediciendo +1
                base_idx = end - n_proc
                for k in range(n_proc):
                    target = win_tokens[base_idx + k + 1]
                    total_nll += compute_token_nll(logits_mat[k], target)
                    total_pred += 1
    finally:
        lm.trace_lm_kv_free(c_kv)
        lm.trace_lm_workspace_free(c_ws)

    elapsed = time.perf_counter() - t0
    mean_nll = total_nll / max(total_pred, 1)
    ppl = float(math.exp(mean_nll)) if mean_nll < 700.0 else float("inf")
    return {
        "perplexity": ppl,
        "mean_nll": mean_nll,
        "tokens_evaluated": total_pred,
        "elapsed_sec": elapsed,
        "context_len": context_len,
        "use_int8_kv": use_int8_kv,
    }


def check_ppl_exit_gate(ppl: float, ref_ppl: float | None, max_rel_diff: float = 0.01) -> dict[str, Any]:
    """Verifica la puerta de salida de Fase 4A.4: perplejidad dentro de ±1% frente a la referencia."""
    if ref_ppl is None or ref_ppl <= 0.0 or not math.isfinite(ref_ppl):
        return {"reference_ppl": ref_ppl, "rel_diff": None, "passed": None}
    rel_diff = abs(ppl - ref_ppl) / ref_ppl
    return {
        "reference_ppl": ref_ppl,
        "rel_diff": rel_diff,
        "rel_diff_pct": rel_diff * 100.0,
        "max_allowed_pct": max_rel_diff * 100.0,
        "passed": bool(rel_diff <= max_rel_diff),
    }


def benchmark_latency_c(
    lm: ctypes.CDLL,
    c_model: ctypes.c_void_p,
    prefill_lengths: list[int],
    decode_steps: int = 32,
    chunk_size: int = 128,
    use_int8_kv: int = 1,
) -> dict[str, Any]:
    """Mide cold prefill TTFT y decodificación autorregresiva S=1 en el motor C."""
    vocab_size = int(lm.trace_lm_get_vocab_size(c_model))
    c_cfg = lm.trace_lm_get_config(c_model)
    max_ctx = max(prefill_lengths) + decode_steps + 4

    c_ws = lm.trace_lm_workspace_create(c_cfg, chunk_size)
    c_kv = lm.trace_lm_kv_create(c_cfg, max_ctx, use_int8_kv)
    if not c_ws or not c_kv:
        if c_kv:
            lm.trace_lm_kv_free(c_kv)
        if c_ws:
            lm.trace_lm_workspace_free(c_ws)
        raise RuntimeError("No se pudo crear workspace/KV para benchmark de latencia.")

    logits_1 = (ctypes.c_float * vocab_size)()
    results: dict[str, Any] = {}
    try:
        for length in prefill_lengths:
            lm.trace_lm_kv_reset(c_kv)
            prompt = [(i * 29 + 11) % vocab_size for i in range(length)]

            t0 = time.perf_counter()
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                c_toks = (ctypes.c_int * end)(*prompt[:end])
                rc = int(lm.trace_lm_forward(c_model, c_kv, c_toks, end, logits_1, 1, c_ws))
                if rc < 0:
                    raise RuntimeError(f"trace_lm_forward falló en prefill L={length} (rc={rc})")
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            seq = list(prompt)
            next_tok = int(lm.trace_lm_sample_greedy(logits_1, vocab_size))
            step_ms: list[float] = []
            for _ in range(decode_steps):
                seq.append(next_tok)
                c_seq = (ctypes.c_int * len(seq))(*seq)
                ts = time.perf_counter()
                rc = int(lm.trace_lm_forward(c_model, c_kv, c_seq, len(seq), logits_1, 1, c_ws))
                step_ms.append((time.perf_counter() - ts) * 1000.0)
                if rc < 0:
                    raise RuntimeError(f"trace_lm_forward falló en decode L={length} (rc={rc})")
                next_tok = int(lm.trace_lm_sample_greedy(logits_1, vocab_size))

            arr_ms = np.array(step_ms, dtype=np.float64)
            total_dec_s = float(np.sum(arr_ms)) / 1000.0
            results[f"L={length}"] = {
                "ttft_ms": ttft_ms,
                "prefill_tok_per_s": float(length / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0.0,
                "decode_tok_per_s": float(decode_steps / total_dec_s) if total_dec_s > 0 else 0.0,
                "decode_median_ms": float(np.median(arr_ms)),
                "decode_p95_ms": float(np.percentile(arr_ms, 95)),
            }
    finally:
        lm.trace_lm_kv_free(c_kv)
        lm.trace_lm_workspace_free(c_ws)
    return results


def encode_text_file(tok_lib: ctypes.CDLL, tokenizer_path: str, text_path: str, max_tokens: int) -> list[int]:
    tok_buf = ctypes.create_string_buffer(tok_lib.trace_tokenizer_sizeof())
    rc = int(tok_lib.trace_tokenizer_load(tok_buf, tokenizer_path.encode()))
    if rc != 0:
        raise RuntimeError(f"No se pudo cargar tokenizador {tokenizer_path} (rc={rc})")
    try:
        raw = Path(text_path).read_bytes()
        out = (ctypes.c_int32 * max_tokens)()
        n = int(tok_lib.trace_tokenizer_encode(tok_buf, raw, len(raw), out, max_tokens))
        return [int(out[i]) for i in range(n)]
    finally:
        tok_lib.trace_tokenizer_free(tok_buf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 2B y evaluación de Perplejidad (Fase 4A.4).")
    parser.add_argument("--model", required=True, help="Ruta al binario Paged BitCOS v1.1 (.bin)")
    parser.add_argument("--tokenizer", default=None, help="Ruta al tokenizador TRTK v2 (.bin)")
    parser.add_argument("--eval-text", default=None, help="Archivo UTF-8 para evaluar perplejidad (ej. wikitext2)")
    parser.add_argument("--max-eval-tokens", type=int, default=2048, help="Límite de tokens a evaluar en PPL")
    parser.add_argument("--ppl-context", type=int, default=512, help="Ventana de contexto para PPL")
    parser.add_argument("--prefill-lengths", default="128,512", help="Longitudes de prefill separadas por coma")
    parser.add_argument("--decode-steps", type=int, default=32, help="Tokens generados en benchmark de decode")
    parser.add_argument("--int8-kv", type=int, choices=(0, 1), default=1, help="1=INT8 KV, 0=FP32 KV")
    parser.add_argument("--ref-ppl", type=float, default=None, help="Perplejidad de referencia (bitnet.cpp) para gate ±1%%")
    parser.add_argument("--output", required=True, help="Ruta JSON de salida (preferir /tmp/ en pruebas)")
    args = parser.parse_args()

    lm, tok = load_native_libs()
    t_load0 = time.perf_counter()
    c_model = lm.trace_lm_load(args.model.encode())
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    if not c_model:
        raise RuntimeError(f"Error al cargar modelo Paged BitCOS: {args.model}")

    emb_names = {0: "float32", 1: "float16", 2: "int8"}
    attn_names = {0: "mla", 1: "gqa"}
    report: dict[str, Any] = {
        "model_path": args.model,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "active_isa": lm.trace_cpu_active_isa().decode(),
        "load_ms": load_ms,
        "used_odirect": int(lm.trace_lm_get_used_odirect(c_model)),
        "pread_calls": int(lm.trace_lm_get_pread_calls(c_model)),
        "vocab_size": int(lm.trace_lm_get_vocab_size(c_model)),
        "attn_type": attn_names.get(int(lm.trace_lm_get_attn_type(c_model)), "unknown"),
        "embedding_dtype": emb_names.get(int(lm.trace_lm_get_tok_emb_dtype(c_model)), "unknown"),
        "use_int8_kv": args.int8_kv,
    }

    try:
        prefill_lens = [int(x.strip()) for x in args.prefill_lengths.split(",") if x.strip()]
        report["latency"] = benchmark_latency_c(
            lm, c_model, prefill_lengths=prefill_lens, decode_steps=args.decode_steps, use_int8_kv=args.int8_kv
        )

        if args.eval_text and args.tokenizer:
            tokens = encode_text_file(tok, args.tokenizer, args.eval_text, args.max_eval_tokens)
            ppl_res = evaluate_perplexity_c(
                lm, c_model, tokens, context_len=args.ppl_context, use_int8_kv=args.int8_kv
            )
            ppl_res["exit_gate"] = check_ppl_exit_gate(ppl_res["perplexity"], args.ref_ppl)
            report["perplexity"] = ppl_res
    finally:
        lm.trace_lm_free(c_model)

    report["peak_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

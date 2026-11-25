#!/usr/bin/env python3
# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Runner CLI de Fase 4B: Banco de Evaluación de 4 Dominios + Matriz Simétrica C1-C3
sobre el motor nativo C (`libtrace_lm.so` + `libtrace_tokenizer.so` + `trace_json_grammar_*`).

Uso típico (bajo autorización de Nivel 1 y OMP_NUM_THREADS=2):
  OMP_NUM_THREADS=2 PYTHONPATH=. python3 scripts/run_bench_4b.py \
      --model checkpoints/bitnet_2b_fp16.bin \
      --tokenizer checkpoints/bitnet_2b_tok_v2.bin \
      --max-new-tokens 48 \
      --json-out /tmp/trace_4b_bench/report_c1_c3.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from trace.eval.bench_4domains import GenerationResult, evaluate_4domains

REPO_ROOT = Path(__file__).resolve().parent.parent


class NativeTraceLMBackend:
    """Backend nativo C sobre libtrace_lm.so + libtrace_tokenizer.so con máscara JSON opcional."""

    def __init__(self, model_path: str, tokenizer_path: str, max_ctx: int = 1024):
        lm_so = REPO_ROOT / "libtrace_lm.so"
        tok_so = REPO_ROOT / "libtrace_tokenizer.so"
        if not lm_so.exists() or not tok_so.exists():
            raise FileNotFoundError("Faltan libtrace_lm.so o libtrace_tokenizer.so. Compila con 'make all'.")

        self.lm = ctypes.CDLL(str(lm_so))
        self.tok = ctypes.CDLL(str(tok_so))
        self._bind_symbols()

        self.c_model = self.lm.trace_lm_load(model_path.encode("utf-8"))
        if not self.c_model:
            raise RuntimeError(f"Error cargando modelo C: {model_path}")

        self.vocab_size = int(self.lm.trace_lm_get_vocab_size(self.c_model))
        self.c_cfg = self.lm.trace_lm_get_config(self.c_model)
        self.max_ctx = int(max_ctx)
        self.chunk_size = 128
        self.ws = self.lm.trace_lm_workspace_create(self.c_cfg, self.chunk_size)
        self.kv = self.lm.trace_lm_kv_create(self.c_cfg, self.max_ctx, 1)

        tok_sz = int(self.tok.trace_tokenizer_sizeof())
        self.tok_buf = ctypes.create_string_buffer(tok_sz)
        rc = int(self.tok.trace_tokenizer_load(self.tok_buf, tokenizer_path.encode("utf-8")))
        if rc != 0:
            self.close()
            raise RuntimeError(f"Error cargando tokenizador C ({rc}): {tokenizer_path}")

        self.gram_buf = ctypes.create_string_buffer(int(self.tok.trace_json_grammar_sizeof()))
        self.logits = np.empty(self.vocab_size, dtype=np.float32)

    def _bind_symbols(self) -> None:
        lm = self.lm
        lm.trace_lm_load.argtypes = [ctypes.c_char_p]
        lm.trace_lm_load.restype = ctypes.c_void_p
        lm.trace_lm_free.argtypes = [ctypes.c_void_p]
        lm.trace_lm_get_config.argtypes = [ctypes.c_void_p]
        lm.trace_lm_get_config.restype = ctypes.c_void_p
        lm.trace_lm_get_vocab_size.argtypes = [ctypes.c_void_p]
        lm.trace_lm_get_vocab_size.restype = ctypes.c_int
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

        tok = self.tok
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
        tok.trace_tokenizer_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        tok.trace_tokenizer_decode.restype = ctypes.c_int
        tok.trace_tokenizer_format_chat_llama3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        tok.trace_tokenizer_format_chat_llama3.restype = ctypes.c_int

        tok.trace_json_grammar_sizeof.restype = ctypes.c_size_t
        tok.trace_json_grammar_init.argtypes = [ctypes.c_void_p]
        tok.trace_json_grammar_is_done.argtypes = [ctypes.c_void_p]
        tok.trace_json_grammar_is_done.restype = ctypes.c_int
        tok.trace_json_grammar_accept_token.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        tok.trace_json_grammar_accept_token.restype = ctypes.c_int
        tok.trace_json_grammar_mask_logits.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        tok.trace_json_grammar_mask_logits.restype = ctypes.c_int

    def close(self) -> None:
        if getattr(self, "tok_buf", None) is not None:
            self.tok.trace_tokenizer_free(self.tok_buf)
            self.tok_buf = None
        if getattr(self, "kv", None):
            self.lm.trace_lm_kv_free(self.kv)
            self.kv = None
        if getattr(self, "ws", None):
            self.lm.trace_lm_workspace_free(self.ws)
            self.ws = None
        if getattr(self, "c_model", None):
            self.lm.trace_lm_free(self.c_model)
            self.c_model = None

    def _encode_chat(self, prompt: str) -> list[int]:
        out_arr = (ctypes.c_int32 * (self.max_ctx - 64))()
        sys_msg = b"Eres un asistente tecnico conciso y preciso. Responde solo lo solicitado."
        n = int(
            self.tok.trace_tokenizer_format_chat_llama3(
                self.tok_buf,
                sys_msg,
                prompt.encode("utf-8"),
                1,
                out_arr,
                len(out_arr),
            )
        )
        if n <= 0:
            n = int(
                self.tok.trace_tokenizer_encode(
                    self.tok_buf,
                    prompt.encode("utf-8"),
                    1,
                    out_arr,
                    len(out_arr),
                )
            )
        return [int(out_arr[i]) for i in range(max(0, n))]

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        arr = (ctypes.c_int32 * len(ids))(*ids)
        out = ctypes.create_string_buffer(max(256, len(ids) * 16))
        n = int(self.tok.trace_tokenizer_decode(self.tok_buf, arr, len(ids), out, len(out)))
        if n <= 0:
            return ""
        return out.raw[:n].decode("utf-8", errors="replace")

    @staticmethod
    def _softmax_max_prob(logits: np.ndarray, chosen_id: int) -> float:
        finite = logits[np.isfinite(logits)]
        if finite.size == 0:
            return 0.0
        m = float(np.max(finite))
        probs_sum = float(np.sum(np.exp(finite.astype(np.float64) - m)))
        if probs_sum <= 0.0 or not np.isfinite(logits[chosen_id]):
            return 0.0
        return float(np.exp(float(logits[chosen_id]) - m) / probs_sum)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        json_constrained: bool = False,
    ) -> GenerationResult:
        self.lm.trace_lm_kv_reset(self.kv)
        prompt_ids = self._encode_chat(prompt)
        if not prompt_ids:
            return GenerationResult(text="", tokens_generated=0)

        max_prompt = max(1, self.max_ctx - max_new_tokens - 4)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        # Prefill por chunks de self.chunk_size con Prefix Caching (c_kv es arg 2, c_ws es arg 7)
        logits_ptr = self.logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for start in range(0, len(prompt_ids), self.chunk_size):
            end = min(start + self.chunk_size, len(prompt_ids))
            in_arr = (ctypes.c_int * end)(*prompt_ids[:end])
            rc = int(self.lm.trace_lm_forward(self.c_model, self.kv, in_arr, end, logits_ptr, 1, self.ws))
            if rc < 0:
                raise RuntimeError(f"trace_lm_forward falló en prefill (end={end}, rc={rc})")

        if json_constrained:
            self.tok.trace_json_grammar_init(self.gram_buf)

        seq = list(prompt_ids)
        gen_ids: list[int] = []
        tok_confs: list[float] = []
        # En TRTK v2 (128K LLaMA-3), ID 2 es '#' (NO EOS); parar solo en EOS/EOT reales del vocabulario
        stop_ids = {128001, 128009} if self.vocab_size > 100000 else {259}
        # Modo adaptativo para C3: si el modelo inicia con '<think>', permitir razonamiento
        # libre hasta emitir '<tool_call>' y activar entonces la máscara JSON PDA en C.
        grammar_active = False

        for step in range(max_new_tokens):
            if json_constrained:
                if grammar_active:
                    if int(self.tok.trace_json_grammar_is_done(self.gram_buf)):
                        break
                    allowed = int(self.tok.trace_json_grammar_mask_logits(self.gram_buf, self.tok_buf, logits_ptr))
                    if allowed <= 0:
                        break
                elif step == 0:
                    # Si el argmax natural no empieza por '<', activar máscara JSON estricta desde token 0
                    raw_top = int(self.lm.trace_lm_sample_greedy(logits_ptr, self.vocab_size))
                    raw_piece = self._decode([raw_top]) if raw_top >= 0 else ""
                    if not raw_piece.lstrip().startswith("<"):
                        grammar_active = True
                        allowed = int(self.tok.trace_json_grammar_mask_logits(self.gram_buf, self.tok_buf, logits_ptr))
                        if allowed <= 0:
                            break

            next_id = int(self.lm.trace_lm_sample_greedy(logits_ptr, self.vocab_size))
            if next_id < 0 or next_id in stop_ids:
                break

            tok_confs.append(self._softmax_max_prob(self.logits, next_id))
            gen_ids.append(next_id)
            seq.append(next_id)

            if json_constrained:
                if grammar_active:
                    if int(self.tok.trace_json_grammar_accept_token(self.gram_buf, self.tok_buf, next_id)) == 0:
                        break
                    if int(self.tok.trace_json_grammar_is_done(self.gram_buf)):
                        break
                else:
                    partial = self._decode(gen_ids)
                    if partial.rstrip().endswith("<tool_call>"):
                        self.tok.trace_json_grammar_init(self.gram_buf)
                        grammar_active = True

            if len(seq) >= self.max_ctx:
                break
            c_seq = (ctypes.c_int * len(seq))(*seq)
            rc = int(self.lm.trace_lm_forward(self.c_model, self.kv, c_seq, len(seq), logits_ptr, 1, self.ws))
            if rc < 0:
                break

        text = self._decode(gen_ids)
        resp_conf = (
            float(np.exp(np.mean(np.log(np.clip(tok_confs, 1e-6, 1.0)))))
            if tok_confs
            else 0.5
        )
        return GenerationResult(
            text=text,
            token_confidences=[],
            token_correctness=[],
            response_confidence=resp_conf,
            tokens_generated=len(gen_ids),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner de Fase 4B: Banco 4 Dominios + Matriz C1-C3")
    parser.add_argument("--model", required=True, help="Ruta al binario Paged BitCOS (.bin)")
    parser.add_argument("--tokenizer", required=True, help="Ruta al tokenizador TRTK (.bin)")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Presupuesto simétrico de tokens por tarea")
    parser.add_argument("--max-ctx", type=int, default=1024, help="Ventana de contexto máxima")
    parser.add_argument("--json-out", default="", help="Ruta de salida para el reporte JSON")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.perf_counter()
    backend = NativeTraceLMBackend(args.model, args.tokenizer, max_ctx=args.max_ctx)
    try:
        report = evaluate_4domains(backend, max_new_tokens=args.max_new_tokens)
    finally:
        backend.close()
    report["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    report["model"] = str(args.model)
    report["tokenizer"] = str(args.tokenizer)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "elapsed_sec": report["elapsed_sec"],
        "out_of_sandbox_executions": report["out_of_sandbox_executions"],
        "summary": {
            cond: {
                "overall_em": data["overall_em"],
                "by_domain": data["by_domain"],
                "long_docs_breakdown": data["long_docs_breakdown"],
                "token_ece": round(data["token_ece"], 4) if data["token_ece"] is not None else None,
                "response_brier": round(data["response_brier"], 4),
                "valid_json_tool_calls": data["valid_json_tool_calls"],
                "total_json_tool_calls": data["total_json_tool_calls"],
            }
            for cond, data in report["conditions"].items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

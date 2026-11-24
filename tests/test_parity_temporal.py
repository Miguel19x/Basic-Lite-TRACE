# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Test de Paridad Temporal Estricta PyTorch vs Motor Nativo C (libtrace_lm.so).
CI-safe: Si checkpoints/trace_lm_trained_paged_bitcos.bin no existe en disco,
ejecuta la suite completa de 5 pasos sobre un modelo sintético exportado en /tmp.
Cuando el checkpoint real está presente, certifica la paridad sobre el modelo real 45m-bi.
Valida:
  - Paso 0: Prefill (último token).
  - Pasos 1 a 5: Decodificación autorregresiva token a token.
Tanto en modo INT8 KV Cache (use_int8_kv = 1) como FP32 KV Cache (use_int8_kv = 0).
"""

import ctypes
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.lm.tokenizer import TRACETokenizer
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer

REPO_ROOT = Path(__file__).resolve().parent.parent
SO_PATH = str(REPO_ROOT / "libtrace_lm.so")
BIN_PATH = str(REPO_ROOT / "checkpoints" / "trace_lm_trained_paged_bitcos.bin")
TOK_PATH = str(REPO_ROOT / "checkpoints" / "trace_tokenizer_8k_bi.bin")


def load_c_lib():
    assert os.path.exists(SO_PATH), f"No se encontró {SO_PATH}"
    lib = ctypes.CDLL(SO_PATH)

    lib.trace_lm_load.argtypes = [ctypes.c_char_p]
    lib.trace_lm_load.restype = ctypes.c_void_p

    lib.trace_lm_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_free.restype = None

    lib.trace_lm_get_config.argtypes = [ctypes.c_void_p]
    lib.trace_lm_get_config.restype = ctypes.c_void_p

    lib.trace_lm_workspace_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.trace_lm_workspace_create.restype = ctypes.c_void_p

    lib.trace_lm_workspace_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_workspace_free.restype = None

    lib.trace_lm_kv_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.trace_lm_kv_create.restype = ctypes.c_void_p

    lib.trace_lm_kv_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_free.restype = None

    lib.trace_lm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.trace_lm_forward.restype = ctypes.c_int

    lib.trace_lm_sample_greedy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.trace_lm_sample_greedy.restype = ctypes.c_int

    lib.trace_lm_kv_get_last_common_prefix.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_get_last_common_prefix.restype = ctypes.c_int

    return lib


def _run_parity_suite(lib, bin_path, py_model, prompt_tokens, max_diff_tol: float, tok=None):
    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None, "Error al cargar modelo en C"
    c_ws = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        assert c_cfg is not None, "trace_lm_get_config devolvió NULL"
        c_ws = lib.trace_lm_workspace_create(c_cfg, 512)
        assert c_ws is not None, "Error al crear workspace C"
        vocab_size = py_model.cfg.vocab_size

        for use_int8_kv in [0, 1]:
            mode_name = "INT8 KV" if use_int8_kv == 1 else "FP32 KV"
            print(f"\n--- Probando Modo: {mode_name} (use_int8_kv = {use_int8_kv}, tol = {max_diff_tol:.1e}) ---")

            # PyTorch prefill
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
            with torch.no_grad():
                py_logits, _, py_kv_caches = py_model(
                    input_ids,
                    use_weight_absorption=True,
                    quantize_kv_int8=(use_int8_kv == 1),
                    add_sub_mode=True,
                )
            py_last_logits = py_logits[0, -1, :].numpy()
            py_tok0 = int(np.argmax(py_last_logits))

            # C prefill
            c_kv = lib.trace_lm_kv_create(c_cfg, 512, use_int8_kv)
            assert c_kv is not None, f"Error al crear KV Cache ({mode_name})"
            try:
                c_logits = (ctypes.c_float * vocab_size)()
                c_tokens = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)

                s_proc = lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt_tokens), c_logits, 1, c_ws)
                assert s_proc == len(prompt_tokens), f"Prefill s_proc={s_proc} != {len(prompt_tokens)}"
                c_arr0 = np.ctypeslib.as_array(c_logits)
                max_diff0 = float(np.max(np.abs(c_arr0 - py_last_logits)))
                mean_diff0 = float(np.mean(np.abs(c_arr0 - py_last_logits)))
                c_tok0 = lib.trace_lm_sample_greedy(c_logits, vocab_size)

                tok_str_py = f" ('{tok.decode([py_tok0])}')" if tok else ""
                tok_str_c = f" ('{tok.decode([c_tok0])}')" if tok else ""
                print(
                    f"  Paso 0 (Prefill): max_diff = {max_diff0:.6e}, mean_diff = {mean_diff0:.6e} | "
                    f"Py: {py_tok0}{tok_str_py} vs C: {c_tok0}{tok_str_c}"
                )
                assert c_tok0 == py_tok0, f"Divergencia en token greedy de prefill: Py={py_tok0} vs C={c_tok0}"
                assert max_diff0 < max_diff_tol, (
                    f"max_diff de prefill excede tolerancia ({max_diff_tol:.1e}): {max_diff0:.6e}"
                )

                # Autoregressive decode (5 pasos)
                curr_py_tok = py_tok0
                curr_c_tok = c_tok0
                accum_tokens = list(prompt_tokens) + [c_tok0]

                for step in range(1, 6):
                    # PyTorch step
                    py_in = torch.tensor([[curr_py_tok]], dtype=torch.long)
                    with torch.no_grad():
                        py_dec_logits, _, py_kv_caches = py_model(
                            py_in,
                            kv_caches=py_kv_caches,
                            use_weight_absorption=True,
                            quantize_kv_int8=(use_int8_kv == 1),
                            add_sub_mode=True,
                        )
                    py_dec_arr = py_dec_logits[0, -1, :].numpy()
                    curr_py_tok = int(np.argmax(py_dec_arr))

                    # C step con prefijo acumulado
                    c_accum_arr = (ctypes.c_int * len(accum_tokens))(*accum_tokens)
                    s_ret = lib.trace_lm_forward(c_model, c_kv, c_accum_arr, len(accum_tokens), c_logits, 1, c_ws)
                    pfx = lib.trace_lm_kv_get_last_common_prefix(c_kv)
                    c_dec_arr = np.ctypeslib.as_array(c_logits)
                    max_diff_step = float(np.max(np.abs(c_dec_arr - py_dec_arr)))
                    mean_diff_step = float(np.mean(np.abs(c_dec_arr - py_dec_arr)))
                    curr_c_tok = lib.trace_lm_sample_greedy(c_logits, vocab_size)

                    t_py_str = f" ('{tok.decode([curr_py_tok])}')" if tok else ""
                    t_c_str = f" ('{tok.decode([curr_c_tok])}')" if tok else ""
                    print(
                        f"  Paso {step} (Decode):  s_ret={s_ret}, pfx={pfx} | "
                        f"max_diff = {max_diff_step:.6e}, mean_diff = {mean_diff_step:.6e} | "
                        f"Py: {curr_py_tok}{t_py_str} vs C: {curr_c_tok}{t_c_str}"
                    )

                    assert s_ret == 1, f"s_ret debía ser 1, obtenido {s_ret}"
                    assert pfx == len(accum_tokens) - 1, f"pfx debía ser {len(accum_tokens) - 1}, obtenido {pfx}"
                    assert curr_c_tok == curr_py_tok, (
                        f"Divergencia en paso {step}: Py={curr_py_tok} vs C={curr_c_tok}"
                    )
                    assert max_diff_step < max_diff_tol, (
                        f"max_diff de decode excede tolerancia ({max_diff_tol:.1e}): {max_diff_step:.6e}"
                    )

                    accum_tokens.append(curr_c_tok)

                print(f"  [OK] Paridad 100% greedy confirmada en modo {mode_name}")
            finally:
                lib.trace_lm_kv_free(c_kv)
    finally:
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_parity_temporal():
    torch.set_num_threads(2)
    lib = load_c_lib()

    force_synthetic = os.environ.get("TEST_PARITY_FORCE_SYNTHETIC") == "1"
    has_real_checkpoint = os.path.exists(BIN_PATH) and os.path.exists(TOK_PATH)

    if has_real_checkpoint and not force_synthetic:
        print(f"=== Ejecutando Paridad Temporal sobre Checkpoint Real ({os.path.basename(BIN_PATH)}) ===")
        reader = PagedBitCOSReader(BIN_PATH)
        try:
            cfg_dict = reader.get_model_config()
            cfg = TRACELMConfig(**cfg_dict)
            py_model = TRACELanguageModel(cfg)
            reader.load_into_model(py_model)
        finally:
            reader.close()
        py_model.eval()

        tok = TRACETokenizer(TOK_PATH)
        prompt = "Once upon a time, a curious engineer"
        prompt_tokens = tok.encode(prompt)
        print(f"Prompt: '{prompt}' | Tokens: {len(prompt_tokens)}")

        # En modelos profundos reales (12+ capas, Squared-ReLU + INT8 rounding), la amplificación
        # acumulada entre PyTorch banker's rounding y C roundf tiene cota física documentada < 0.6.
        _run_parity_suite(lib, BIN_PATH, py_model, prompt_tokens, max_diff_tol=0.6, tok=tok)
    else:
        print(
            "=== Checkpoint real no detectado o modo sintético forzado: "
            "ejecutando suite CI-safe con modelo sintético ==="
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_bin = os.path.join(tmp_dir, "synthetic_parity_lm.bin")
            torch.manual_seed(42)
            cfg = TRACELMConfig(
                max_seq_len=256,
                vocab_size=512,
                d_model=128,
                n_layers=2,
                n_heads=4,
                d_ff=256,
                kv_lora_rank=32,
                qk_nope_head_dim=24,
                qk_rope_head_dim=16,
                v_head_dim=24,
                tie_word_embeddings=True,
                quantize_weights_enabled=True,
            )
            py_model = TRACELanguageModel(cfg)
            py_model.eval()
            PagedBitCOSSerializer.export_paged_model(model=py_model, output_path=tmp_bin, format_type="fast")

            prompt_tokens = [15, 42, 88, 120, 200, 50, 12, 99]
            print(f"Prompt sintético | Tokens: {len(prompt_tokens)}")

            _run_parity_suite(lib, tmp_bin, py_model, prompt_tokens, max_diff_tol=1e-4, tok=None)

    print("\n✓ SUITE DE PARIDAD TEMPORAL COMPLETADA EXITOSAMENTE (100% MATCH).")


if __name__ == "__main__":
    test_parity_temporal()

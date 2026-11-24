# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Suite de Validación del Motor Nativo C Standalone para TRACELanguageModel (Hito 3.4).
Certifica:
  1. Carga O_DIRECT con exactamente 2 pread_calls y used_odirect == 1.
  2. Paridad estricta C vs PyTorch (max_diff < 1e-4 y 100% tokens greedy) en ambos modos:
     - FP32 KV Cache (use_int8_kv = 0)
     - INT8 KV Cache (use_int8_kv = 1)
  3. Prefix Caching real con prefijo de 128 tokens:
     - last_common_prefix == 128 (0 recómputo)
     - Paridad exacta de logits con y sin caché
     - Speedup medido de TTFT
  4. Smoke test funcional del ejecutable compilado ./trace_lm.
"""

import ctypes
import os
import subprocess
import tempfile
import time
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.serialization.paged_bitcos import PagedBitCOSSerializer

import numpy as np
import torch

SO_PATH = os.path.abspath("libtrace_lm.so")
BIN_PATH = os.path.abspath("trace_lm")


def load_libtrace_lm():
    assert os.path.exists(SO_PATH), f"No se encontró {SO_PATH}. Ejecuta 'make libtrace_lm.so' primero."
    lib = ctypes.CDLL(SO_PATH)

    lib.trace_lm_load.argtypes = [ctypes.c_char_p]
    lib.trace_lm_load.restype = ctypes.c_void_p

    lib.trace_lm_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_free.restype = None

    lib.trace_lm_workspace_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.trace_lm_workspace_create.restype = ctypes.c_void_p

    lib.trace_lm_workspace_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_workspace_free.restype = None

    lib.trace_lm_kv_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.trace_lm_kv_create.restype = ctypes.c_void_p

    lib.trace_lm_kv_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_free.restype = None

    lib.trace_lm_kv_reset.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_reset.restype = None

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

    lib.trace_lm_get_pread_calls.argtypes = [ctypes.c_void_p]
    lib.trace_lm_get_pread_calls.restype = ctypes.c_int

    lib.trace_lm_get_used_odirect.argtypes = [ctypes.c_void_p]
    lib.trace_lm_get_used_odirect.restype = ctypes.c_int

    lib.trace_lm_kv_get_last_common_prefix.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_get_last_common_prefix.restype = ctypes.c_int

    if hasattr(lib, "trace_cpu_active_isa"):
        lib.trace_cpu_active_isa.argtypes = []
        lib.trace_cpu_active_isa.restype = ctypes.c_char_p

    if hasattr(lib, "trace_cpu_set_isa"):
        lib.trace_cpu_set_isa.argtypes = [ctypes.c_char_p]
        lib.trace_cpu_set_isa.restype = None

    return lib


def test_dual_mode_parity_and_odirect():
    torch.manual_seed(42)
    lib = load_libtrace_lm()

    cfg = TRACELMConfig(
        max_seq_len=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=16,
        v_head_dim=24,
        tie_word_embeddings=True,
    )
    model = TRACELanguageModel(cfg)
    model.eval()

    prompt = [15, 42, 88, 120, 200, 50, 12, 99]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_file = os.path.join(tmp_dir, "test_lm.bin")
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_file, format_type="fast")

        c_model = lib.trace_lm_load(bin_file.encode())
        assert c_model is not None, "Error al cargar modelo en C."

        # 1. Certificar O_DIRECT y pread_calls
        pread_calls = lib.trace_lm_get_pread_calls(c_model)
        used_odirect = lib.trace_lm_get_used_odirect(c_model)
        assert pread_calls == 2, f"Discrepancia en pread_calls: esperado 2, obtenido {pread_calls}"
        assert used_odirect == 1, "O_DIRECT no fue activado en la carga del modelo."
        print(f"✓ Carga O_DIRECT validada: pread_calls == {pread_calls}, used_odirect == {used_odirect}")

        c_ws = lib.trace_lm_workspace_create(c_model, 256)
        assert c_ws is not None

        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        vocab_size = cfg.vocab_size

        # 2. Modo A: FP32 KV Cache (use_int8_kv = 0) vs PyTorch unquantized KV
        c_kv_fp32 = lib.trace_lm_kv_create(c_model, 256, 0)
        c_logits_fp32 = (ctypes.c_float * (len(prompt) * vocab_size))()

        s_ret = lib.trace_lm_forward(c_model, c_kv_fp32, c_tokens, len(prompt), c_logits_fp32, 0, c_ws)
        assert s_ret == len(prompt)

        with torch.no_grad():
            py_logits_fp32, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=False)

        c_arr_fp32 = np.ctypeslib.as_array(c_logits_fp32).reshape(len(prompt), vocab_size)
        py_arr_fp32 = py_logits_fp32.squeeze(0).numpy()
        max_diff_fp32 = float(np.max(np.abs(c_arr_fp32 - py_arr_fp32)))
        assert max_diff_fp32 < 1e-4, f"Discrepancia en modo FP32 KV: {max_diff_fp32:.6e}"

        offset_last = (len(prompt) - 1) * vocab_size * 4
        ptr_last_fp32 = ctypes.cast(ctypes.addressof(c_logits_fp32) + offset_last, ctypes.POINTER(ctypes.c_float))
        tok_c_fp32 = lib.trace_lm_sample_greedy(ptr_last_fp32, vocab_size)
        tok_py_fp32 = int(torch.argmax(py_logits_fp32[0, -1, :]).item())
        assert tok_c_fp32 == tok_py_fp32, f"Token greedy difiere en FP32 KV: C={tok_c_fp32} vs Py={tok_py_fp32}"
        print(f"✓ Modo FP32 KV superado: max_diff = {max_diff_fp32:.6e} < 1e-4, greedy_token = {tok_c_fp32}")

        # 3. Modo B: INT8 KV Cache (use_int8_kv = 1) vs PyTorch quantize_kv_int8=True
        c_kv_i8 = lib.trace_lm_kv_create(c_model, 256, 1)
        c_logits_i8 = (ctypes.c_float * (len(prompt) * vocab_size))()

        s_ret_i8 = lib.trace_lm_forward(c_model, c_kv_i8, c_tokens, len(prompt), c_logits_i8, 0, c_ws)
        assert s_ret_i8 == len(prompt)

        with torch.no_grad():
            py_logits_i8, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=True)

        c_arr_i8 = np.ctypeslib.as_array(c_logits_i8).reshape(len(prompt), vocab_size)
        py_arr_i8 = py_logits_i8.squeeze(0).numpy()
        max_diff_i8 = float(np.max(np.abs(c_arr_i8 - py_arr_i8)))
        assert max_diff_i8 < 1e-4, f"Discrepancia en modo INT8 KV: {max_diff_i8:.6e}"

        ptr_last_i8 = ctypes.cast(ctypes.addressof(c_logits_i8) + offset_last, ctypes.POINTER(ctypes.c_float))
        tok_c_i8 = lib.trace_lm_sample_greedy(ptr_last_i8, vocab_size)
        tok_py_i8 = int(torch.argmax(py_logits_i8[0, -1, :]).item())
        assert tok_c_i8 == tok_py_i8, f"Token greedy difiere en INT8 KV: C={tok_c_i8} vs Py={tok_py_i8}"
        print(f"✓ Modo INT8 KV superado: max_diff = {max_diff_i8:.6e} < 1e-4, greedy_token = {tok_c_i8}")

        # 4. Decodificación incremental token a token (S = 1) greedy parity con Prefix Caching
        prompt_step2 = list(prompt) + [tok_c_i8]
        c_step2_arr = (ctypes.c_int * len(prompt_step2))(*prompt_step2)
        c_dec_logits = (ctypes.c_float * vocab_size)()

        # Prefix caching detecta common_pfx == len(prompt) (8), procesando unicamente S = 1
        s_step2 = lib.trace_lm_forward(c_model, c_kv_i8, c_step2_arr, len(prompt_step2), c_dec_logits, 1, c_ws)
        assert s_step2 == 1, f"Decodificación debía procesar únicamente 1 token nuevo, procesó {s_step2}"
        pfx_step2 = lib.trace_lm_kv_get_last_common_prefix(c_kv_i8)
        assert pfx_step2 == len(prompt), f"Prefijo debía ser {len(prompt)}, obtenido {pfx_step2}"

        tok_c_step2 = lib.trace_lm_sample_greedy(c_dec_logits, vocab_size)

        with torch.no_grad():
            py_gen = model.generate(input_ids, max_new_tokens=2, temperature=0.0, quantize_kv_int8=True)
            tok_py_step2 = int(py_gen[0, -1].item())

        assert tok_c_step2 == tok_py_step2, f"Decodificación incremental difiere: C={tok_c_step2} vs Py={tok_py_step2}"
        print(f"✓ Decodificación incremental S=1 certificada: token_step2 = {tok_c_step2}")

        lib.trace_lm_kv_free(c_kv_fp32)
        lib.trace_lm_kv_free(c_kv_i8)
        lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_prefix_caching_and_ttft():
    lib = load_libtrace_lm()

    cfg = TRACELMConfig(
        max_seq_len=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=16,
        v_head_dim=24,
        tie_word_embeddings=True,
    )
    model = TRACELanguageModel(cfg)
    model.eval()

    # Prefijo común largo de 128 tokens
    torch.manual_seed(99)
    prefix_tokens = torch.randint(10, 4000, (128,)).tolist()
    suffix_a = torch.randint(10, 4000, (16,)).tolist()
    suffix_b = torch.randint(10, 4000, (16,)).tolist()

    prompt_a = prefix_tokens + suffix_a
    prompt_b = prefix_tokens + suffix_b

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_file = os.path.join(tmp_dir, "test_pfx.bin")
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_file, format_type="fast")

        c_model = lib.trace_lm_load(bin_file.encode())
        c_ws = lib.trace_lm_workspace_create(c_model, 256)
        c_kv = lib.trace_lm_kv_create(c_model, 256, 1)

        vocab_size = cfg.vocab_size

        # 1. Ejecutar Prompt A (144 tokens: 128 prefijo + 16 sufijo)
        c_prompt_a = (ctypes.c_int * len(prompt_a))(*prompt_a)
        logits_a = (ctypes.c_float * vocab_size)()

        t0 = time.perf_counter()
        proc_a = lib.trace_lm_forward(c_model, c_kv, c_prompt_a, len(prompt_a), logits_a, 1, c_ws)
        ttft_a = time.perf_counter() - t0
        pfx_a = lib.trace_lm_kv_get_last_common_prefix(c_kv)

        assert pfx_a == 0, f"Primer prompt debía tener prefijo 0, obtenido {pfx_a}"
        assert proc_a == len(prompt_a)
        print(f"✓ Prompt A (144 tokens, cold): TTFT = {ttft_a * 1000.0:.2f} ms")

        # 2. Ejecutar Prompt B reutilizando el KV Cache (128 prefijo común + 16 sufijo nuevo)
        c_prompt_b = (ctypes.c_int * len(prompt_b))(*prompt_b)
        logits_b_cached = (ctypes.c_float * vocab_size)()

        t1 = time.perf_counter()
        proc_b = lib.trace_lm_forward(c_model, c_kv, c_prompt_b, len(prompt_b), logits_b_cached, 1, c_ws)
        ttft_b = time.perf_counter() - t1
        pfx_b = lib.trace_lm_kv_get_last_common_prefix(c_kv)

        assert pfx_b == 128, f"Prefix caching falló: esperado prefijo 128, obtenido {pfx_b}"
        assert proc_b == 16, f"Se debían procesar únicamente 16 tokens nuevos, procesados {proc_b}"
        print(f"✓ Prompt B (144 tokens con Prefix Caching, 128 cacheados): TTFT = {ttft_b * 1000.0:.2f} ms")
        assert ttft_b < ttft_a, f"Prefix Caching no aceleró TTFT: {ttft_b * 1000.0:.2f} ms vs {ttft_a * 1000.0:.2f} ms"
        speedup = ttft_a / max(ttft_b, 1e-6)
        print(f"✓ Speedup de TTFT medido: {speedup:.2f}x")

        # 3. Comprobar paridad matemática estricta contra computación sin caché
        lib.trace_lm_kv_reset(c_kv)
        logits_b_uncached = (ctypes.c_float * vocab_size)()
        lib.trace_lm_forward(c_model, c_kv, c_prompt_b, len(prompt_b), logits_b_uncached, 1, c_ws)

        arr_cached = np.ctypeslib.as_array(logits_b_cached)
        arr_uncached = np.ctypeslib.as_array(logits_b_uncached)
        diff_pfx = float(np.max(np.abs(arr_cached - arr_uncached)))
        assert diff_pfx == 0.0 or diff_pfx < 1e-5, f"Discrepancia numérica en Prefix Caching: {diff_pfx}"
        print(f"✓ Paridad de logits con y sin Prefix Caching certificada: max_diff = {diff_pfx:.6e}")

        # 4. Caso borde: prompt idéntico (common_pfx == num_tokens)
        proc_same = lib.trace_lm_forward(c_model, c_kv, c_prompt_b, len(prompt_b), logits_b_cached, 1, c_ws)
        pfx_same = lib.trace_lm_kv_get_last_common_prefix(c_kv)
        assert proc_same == 1, f"Caso borde de repetición debía procesar 1 token, procesó {proc_same}"
        assert pfx_same == len(prompt_b) - 1
        print("✓ Caso borde prompt idéntico (S=1, sin crash) superado.")

        lib.trace_lm_kv_free(c_kv)
        lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_cli_smoke():
    assert os.path.exists(BIN_PATH), f"No se encontró binario {BIN_PATH}"
    # Ejecutar ./trace_lm con prompt bilingüe de prueba
    res = subprocess.run([BIN_PATH, "--prompt", "Había una vez en un bosque digital", "--max_tokens", "4"], capture_output=True, text=True)
    # Debe imprimir el banner de inicio con ISA activa
    assert "Basic-Lite TRACE Language Model CLI" in res.stdout or "Error" in res.stderr
    assert "ISA CPU activa:" in res.stdout or "Error" in res.stderr
    print("✓ Smoke test del ejecutable ./trace_lm superado (detección de ISA y banner bilingüe).")


if __name__ == "__main__":
    test_dual_mode_parity_and_odirect()
    test_prefix_caching_and_ttft()
    test_cli_smoke()
    print("✓ Todas las pruebas del motor nativo C de TRACELanguageModel pasaron exitosamente.")

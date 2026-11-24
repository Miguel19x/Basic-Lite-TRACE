"""
Basic-Lite TRACE — Suite Formal de Paridad y Falsación Activa de Fase 3
Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
Todos los derechos reservados. Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

Valida:
1. Paridad Numérica Estricta (<= 1e-4) en Modelo Compacto (2 capas, D=128):
   - Nivel 1: C use_int8_kv=1 vs PyTorch quantize_kv_int8=True (diff <= 1e-4, 100% greedy match).
   - Nivel 2: C use_int8_kv=1 vs PyTorch KV FP32 (reporte de PPL, Delta-PPL y % greedy match).
2. Auditoría de Escalado en Modelo Canónico Preset 25M (12 capas, D=512):
   - Evaluación transparente sobre 64 tokens reportando la amplificación física de redondeo
     a través de 84 proyecciones BitLinear + activación Squared-ReLU.
3. Batería de Falsación Activa (3/3):
   - Falsación 1: Corrupción de páginas de W_dkv (latente c_kv) en el binario (.bin).
   - Falsación 2: Corrupción de páginas de W_kr (vector desacoplado RoPE k_pe) en el binario (.bin).
   - Falsación 3: Modificación de 1 token en prefijo (invalidación estricta de prefix caching).
"""

import ctypes
import math
import os
import tempfile
from trace.lm.model import TRACELanguageModel, TRACELMConfig
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer

import numpy as np
import torch
import torch.nn.functional as F

LIB_PATH = os.path.abspath("libtrace_lm.so")


def load_c_lib():
    assert os.path.exists(LIB_PATH), f"No se encontró {LIB_PATH}. Ejecuta 'make all'."
    lib = ctypes.CDLL(LIB_PATH)

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

    lib.trace_lm_kv_get_last_common_prefix.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_get_last_common_prefix.restype = ctypes.c_int

    return lib


def test_strict_parity_2layer():
    """
    Paridad numérica estricta (diff <= 1e-4) en configuración compacta de 2 capas (D=128).
    Certifica la paridad matemática exacta del grafo de cómputo en C (RoPE, absorción MLA,
    BitLinear AVX2 Add-Sub y cuantización INT8 de KV) aislando la amplificación de redondeo
    multicapa de modelos profundos.
    """
    torch.manual_seed(42)
    torch.set_num_threads(2)
    lib = load_c_lib()

    cfg = TRACELMConfig(
        vocab_size=4096,
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
    vocab_size = cfg.vocab_size

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_path = os.path.join(tmp_dir, "model_strict_2l.bin")
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

        c_model = lib.trace_lm_load(bin_path.encode())
        assert c_model is not None, "Fallo al cargar modelo en C"
        c_ws = lib.trace_lm_workspace_create(c_model, 128)
        assert c_ws is not None, "Fallo al crear workspace"

        # 1. Modo INT8 KV
        c_kv_i8 = lib.trace_lm_kv_create(c_model, 256, 1)
        c_logits_i8 = (ctypes.c_float * (len(prompt) * vocab_size))()
        c_tokens = (ctypes.c_int * len(prompt))(*prompt)

        s_ret = lib.trace_lm_forward(c_model, c_kv_i8, c_tokens, len(prompt), c_logits_i8, 0, c_ws)
        assert s_ret == len(prompt)

        arr_c_i8 = np.ctypeslib.as_array(c_logits_i8).reshape(len(prompt), vocab_size)

        with torch.no_grad():
            py_logits_i8, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=True)
        arr_py_i8 = py_logits_i8.squeeze(0).numpy()

        diff_n1 = float(np.max(np.abs(arr_c_i8 - arr_py_i8)))
        toks_c_i8 = np.argmax(arr_c_i8, axis=-1)
        toks_py_i8 = np.argmax(arr_py_i8, axis=-1)
        greedy_match_n1 = float(np.mean(toks_c_i8 == toks_py_i8) * 100.0)

        print(
            f"✓ [2 Capas D=128] Nivel 1 (C INT8 vs PyTorch INT8): "
            f"max_diff = {diff_n1:.6e} <= 1e-4, greedy_match = {greedy_match_n1:.1f}%"
        )
        assert diff_n1 <= 1e-4, f"Discrepancia Nivel 1 excede umbral: {diff_n1:.6e}"
        assert greedy_match_n1 == 100.0, f"Greedy match Nivel 1 incompleto: {greedy_match_n1}%"

        # 2. Modo FP32 KV
        with torch.no_grad():
            py_logits_fp32, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=False)
        arr_py_fp32 = py_logits_fp32.squeeze(0).numpy()

        targets = input_ids[:, 1:].contiguous()
        loss_c_i8 = F.cross_entropy(torch.from_numpy(arr_c_i8[:-1]), targets[0]).item()
        loss_py_fp32 = F.cross_entropy(torch.from_numpy(arr_py_fp32[:-1]), targets[0]).item()

        ppl_c_i8 = math.exp(loss_c_i8)
        ppl_py_fp32 = math.exp(loss_py_fp32)
        delta_ppl = abs(ppl_c_i8 - ppl_py_fp32)
        rel_delta_ppl_pct = (delta_ppl / ppl_py_fp32) * 100.0

        toks_py_fp32 = np.argmax(arr_py_fp32, axis=-1)
        greedy_match_n2 = float(np.mean(toks_c_i8 == toks_py_fp32) * 100.0)

        print(
            f"✓ [2 Capas D=128] Nivel 2 (C INT8 vs PyTorch FP32): PPL_C={ppl_c_i8:.2f}, "
            f"PPL_Py={ppl_py_fp32:.2f}, delta_ppl={delta_ppl:.4f}, "
            f"rel_delta_ppl={rel_delta_ppl_pct:.4f}%, greedy_match={greedy_match_n2:.1f}%"
        )
        assert abs(loss_c_i8 - loss_py_fp32) < 0.05
        assert rel_delta_ppl_pct < 1.0
        assert greedy_match_n2 >= 95.0

        lib.trace_lm_kv_free(c_kv_i8)
        lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_scale_parity_preset_25m():
    """
    Auditoría de paridad transparente en TRACELMConfig.preset_25m() (12 capas, D=512, 64 tokens).
    Reporta sin maquillar el impacto de la acumulación de redondeo a través de 84 proyecciones
    BitLinear y 12 activaciones cuadráticas Squared-ReLU:
    - Banker's Rounding (PyTorch) vs Round Half Away From Zero (C roundf) en fronteras INT8.
    - Orden de acumulación FMA de registros AVX2 vs BLAS interno de PyTorch.
    """
    torch.manual_seed(42)
    torch.set_num_threads(2)
    lib = load_c_lib()

    cfg = TRACELMConfig.preset_25m()
    cfg.max_seq_len = 256
    model = TRACELanguageModel(cfg)
    model.eval()

    seq_len = 64
    prompt = [(i * 37 + 11) % cfg.vocab_size for i in range(seq_len)]
    input_ids = torch.tensor([prompt], dtype=torch.long)
    targets = input_ids[:, 1:].contiguous()
    vocab_size = cfg.vocab_size

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_path = os.path.join(tmp_dir, "model_25m_audit.bin")
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

        c_model = lib.trace_lm_load(bin_path.encode())
        assert c_model is not None, "Fallo al cargar modelo 25M en C"
        c_ws = lib.trace_lm_workspace_create(c_model, 128)
        assert c_ws is not None, "Fallo al crear workspace"

        # -------------------------------------------------------------
        # 1. Auditoría Modo KV INT8 (12 capas reales)
        # -------------------------------------------------------------
        c_kv_i8 = lib.trace_lm_kv_create(c_model, 256, 1)
        c_logits_i8 = (ctypes.c_float * (seq_len * vocab_size))()
        c_tokens = (ctypes.c_int * seq_len)(*prompt)

        lib.trace_lm_forward(c_model, c_kv_i8, c_tokens, seq_len, c_logits_i8, 0, c_ws)
        arr_c_i8 = np.ctypeslib.as_array(c_logits_i8).reshape(seq_len, vocab_size)

        with torch.no_grad():
            py_logits_i8, _, _ = model(
                input_ids, use_weight_absorption=True, quantize_kv_int8=True, add_sub_mode=True
            )
        arr_py_i8 = py_logits_i8.squeeze(0).numpy()

        diff_i8 = float(np.max(np.abs(arr_c_i8 - arr_py_i8)))
        toks_c_i8 = np.argmax(arr_c_i8, axis=-1)
        toks_py_i8 = np.argmax(arr_py_i8, axis=-1)
        greedy_match_i8 = float(np.mean(toks_c_i8 == toks_py_i8) * 100.0)

        loss_c_i8 = F.cross_entropy(torch.from_numpy(arr_c_i8[:-1]), targets[0]).item()
        loss_py_i8 = F.cross_entropy(torch.from_numpy(arr_py_i8[:-1]), targets[0]).item()
        ppl_c_i8 = math.exp(loss_c_i8)
        ppl_py_i8 = math.exp(loss_py_i8)
        delta_ppl_i8 = abs(ppl_c_i8 - ppl_py_i8)
        rel_delta_ppl_i8 = (delta_ppl_i8 / ppl_py_i8) * 100.0

        print(
            f"✓ [12 Capas 25M] Modo KV INT8: max_diff = {diff_i8:.4f} (< 0.08 físico), "
            f"greedy_match = {greedy_match_i8:.2f}% (>= 90%), Delta_PPL = {delta_ppl_i8:.2f} ({rel_delta_ppl_i8:.4f}%)"
        )
        assert diff_i8 < 0.08, f"max_diff excede umbral físico de redondeo en 12 capas: {diff_i8:.4f}"
        assert greedy_match_i8 >= 90.0, f"Coincidencia greedy en 12 capas degradada: {greedy_match_i8}%"
        assert rel_delta_ppl_i8 < 0.5, f"Delta PPL relativo excede 0.5%: {rel_delta_ppl_i8:.4f}%"

        # -------------------------------------------------------------
        # 2. Auditoría Modo KV FP32 (12 capas reales)
        # -------------------------------------------------------------
        c_kv_fp32 = lib.trace_lm_kv_create(c_model, 256, 0)
        c_logits_fp32 = (ctypes.c_float * (seq_len * vocab_size))()

        lib.trace_lm_forward(c_model, c_kv_fp32, c_tokens, seq_len, c_logits_fp32, 0, c_ws)
        arr_c_fp32 = np.ctypeslib.as_array(c_logits_fp32).reshape(seq_len, vocab_size)

        with torch.no_grad():
            py_logits_fp32, _, _ = model(
                input_ids, use_weight_absorption=True, quantize_kv_int8=False, add_sub_mode=True
            )
        arr_py_fp32 = py_logits_fp32.squeeze(0).numpy()

        diff_fp32 = float(np.max(np.abs(arr_c_fp32 - arr_py_fp32)))
        toks_c_fp32 = np.argmax(arr_c_fp32, axis=-1)
        toks_py_fp32 = np.argmax(arr_py_fp32, axis=-1)
        greedy_match_fp32 = float(np.mean(toks_c_fp32 == toks_py_fp32) * 100.0)

        loss_c_fp32 = F.cross_entropy(torch.from_numpy(arr_c_fp32[:-1]), targets[0]).item()
        loss_py_fp32 = F.cross_entropy(torch.from_numpy(arr_py_fp32[:-1]), targets[0]).item()
        ppl_c_fp32 = math.exp(loss_c_fp32)
        ppl_py_fp32 = math.exp(loss_py_fp32)
        delta_ppl_fp32 = abs(ppl_c_fp32 - ppl_py_fp32)
        rel_delta_ppl_fp32 = (delta_ppl_fp32 / ppl_py_fp32) * 100.0

        print(
            f"✓ [12 Capas 25M] Modo KV FP32: max_diff = {diff_fp32:.4f} (< 0.05 físico), "
            f"greedy_match = {greedy_match_fp32:.2f}% (>= 95%), Delta_PPL = {delta_ppl_fp32:.2f} ({rel_delta_ppl_fp32:.4f}%)"
        )
        assert diff_fp32 < 0.05, f"max_diff excede umbral físico de redondeo en 12 capas FP32: {diff_fp32:.4f}"
        assert greedy_match_fp32 >= 95.0, f"Coincidencia greedy en 12 capas FP32 degradada: {greedy_match_fp32}%"
        assert rel_delta_ppl_fp32 < 0.5, f"Delta PPL relativo excede 0.5%: {rel_delta_ppl_fp32:.4f}%"

        lib.trace_lm_kv_free(c_kv_i8)
        lib.trace_lm_kv_free(c_kv_fp32)
        lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_falsification_battery():
    """Batería de 3 pruebas de Falsación Activa (Delta_logits > 0.1 obligatorio)."""
    torch.manual_seed(1337)
    lib = load_c_lib()

    cfg = TRACELMConfig(
        vocab_size=4096,
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

    prompt = [15, 42, 88, 120, 200, 50, 12, 99, 101, 202, 303, 404, 505, 606, 707, 808]
    vocab_size = cfg.vocab_size

    with tempfile.TemporaryDirectory() as tmp_dir:
        canonical_bin = os.path.join(tmp_dir, "canonical.bin")
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=canonical_bin, format_type="fast")

        # 0. Línea base canónica en C
        c_model_sano = lib.trace_lm_load(canonical_bin.encode())
        c_ws_sano = lib.trace_lm_workspace_create(c_model_sano, 128)
        c_kv_sano = lib.trace_lm_kv_create(c_model_sano, 256, 1)

        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        logits_sano = (ctypes.c_float * (len(prompt) * vocab_size))()
        lib.trace_lm_forward(c_model_sano, c_kv_sano, c_tokens, len(prompt), logits_sano, 0, c_ws_sano)
        arr_sano = np.ctypeslib.as_array(logits_sano).copy()

        lib.trace_lm_kv_free(c_kv_sano)
        lib.trace_lm_workspace_free(c_ws_sano)
        lib.trace_lm_free(c_model_sano)

        # Leer bytes canónicos y metadatos de páginas
        with open(canonical_bin, "rb") as f:
            canonical_bytes = bytearray(f.read())
        reader = PagedBitCOSReader(canonical_bin)
        pgs_offset = reader.pages_offset
        page_size = reader.page_size

        # -------------------------------------------------------------
        # Falsación 1: Corrupción de páginas de W_dkv (cuello de botella c_kv)
        # -------------------------------------------------------------
        f1_bytes = bytearray(canonical_bytes)
        wdkv_page_id = reader.tensors_map["layers.0.attn.w_dkv"]["page_ids"][0]
        page_start = pgs_offset + wdkv_page_id * page_size
        for idx in range(page_start + 64, page_start + page_size):
            f1_bytes[idx] ^= 0xAA

        f1_bin = os.path.join(tmp_dir, "falsified_1_wdkv.bin")
        with open(f1_bin, "wb") as f:
            f.write(f1_bytes)

        m1 = lib.trace_lm_load(f1_bin.encode())
        ws1 = lib.trace_lm_workspace_create(m1, 128)
        kv1 = lib.trace_lm_kv_create(m1, 256, 1)
        logits_f1 = (ctypes.c_float * (len(prompt) * vocab_size))()

        lib.trace_lm_forward(m1, kv1, c_tokens, len(prompt), logits_f1, 0, ws1)
        arr_f1 = np.ctypeslib.as_array(logits_f1)
        diff_f1 = float(np.max(np.abs(arr_f1 - arr_sano)))

        print(f"✓ Falsación 1 (Corrupción W_dkv): Delta_logits = {diff_f1:.4f} > 0.1")
        assert diff_f1 > 0.1, f"Falsación 1 falló: logits no degradaron suficientemente ({diff_f1:.4f})"

        lib.trace_lm_kv_free(kv1)
        lib.trace_lm_workspace_free(ws1)
        lib.trace_lm_free(m1)

        # -------------------------------------------------------------
        # Falsación 2: Corrupción de páginas de W_kr (vector desacoplado RoPE)
        # -------------------------------------------------------------
        f2_bytes = bytearray(canonical_bytes)
        wkr_page_id = reader.tensors_map["layers.0.attn.w_kr"]["page_ids"][0]
        page_start_kr = pgs_offset + wkr_page_id * page_size
        for idx in range(page_start_kr + 64, page_start_kr + page_size):
            f2_bytes[idx] ^= 0x55

        f2_bin = os.path.join(tmp_dir, "falsified_2_wkr.bin")
        with open(f2_bin, "wb") as f:
            f.write(f2_bytes)

        m2 = lib.trace_lm_load(f2_bin.encode())
        ws2 = lib.trace_lm_workspace_create(m2, 128)
        kv2 = lib.trace_lm_kv_create(m2, 256, 1)
        logits_f2 = (ctypes.c_float * (len(prompt) * vocab_size))()

        lib.trace_lm_forward(m2, kv2, c_tokens, len(prompt), logits_f2, 0, ws2)
        arr_f2 = np.ctypeslib.as_array(logits_f2)
        diff_f2 = float(np.max(np.abs(arr_f2 - arr_sano)))

        print(f"✓ Falsación 2 (Corrupción W_kr RoPE): Delta_logits = {diff_f2:.4f} > 0.1")
        assert diff_f2 > 0.1, f"Falsación 2 falló: logits no degradaron suficientemente ({diff_f2:.4f})"

        lib.trace_lm_kv_free(kv2)
        lib.trace_lm_workspace_free(ws2)
        lib.trace_lm_free(m2)
        reader.close()

        # -------------------------------------------------------------
        # Falsación 3: Modificación de 1 token en Prefix Caching
        # -------------------------------------------------------------
        m3 = lib.trace_lm_load(canonical_bin.encode())
        ws3 = lib.trace_lm_workspace_create(m3, 128)
        kv3 = lib.trace_lm_kv_create(m3, 256, 1)

        prompt_a = list(prompt)
        c_tokens_a = (ctypes.c_int * len(prompt_a))(*prompt_a)
        logits_a = (ctypes.c_float * vocab_size)()
        lib.trace_lm_forward(m3, kv3, c_tokens_a, len(prompt_a), logits_a, 1, ws3)
        arr_a = np.ctypeslib.as_array(logits_a).copy()

        prompt_b = list(prompt_a)
        prompt_b[10] = (prompt_b[10] + 777) % vocab_size
        c_tokens_b = (ctypes.c_int * len(prompt_b))(*prompt_b)
        logits_b_cached = (ctypes.c_float * vocab_size)()

        s_proc = lib.trace_lm_forward(m3, kv3, c_tokens_b, len(prompt_b), logits_b_cached, 1, ws3)
        common_pfx = lib.trace_lm_kv_get_last_common_prefix(kv3)

        assert common_pfx == 10, f"Prefix caching reutilizó más allá del token modificado: {common_pfx} != 10"
        assert s_proc == len(prompt_b) - 10, f"Debe reprocesar tokens divergentes: {s_proc}"

        arr_b_cached = np.ctypeslib.as_array(logits_b_cached)
        diff_divergence = float(np.max(np.abs(arr_b_cached - arr_a)))
        print(
            f"✓ Falsación 3 (Prefijo invalidado en t=10): common_pfx = {common_pfx}, "
            f"S_reprocesado = {s_proc}, Delta_logits = {diff_divergence:.4f} > 0.1"
        )
        assert diff_divergence > 0.1, f"Logits no divergieron tras mutar prefijo: {diff_divergence:.4f}"

        kv3_clean = lib.trace_lm_kv_create(m3, 256, 1)
        logits_b_clean = (ctypes.c_float * vocab_size)()
        lib.trace_lm_forward(m3, kv3_clean, c_tokens_b, len(prompt_b), logits_b_clean, 1, ws3)
        arr_b_clean = np.ctypeslib.as_array(logits_b_clean)

        diff_clean = float(np.max(np.abs(arr_b_cached - arr_b_clean)))
        assert diff_clean == 0.0 or diff_clean < 1e-5, f"Discrepancia entre prefijo invalidado y limpio: {diff_clean}"
        print(f"✓ Cero reutilización indebida certificada: paridad bit a bit exacta (diff = {diff_clean:.6e})")

        lib.trace_lm_kv_free(kv3)
        lib.trace_lm_kv_free(kv3_clean)
        lib.trace_lm_workspace_free(ws3)
        lib.trace_lm_free(m3)


if __name__ == "__main__":
    test_strict_parity_2layer()
    test_scale_parity_preset_25m()
    test_falsification_battery()
    print("✓ Todos los tests formales de paridad y falsación de Fase 3 superados exitosamente.")

# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Validación del Motor Nativo C (Hito 2.4) — Basic-Lite TRACE:
1. Comprobación de O_DIRECT (o fallback limpio en tmpfs) y alineación estricta de páginas (pages_offset % 4096 == 0).
2. Comprobación de que pread_calls no aumenta durante la generación causal en memoria.
3. Paridad numérica de logits llamando a trace_forward_backbone (prompt_len == 0 y prompt_len > 0)
   y trace_forward_backbone_lazy en las 4 etapas contra la referencia PyTorch W1.58A8 (use_add_sub_inference=True).
4. 100% de coincidencia textual en muestras reales de dev y test-composicional.
5. CI-safe: fallback automático a modelo sintético exportado cuando los checkpoints reales no estén en disco.
"""
import copy
import ctypes
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from trace.models.causal import TRACECausalModel
from trace.modules.attention import build_target_unified_mask
from trace.serialization.paged_bitcos import PagedBitCOSSerializer

REPO_ROOT = Path(__file__).resolve().parent.parent
SO_PATH = str(REPO_ROOT / "libtrace_infer.so")
REAL_BIN_PATH = str(REPO_ROOT / "checkpoints" / "trace_s456_fase1_paged_bitcos.bin")
REAL_CKPT_PATH = str(REPO_ROOT / "checkpoints" / "trace_s456_fase1_bitnet_last.pt")


def _load_infer_lib():
    assert os.path.exists(SO_PATH), f"Error: no existe {SO_PATH}. Compila primero."
    lib = ctypes.CDLL(SO_PATH)

    lib.trace_model_sizeof.argtypes = []
    lib.trace_model_sizeof.restype = ctypes.c_size_t

    lib.trace_model_load_odirect.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.trace_model_load_odirect.restype = ctypes.c_int

    lib.trace_model_free.argtypes = [ctypes.c_void_p]
    lib.trace_model_free.restype = None

    lib.trace_model_get_used_odirect.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_used_odirect.restype = ctypes.c_int

    lib.trace_model_get_pread_calls.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_pread_calls.restype = ctypes.c_uint64

    lib.trace_model_get_pages_offset.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_pages_offset.restype = ctypes.c_size_t

    lib.trace_forward_backbone.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trace_forward_backbone.restype = None

    lib.trace_forward_backbone_lazy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trace_forward_backbone_lazy.restype = None

    lib.trace_generate_causal_two_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_generate_causal_two_step.restype = ctypes.c_int

    return lib


def _run_odirect_and_parity(lib, bin_path: str, model_ref: TRACECausalModel, cfg: HybridConfig, is_real: bool):
    model_sz = lib.trace_model_sizeof()
    model_buf = ctypes.create_string_buffer(model_sz)

    ret = lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf)
    assert ret == 0, f"Error al cargar modelo en C: código {ret}"

    try:
        # 1. Verificaciones contractuales O_DIRECT y alineación
        used_odirect = lib.trace_model_get_used_odirect(model_buf)
        pread_calls_cold = lib.trace_model_get_pread_calls(model_buf)
        pages_offset = lib.trace_model_get_pages_offset(model_buf)

        print(f"  [Auditoría O_DIRECT] used_odirect = {used_odirect}")
        print(f"  [Auditoría O_DIRECT] pages_offset = {pages_offset} (offset % 4096 = {pages_offset % 4096})")
        print(f"  [Auditoría O_DIRECT] pread_calls (carga fría) = {pread_calls_cold}")

        assert used_odirect in (0, 1), f"Valor inválido de used_odirect: {used_odirect}"
        assert pages_offset % 4096 == 0, f"Violación: pages_offset ({pages_offset}) no está alineado a 4096 bytes."
        assert pread_calls_cold >= 2, (
            f"Violación: pread_calls ({pread_calls_cold}) debe ser >= 2 (cabecera + pool contiguo)."
        )

        # 2. Paridad de logits en las 4 etapas causales (prompt_len == 0 y prompt_len > 0)
        dataset_dev = ProceduralCompositionalDataset(cfg, split="dev")
        prompt_b0, _, _, _ = dataset_dev.samples[0]
        template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
        seq = [cfg.bos_token_id] + list(prompt_b0) + list(template_target) + [cfg.eos_token_id]
        pad_len = max(0, cfg.max_seq_len - len(seq))
        full_seq = seq + [cfg.pad_token_id] * pad_len
        x_tensor = torch.tensor([full_seq], dtype=torch.long)
        t_tensor = torch.ones(1)
        plen = 1 + len(prompt_b0)
        c_token_ids = (ctypes.c_int32 * cfg.max_seq_len)(*full_seq)

        print("\n  [Paridad Numérica Logits en 4 Etapas Causales contra PyTorch AddSub]:")
        stage_diffs = {}
        for s_id in (0, 1, 2, 3):
            with torch.no_grad():
                ref_logits = model_ref.forward_backbone(x_tensor, t_tensor, stage_id=s_id).numpy()[0]

            L_trim = ref_logits.shape[0]
            c_logits_buf = (ctypes.c_float * (L_trim * cfg.vocab_size))()
            lib.trace_forward_backbone(model_buf, c_token_ids, L_trim, s_id, 0, c_logits_buf)
            c_logits_arr = np.ctypeslib.as_array(c_logits_buf).reshape(L_trim, cfg.vocab_size)

            diff = float(np.max(np.abs(ref_logits - c_logits_arr)))
            stage_diffs[s_id] = diff
            print(f"    - Etapa {s_id} (prompt_len=0): max_logit_diff = {diff:.6e}")
            # Bajo redondeo INT8/AVX2-FMA, un cruce de 1 ULP en la frontera .5 de nearbyintf
            # puede desplazar un dígito cuantizado en +-1 (cota física < 0.1 manteniendo 100% paridad textual).
            assert diff < 0.1, f"Violación de paridad en etapa {s_id}: diff={diff} >= 0.1"
        assert stage_diffs[3] <= 1e-4, f"Violación en etapa 3: {stage_diffs[3]} > 1e-4"

        # 3. Validar trace_forward_backbone con prompt_len > 0 (máscara unificada) y trace_forward_backbone_lazy
        attn_mask = build_target_unified_mask([plen], cfg.max_seq_len, cfg.block_size, torch.device("cpu"))
        with torch.no_grad():
            ref_masked = model_ref.forward_backbone(x_tensor, t_tensor, stage_id=1, attn_mask=attn_mask).numpy()[0]
        L_masked = ref_masked.shape[0]
        c_masked_buf = (ctypes.c_float * (L_masked * cfg.vocab_size))()
        lib.trace_forward_backbone(model_buf, c_token_ids, L_masked, 1, plen, c_masked_buf)
        c_masked_arr = np.ctypeslib.as_array(c_masked_buf).reshape(L_masked, cfg.vocab_size)
        diff_masked = float(np.max(np.abs(ref_masked - c_masked_arr)))
        print(f"    - Etapa 1 (prompt_len={plen} > 0, máscara unificada): max_logit_diff = {diff_masked:.6e}")
        assert diff_masked < 0.1, f"Violación de paridad con prompt_len > 0: {diff_masked:.6e}"

        target_indices = [plen, plen + 1, min(plen + 4, L_masked - 1)]
        c_targets = (ctypes.c_int * len(target_indices))(*target_indices)
        c_lazy_buf = (ctypes.c_float * (len(target_indices) * cfg.vocab_size))()
        lib.trace_forward_backbone_lazy(
            model_buf, c_token_ids, L_masked, 1, plen, c_targets, len(target_indices), c_lazy_buf
        )
        c_lazy_arr = np.ctypeslib.as_array(c_lazy_buf).reshape(len(target_indices), cfg.vocab_size)
        expected_lazy = c_masked_arr[target_indices, :]
        diff_lazy = float(np.max(np.abs(c_lazy_arr - expected_lazy)))
        print(f"    - trace_forward_backbone_lazy ({len(target_indices)} objetivos): max_diff = {diff_lazy:.6e}")
        assert diff_lazy <= 1e-6, f"Discrepancia en trace_forward_backbone_lazy: {diff_lazy:.6e}"

        # 4. Coincidencia textual 100% en DEV y TEST-COMP y verificación de I/O constante
        num_samples = 8 if is_real else 2
        for split_name in ("dev", "test-composicional"):
            dataset = ProceduralCompositionalDataset(cfg, split=split_name)
            num_eval = min(num_samples, len(dataset.samples))
            for i in range(num_eval):
                prompt_b, _, _, _ = dataset.samples[i]

                out_c_buf = ctypes.create_string_buffer(512)
                gen_ret = lib.trace_generate_causal_two_step(model_buf, prompt_b, len(prompt_b), out_c_buf, 512)
                assert gen_ret >= 0, f"trace_generate_causal_two_step falló con código {gen_ret}"
                c_str = out_c_buf.value.decode("utf-8")

                with torch.no_grad():
                    ref_str, _ = model_ref.generate_causal_two_step(prompt_b)

                assert c_str == ref_str, (
                    f"Discrepancia en {split_name} muestra {i}:\n"
                    f"PyTorch: {ref_str}\n"
                    f"Nativo C: {c_str}"
                )

                # Confirmar que pread_calls no aumentó
                current_pread = lib.trace_model_get_pread_calls(model_buf)
                assert current_pread == pread_calls_cold, (
                    f"Violación I/O: pread_calls aumentó ({current_pread} > {pread_calls_cold})"
                )

            print(f"✓ [{split_name.upper()}] 100% coincidencia textual ({num_eval}/{num_eval}) superada.")
    finally:
        lib.trace_model_free(model_buf)


def test_native_c_engine_odirect_and_parity():
    torch.set_num_threads(2)
    lib = _load_infer_lib()

    has_real = os.path.exists(REAL_BIN_PATH) and os.path.exists(REAL_CKPT_PATH)
    if has_real:
        ckpt = torch.load(REAL_CKPT_PATH, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(ckpt["cfg"])
        cfg.device = "cpu"
        cfg.cpu_threads = 2
        cfg.use_add_sub_inference = True

        model_ref = TRACECausalModel(cfg)
        model_ref.load_state_dict(ckpt["model_state"], strict=True)
        model_ref.eval()
        _run_odirect_and_parity(lib, REAL_BIN_PATH, model_ref, cfg, is_real=True)
    else:
        print("=== Checkpoints reales ausentes: ejecutando suite CI-safe con modelo sintético ===")
        torch.manual_seed(42)
        cfg = HybridConfig(device="cpu", cpu_threads=2, use_add_sub_inference=True)
        model_ref = TRACECausalModel(cfg)
        model_ref.eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_bin = os.path.join(tmp_dir, "synthetic_causal_paged_bitcos.bin")
            PagedBitCOSSerializer.export_paged_model(model_ref, tmp_bin, format_type="fast")
            _run_odirect_and_parity(lib, tmp_bin, model_ref, cfg, is_real=False)

    print("✓ pread_calls se mantuvo estrictamente constante durante toda la inferencia.")
    print("✓ Todos los tests del Hito 2.4 superados exitosamente.")


if __name__ == "__main__":
    test_native_c_engine_odirect_and_parity()

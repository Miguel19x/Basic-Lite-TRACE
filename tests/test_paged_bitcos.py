# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Test de verificación para Paged BitCOS v1 (Hito 2.1):
1. Round-trip exacto de empaquetado/desempaquetado Fast alineado a 32 pesos.
2. Verificación estricta de 0 tensores FP32 de pesos en RAM.
3. Inferencia funcional sobre modelo cargado desde páginas binarias (por defecto y add_sub) con 0 FP32 persistentes.
"""
import os
import tempfile

import numpy as np
import torch

from config import HybridConfig
from trace.layers.bitlinear import BitLinear
from trace.models.causal import TRACECausalModel
from trace.ops.paged_kernel import unpack_page_to_dense
from trace.serialization.paged_bitcos import (
    PagedBitCOSReader,
    PagedBitCOSSerializer,
    count_fp32_weights_in_bitlinear,
)


def test_format_alignment_roundtrip():
    """Verifica que matrices con columnas arbitrarias y múltiplos de 32 se recuperen exactamente en compact y fast."""
    torch.manual_seed(42)
    expected_scale = 0.05
    for format_type in ("compact", "fast"):
        for cols in (192, 384, 50, 100):
            rows = 16
            raw = torch.randint(-1, 2, (rows, cols), dtype=torch.int8).numpy()
            page_bytes = PagedBitCOSSerializer.pack_ternary_page(
                row_start=0,
                out_features=rows,
                in_features=cols,
                scale_w=expected_scale,
                q_w_chunk=raw,
                page_size=4096,
                format_type=format_type,
            )
            r_s, r_cnt, c_cnt, scale_w, codes = unpack_page_to_dense(page_bytes)
            assert r_s == 0 and r_cnt == rows and c_cnt == cols
            assert abs(scale_w - expected_scale) < 1e-7, (
                f"scale_w discrepante en format={format_type}, cols={cols}: {scale_w} != {expected_scale}"
            )
            assert np.array_equal(codes.numpy(), raw), (
                f"Discrepancia en format={format_type}, cols={cols}"
            )


def test_zero_fp32_weights_in_ram():
    """Verifica que load_into_model vacíe cached_w_eff y weight en BitLinear en compact y fast."""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        for format_type in ("compact", "fast"):
            bin_path = os.path.join(tmpdir, f"test_paged_{format_type}.bin")
            PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type=format_type)

            reader = PagedBitCOSReader(bin_path)
            try:
                fresh_model = TRACECausalModel(cfg)
                reader.load_into_model(fresh_model)

                # Prueba negativa: page_id < 0 y page_id fuera de rango
                for bad_id in (-1, reader.total_pages, reader.total_pages + 10):
                    raised = False
                    try:
                        reader.read_page_bytes(bad_id)
                    except IndexError:
                        raised = True
                    assert raised, f"read_page_bytes({bad_id}) debía lanzar IndexError."
            finally:
                reader.close()

            for name, module in fresh_model.named_modules():
                if isinstance(module, BitLinear):
                    assert module.weight.numel() == 0, f"module.weight no vacío en {name}: {module.weight.shape}"
                    assert module.cached_w_eff.numel() == 0, (
                        f"cached_w_eff no vacío en {name}: {module.cached_w_eff.shape}"
                    )
                    assert module.int8_codes.numel() > 0, f"int8_codes vacío en {name}"


def test_forward_pass_loaded_bitcos_zero_fp32():
    """Ejecuta forward pass real en modelo cargado desde Paged BitCOS comprobando paridad de logits y 0 FP32."""
    torch.manual_seed(123)
    torch.set_num_threads(2)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg)
    model.eval()

    x = torch.randint(0, 256, (1, 32), dtype=torch.long)
    t_noise = torch.ones(1)

    with torch.no_grad():
        orig_logits_default = model.forward_backbone(x, t_noise, stage_id=0)
    model.cfg.use_add_sub_inference = True
    with torch.no_grad():
        orig_logits_addsub = model.forward_backbone(x, t_noise, stage_id=0)
    model.cfg.use_add_sub_inference = False

    with tempfile.TemporaryDirectory() as tmpdir:
        for format_type in ("compact", "fast"):
            bin_path = os.path.join(tmpdir, f"test_paged_fwd_{format_type}.bin")
            PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type=format_type)

            # 1. Modo por defecto (integer/packed inference)
            reader = PagedBitCOSReader(bin_path)
            try:
                fresh_model = TRACECausalModel(cfg)
                reader.load_into_model(fresh_model)
            finally:
                reader.close()
            fresh_model.eval()

            assert count_fp32_weights_in_bitlinear(fresh_model) == 0

            with torch.no_grad():
                logits_default = fresh_model.forward_backbone(x, t_noise, stage_id=0)

            assert torch.isfinite(logits_default).all(), "Logits contienen NaN o Inf en modo por defecto."
            assert count_fp32_weights_in_bitlinear(fresh_model) == 0, (
                "Se generaron pesos FP32 tras forward en modo por defecto."
            )
            diff_default = float((orig_logits_default - logits_default).abs().max().item())
            assert diff_default < 1e-4, (
                f"Paridad default fallida en format={format_type}: max_diff={diff_default:.6e}"
            )

            # 2. Modo aditivo/sustractivo (use_add_sub_inference=True)
            fresh_model.cfg.use_add_sub_inference = True
            with torch.no_grad():
                logits_addsub = fresh_model.forward_backbone(x, t_noise, stage_id=0)

            assert torch.isfinite(logits_addsub).all(), "Logits contienen NaN o Inf en modo add_sub."
            assert count_fp32_weights_in_bitlinear(fresh_model) == 0, (
                "Se generaron pesos FP32 tras forward en modo add_sub."
            )
            diff_addsub = float((orig_logits_addsub - logits_addsub).abs().max().item())
            assert diff_addsub < 1e-4, (
                f"Paridad add_sub fallida en format={format_type}: max_diff={diff_addsub:.6e}"
            )


if __name__ == "__main__":
    test_format_alignment_roundtrip()
    print("✓ Test 1: Round-trip de alineación compact/fast con scale_w superado.")
    test_zero_fp32_weights_in_ram()
    print("✓ Test 2: Contrato de 0 pesos FP32 en RAM y pruebas negativas de page_id superado.")
    test_forward_pass_loaded_bitcos_zero_fp32()
    print("✓ Test 3: Forward pass funcional con paridad de logits y 0 FP32 persistentes superado.")
    print("✓ Todos los tests unitarios de Paged BitCOS pasaron exitosamente.")

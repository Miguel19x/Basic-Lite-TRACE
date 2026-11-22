"""
Test de verificación para Paged BitCOS v1 (Hito 2.1):
1. Round-trip exacto de empaquetado/desempaquetado Fast alineado a 32 pesos.
2. Verificación estricta de 0 tensores FP32 de pesos en RAM.
3. Inferencia funcional sobre modelo cargado desde páginas binarias (por defecto y add_sub) con 0 FP32 persistentes.
"""
import os
import tempfile
from trace.layers.bitlinear import BitLinear
from trace.models.causal import TRACECausalModel
from trace.ops.paged_kernel import unpack_page_to_dense
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer

import numpy as np
import torch
import torch.nn as nn

from config import HybridConfig


def count_fp32_weights_in_bitlinear(model: nn.Module) -> int:
    total_fp32 = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            if module.weight is not None and module.weight.dtype == torch.float32:
                total_fp32 += module.weight.numel()
            if hasattr(module, "cached_w_eff") and module.cached_w_eff is not None:
                if module.cached_w_eff.numel() > 0 and module.cached_w_eff.dtype == torch.float32:
                    total_fp32 += module.cached_w_eff.numel()
    return total_fp32


def test_fast_format_32_weight_alignment_roundtrip():
    """Verifica que matrices con columnas arbitrarias y múltiplos de 32 se recuperen exactamente."""
    torch.manual_seed(42)
    for cols in (192, 384, 50, 100):
        rows = 16
        raw = torch.randint(-1, 2, (rows, cols), dtype=torch.int8).numpy()
        page_bytes = PagedBitCOSSerializer.pack_ternary_page(
            row_start=0,
            out_features=rows,
            in_features=cols,
            scale_w=0.05,
            q_w_chunk=raw,
            page_size=4096,
            format_type="fast",
        )
        r_s, r_cnt, c_cnt, scale_w, codes = unpack_page_to_dense(page_bytes)
        assert r_s == 0 and r_cnt == rows and c_cnt == cols
        assert np.array_equal(codes.numpy(), raw), f"Discrepancia en cols={cols}"


def test_zero_fp32_weights_in_ram():
    """Verifica que load_into_model vacíe cached_w_eff y weight en BitLinear."""
    torch.set_num_threads(2)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test_paged.bin")
        PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type="fast")

        reader = PagedBitCOSReader(bin_path)
        fresh_model = TRACECausalModel(cfg)
        reader.load_into_model(fresh_model)
        reader.close()

        for name, module in fresh_model.named_modules():
            if isinstance(module, BitLinear):
                assert module.weight.numel() == 0, f"module.weight no vacío en {name}: {module.weight.shape}"
                assert module.cached_w_eff.numel() == 0, f"cached_w_eff no vacío en {name}: {module.cached_w_eff.shape}"
                assert module.int8_codes.numel() > 0, f"int8_codes vacío en {name}"


def test_forward_pass_loaded_bitcos_zero_fp32():
    """Ejecuta forward pass real en modelo cargado desde Paged BitCOS comprobando finitud y 0 FP32 residuales."""
    torch.set_num_threads(2)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test_paged_fwd.bin")
        PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type="fast")

        # 1. Modo por defecto (integer/packed inference)
        reader = PagedBitCOSReader(bin_path)
        fresh_model = TRACECausalModel(cfg)
        reader.load_into_model(fresh_model)
        reader.close()
        fresh_model.eval()

        assert count_fp32_weights_in_bitlinear(fresh_model) == 0

        # Entrada de prueba de 32 bytes (2 bloques de block_size=16)
        x = torch.randint(0, 256, (1, 32), dtype=torch.long)
        t_noise = torch.ones(1)

        with torch.no_grad():
            logits_default = fresh_model.forward_backbone(x, t_noise, stage_id=0)

        assert torch.isfinite(logits_default).all(), "Logits contienen NaN o Inf en modo por defecto."
        assert count_fp32_weights_in_bitlinear(fresh_model) == 0, "Se generaron pesos FP32 tras forward en modo por defecto."

        # 2. Modo aditivo/sustractivo (use_add_sub_inference=True)
        fresh_model.cfg.use_add_sub_inference = True
        with torch.no_grad():
            logits_addsub = fresh_model.forward_backbone(x, t_noise, stage_id=0)

        assert torch.isfinite(logits_addsub).all(), "Logits contienen NaN o Inf en modo add_sub."
        assert count_fp32_weights_in_bitlinear(fresh_model) == 0, "Se generaron pesos FP32 tras forward en modo add_sub."


if __name__ == "__main__":
    test_fast_format_32_weight_alignment_roundtrip()
    print("✓ Test 1: Round-trip de alineación Fast a 32 pesos superado.")
    test_zero_fp32_weights_in_ram()
    print("✓ Test 2: Contrato de 0 pesos FP32 en RAM superado.")
    test_forward_pass_loaded_bitcos_zero_fp32()
    print("✓ Test 3: Forward pass funcional con 0 FP32 persistentes superado.")
    print("✓ Todos los tests unitarios de Paged BitCOS pasaron exitosamente.")

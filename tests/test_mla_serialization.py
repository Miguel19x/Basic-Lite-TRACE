# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Validación de Serialización Paged BitCOS v1.1 para TRACELanguageModel.
Certifica:
  1. Paridad numérica de logits entre modelo original y recargado en los 4 modos:
     - Estándar (< 1e-6)
     - Absorción completa (< 1e-6)
     - Caché INT8 (< 1e-6)
     - Add-sub exacto (0.0)
     - Generación autorregresiva greedy idéntica.
  2. Contrato de 0 pesos FP32 en RAM tras recarga en BitLinear.
  3. Tied embeddings estrictos: "lm_head.weight" no presente en el binario y ligado en memoria.
  4. Metadatos de arquitectura ("trace-lm-v1") y configuración recuperables por el lector.
  5. Alineación física a 4096 bytes en disco.
"""

import os
import tempfile
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.serialization.paged_bitcos import (
    PagedBitCOSReader,
    PagedBitCOSSerializer,
    count_fp32_weights_in_bitlinear,
)

import torch


def test_roundtrip_and_parity():
    """Valida serialización, metadatos y paridad matemática en los 4 modos de inferencia."""
    torch.manual_seed(42)
    cfg = TRACELMConfig(
        max_seq_len=64,
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

    input_ids = torch.tensor([[10, 45, 120, 258, 80, 99]], dtype=torch.long)

    # 1. Forward original en 4 modos
    with torch.no_grad():
        orig_logits, _, _ = model(input_ids)
        orig_abs_logits, _, _ = model(input_ids, use_weight_absorption=True)
        orig_int8_logits, _, _ = model(input_ids, quantize_kv_int8=True)
        orig_addsub_logits, _, _ = model(input_ids, add_sub_mode=True)
        orig_gen = model.generate(input_ids, max_new_tokens=8, temperature=0.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_path = os.path.join(tmp_dir, "test_lm.bin")

        # 2. Exportación a Paged BitCOS
        summary = PagedBitCOSSerializer.export_paged_model(
            model=model,
            output_path=bin_path,
            format_type="fast",
        )
        assert os.path.exists(bin_path)
        assert summary["architecture"] == "trace-lm-v1"

        # 3. Inspección con PagedBitCOSReader
        reader = PagedBitCOSReader(bin_path)

        # Validación explícita de metadatos y ausencia de lm_head.weight duplicado
        assert reader.get_architecture() == "trace-lm-v1", (
            f"Arquitectura incorrecta: {reader.get_architecture()}"
        )
        config_read = reader.get_model_config()
        assert config_read is not None, "get_model_config() no devolvió configuración."
        assert config_read["d_model"] == cfg.d_model, (
            f"d_model discrepante: {config_read['d_model']} vs {cfg.d_model}"
        )
        assert "lm_head.weight" not in reader.tensors_map, (
            "Violación de invariante: 'lm_head.weight' se serializó por duplicado a pesar de estar ligado."
        )

        # 4. Recarga en modelo nuevo
        fresh_model = TRACELanguageModel(cfg)
        loaded_count = reader.load_into_model(fresh_model)
        fresh_model.eval()
        reader.close()

        assert loaded_count > 0

        # Verificación de contrato 0 FP32 en BitLinear
        fp32_count = count_fp32_weights_in_bitlinear(fresh_model)
        assert fp32_count == 0, f"Se detectaron {fp32_count} pesos FP32 en BitLinear tras recarga."

        # Verificación de tied embeddings intactos
        assert fresh_model.lm_head.weight is fresh_model.tok_emb.weight, (
            "lm_head.weight no comparte referencia con tok_emb.weight tras recarga."
        )

        # 5. Forward recargado en 4 modos y paridad estricta
        with torch.no_grad():
            loaded_logits, _, _ = fresh_model(input_ids)
            loaded_abs_logits, _, _ = fresh_model(input_ids, use_weight_absorption=True)
            loaded_int8_logits, _, _ = fresh_model(input_ids, quantize_kv_int8=True)
            loaded_addsub_logits, _, _ = fresh_model(input_ids, add_sub_mode=True)
            loaded_gen = fresh_model.generate(input_ids, max_new_tokens=8, temperature=0.0)

        diff_std = torch.max(torch.abs(orig_logits - loaded_logits)).item()
        diff_abs = torch.max(torch.abs(orig_abs_logits - loaded_abs_logits)).item()
        diff_int8 = torch.max(torch.abs(orig_int8_logits - loaded_int8_logits)).item()
        diff_addsub = torch.max(torch.abs(orig_addsub_logits - loaded_addsub_logits)).item()

        assert diff_std < 1e-6, f"Discrepancia en modo estándar: {diff_std:.6e}"
        assert diff_abs < 1e-6, f"Discrepancia en modo absorción: {diff_abs:.6e}"
        assert diff_int8 < 1e-6, f"Discrepancia en modo INT8 KV: {diff_int8:.6e}"
        assert diff_addsub == 0.0, f"Discrepancia en modo add-sub: {diff_addsub}"
        assert torch.equal(orig_gen, loaded_gen), "Generación greedy difiere tras recarga."

    print("✓ Paridad en los 4 modos y metadatos de TRACELanguageModel superada.")


def test_export_audit_25m_preset():
    """Valida empaquetado, alineación física a 4096B y 0 FP32 en preset_25m."""
    cfg = TRACELMConfig.preset_25m()
    model = TRACELanguageModel(cfg)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_path = os.path.join(tmp_dir, "trace_lm_25m.bin")
        summary = PagedBitCOSSerializer.export_paged_model(
            model=model,
            output_path=bin_path,
            format_type="fast",
        )
        assert summary["architecture"] == "trace-lm-v1"

        reader = PagedBitCOSReader(bin_path)
        breakdown = reader.get_accounting_breakdown()

        # Validación de alineación física estricta
        page_size = reader.page_size
        assert page_size == 4096
        assert reader.pages_offset % page_size == 0, (
            f"pages_offset ({reader.pages_offset}) no es múltiplo de {page_size}"
        )
        assert reader.total_pages * page_size == (reader.total_size - reader.pages_offset), (
            "El tamaño de la región de páginas no coincide exactamente con total_pages * 4096"
        )

        # Validación de recarga y 0 FP32
        fresh_model = TRACELanguageModel(cfg)
        reader.load_into_model(fresh_model)
        fresh_model.eval()
        reader.close()

        fp32_count = count_fp32_weights_in_bitlinear(fresh_model)
        assert fp32_count == 0, f"Violación de contrato: {fp32_count} pesos FP32 en BitLinear."
        assert fresh_model.lm_head.weight is fresh_model.tok_emb.weight

        # Verificación contable
        res_bytes = breakdown["resident_fp32_params_bytes"]
        page_bytes = breakdown["ternary_pages_bytes"]
        assert res_bytes > 0
        assert page_bytes > 0
        # En el preset 25M, los pesos ternarios en páginas de 2-bit deben ocupar menos de 8 MB
        assert page_bytes < 8 * 1024 * 1024, f"Páginas ternarias superan presupuesto: {page_bytes} bytes"

    print(f"✓ Auditoría Preset 25M superada: {summary['total_pages']} páginas ({page_bytes / 1024:.1f} KB), 0 FP32.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_roundtrip_and_parity()
    test_export_audit_25m_preset()
    print("✓ Todos los tests de serialización Paged BitCOS v1.1 pasaron exitosamente.")

#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Script canónico de exportación Paged BitCOS v1.1 — TRACELanguageModel.
Exporta pesos de TRACELanguageModel a binario alineado a 4096 bytes (O_DIRECT ready).
Conserva W_uk y W_uv en 2 bits originales (sin pre-multiplicar a FP32).
Audita y certifica:
  1. 0 pesos FP32 en RAM para las 9 * n_layers capas BitLinear.
  2. model.lm_head.weight is model.tok_emb.weight (tied embeddings no duplicados).
  3. Alineación estricta a 4096 bytes y páginas exactas de 4096 bytes.
"""

import argparse
import hashlib
import os
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.serialization.paged_bitcos import (
    PagedBitCOSReader,
    PagedBitCOSSerializer,
    count_fp32_weights_in_bitlinear,
)
from typing import Any

import torch


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_lm_checkpoint(
    checkpoint_path: str | None,
    output_path: str,
    preset: str = "25m",
    format_type: str = "fast",
    page_size: int = 4096,
    validate_load: bool = True,
) -> dict[str, Any]:
    torch.set_num_threads(2)

    print("=" * 80)
    print("Basic-Lite TRACE — Exportador Paged BitCOS v1.1 (TRACELanguageModel)")
    print(f"Preset / Checkpoint: {checkpoint_path if checkpoint_path else f'Preset {preset}'}")
    print(f"Binario destino:     {output_path}")
    print(f"Formato páginas:     {format_type} (alineación a 32 pesos/fila)")
    print(f"Tamaño de página:    {page_size} bytes (límite físico OS/SSD)")
    print("=" * 80)

    # 1. Resolver configuración y pesos
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "cfg" in ckpt:
            cfg = TRACELMConfig(**ckpt["cfg"]) if isinstance(ckpt["cfg"], dict) else ckpt["cfg"]
        else:
            if preset == "25m":
                cfg = TRACELMConfig.preset_25m()
            elif preset == "25m-bi":
                cfg = TRACELMConfig.preset_25m_bilingual()
            elif preset == "45m-bi":
                cfg = TRACELMConfig.preset_45m_bilingual()
            else:
                cfg = TRACELMConfig()
        cfg.quantize_weights_enabled = True
        model = TRACELanguageModel(cfg)
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        assert len(missing) == 0 and len(unexpected) == 0, f"Discrepancia state_dict: {missing=}, {unexpected=}"
        print(f"Pesos cargados exitosamente desde {checkpoint_path}")
    else:
        if preset == "25m":
            cfg = TRACELMConfig.preset_25m()
        elif preset == "25m-bi":
            cfg = TRACELMConfig.preset_25m_bilingual()
        elif preset == "45m-bi":
            cfg = TRACELMConfig.preset_45m_bilingual()
        else:
            cfg = TRACELMConfig()
        cfg.quantize_weights_enabled = True
        model = TRACELanguageModel(cfg)

    model.eval()

    # 2. Exportar binario Paged BitCOS
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    summary = PagedBitCOSSerializer.export_paged_model(
        model=model,
        output_path=output_path,
        page_size=page_size,
        format_type=format_type,
    )
    sha256 = compute_sha256(output_path)
    summary["sha256"] = sha256

    print("\n--- RESUMEN DE EXPORTACIÓN ---")
    print(f"Páginas ternarias:        {summary['total_pages']} ({summary['total_pages'] * page_size / 1024:.1f} KB)")
    print(f"Tensores empaquetados:    {summary['tensors_count']}")
    print(f"Tamaño total del binario: {summary['total_bytes'] / 1024:.2f} KB ({summary['total_bytes']} bytes)")
    print(f"SHA-256:                  {sha256}")

    # 3. Auditoría de memoria y validación de recarga
    if validate_load:
        print("\n--- AUDITORÍA DE MEMORIA Y RECARGA (PUERTA DE SALIDA HITO 3.3) ---")
        reader = PagedBitCOSReader(output_path)
        breakdown = reader.get_accounting_breakdown()
        print(f"  Parámetros residentes FP32:             {breakdown['resident_fp32_params_count']} ({breakdown['resident_fp32_params_bytes'] / 1024:.2f} KB)")
        print(f"  Páginas ternarias empaquetadas (2-bit): {breakdown['ternary_pages_count']} ({breakdown['ternary_pages_bytes'] / 1024:.2f} KB)")

        fresh_model = TRACELanguageModel(cfg)
        loaded = reader.load_into_model(fresh_model)
        fresh_model.eval()

        # Comprobación 0 pesos FP32 en BitLinear
        fp32_count = count_fp32_weights_in_bitlinear(fresh_model)
        print(f"  Módulos recargados desde binario:       {loaded}")
        print(f"  Tensores FP32 en matrices BitLinear:    {fp32_count} elementos")
        if fp32_count != 0:
            reader.close()
            raise AssertionError(f"Violación de invariante: {fp32_count} pesos FP32 detectados en RAM.")

        # Comprobación tied embeddings
        if fresh_model.lm_head.weight is not fresh_model.tok_emb.weight:
            reader.close()
            raise AssertionError("Violación de tied embeddings: lm_head.weight no es el mismo objeto que tok_emb.weight.")
        print("  [OK] Tied embeddings preservados (lm_head.weight is tok_emb.weight)")

        # Comprobación de alineación de páginas y múltiplos de 4096
        if reader.pages_offset % page_size != 0:
            reader.close()
            raise AssertionError(f"Alineación física violada: pages_offset={reader.pages_offset} no es múltiplo de {page_size}")
        if reader.total_pages * page_size != (reader.total_size - reader.pages_offset):
            reader.close()
            raise AssertionError("Violación de integridad de páginas: el tamaño del segmento de páginas no coincide con total_pages * page_size")
        print(f"  [OK] Alineación física validada: offset={reader.pages_offset} alineado a {page_size}B, 0 desperdicio.")

        # Paridad funcional
        dummy_x = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        with torch.no_grad():
            y_orig, _, _ = model(dummy_x)
            y_load, _, _ = fresh_model(dummy_x)
            diff = torch.max(torch.abs(y_orig - y_load)).item()
        print(f"  [OK] Paridad numérica forward pass: max_diff = {diff:.6e}")
        reader.close()

    print("\n" + "=" * 80)
    print("Exportación Paged BitCOS v1.1 finalizada con éxito.")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportador Paged BitCOS v1.1 — TRACELanguageModel")
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint .pt opcional")
    parser.add_argument("--output", type=str, default="checkpoints/trace_lm_25m_paged_bitcos.bin", help="Ruta de salida .bin")
    parser.add_argument("--preset", type=str, default="25m", choices=["25m", "25m-bi", "45m-bi", "compact"], help="Preset de arquitectura")
    parser.add_argument("--format", type=str, default="fast", choices=["fast", "compact"], help="Formato de página")
    parser.add_argument("--page_size", type=int, default=4096, help="Tamaño de página en bytes")
    parser.add_argument("--no_validate", action="store_true", help="Omitir auditoría de recarga")
    args = parser.parse_args()

    export_lm_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        preset=args.preset,
        format_type=args.format,
        page_size=args.page_size,
        validate_load=(not args.no_validate),
    )

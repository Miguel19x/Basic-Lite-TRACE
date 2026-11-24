#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Script canónico de exportación Paged BitCOS v1 — Basic-Lite TRACE.
Exporta checkpoints de Fase 1 a binario alineado a 4096 bytes (O_DIRECT ready).
Alinea filas en formato Fast a múltiplos de 32 pesos (1 vector AVX2).
Audita y certifica la condición de '0 pesos FP32 en RAM' para BitLinear.
"""

import argparse
import hashlib
import os
from trace.layers.bitlinear import BitLinear
from trace.models.causal import TRACECausalModel
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer

import torch
import torch.nn as nn


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_fp32_weights_in_bitlinear(model: nn.Module) -> int:
    """Cuenta elementos FP32 en los parámetros y cachés de BitLinear."""
    total_fp32 = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            if module.weight is not None and module.weight.dtype == torch.float32:
                total_fp32 += module.weight.numel()
            if hasattr(module, "cached_w_eff") and module.cached_w_eff is not None:
                if module.cached_w_eff.numel() > 0 and module.cached_w_eff.dtype == torch.float32:
                    total_fp32 += module.cached_w_eff.numel()
    return total_fp32


def export_checkpoint(
    checkpoint_path: str,
    output_path: str,
    format_type: str = "fast",
    page_size: int = 4096,
    validate_load: bool = True,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    # Fijar obligatoriamente 2 hilos físicos para el hardware Intel Core i3-1115G4
    torch.set_num_threads(2)

    print("=" * 80)
    print("Basic-Lite TRACE — Exportador Paged BitCOS v1")
    print(f"Checkpoint origen: {checkpoint_path}")
    print(f"Binario destino:   {output_path}")
    print(f"Formato páginas:   {format_type} (alineación a 32 pesos/fila)")
    print(f"Tamaño de página:  {page_size} bytes (límite físico OS/SSD)")
    print("=" * 80)

    # 1. Cargar obligatoriamente la configuración desde ckpt["cfg"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "cfg" not in ckpt:
        raise KeyError("El checkpoint no contiene el campo obligatorio 'cfg'.")
    cfg = ckpt["cfg"]
    cfg.device = "cpu"
    cfg.cpu_threads = 2
    cfg.paged_bitcos_format = format_type
    cfg.paged_bitcos_page_size = page_size

    # 2. Instanciar modelo canónico y cargar pesos exactos
    model = TRACECausalModel(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f"Discrepancia en state_dict: missing={missing}, unexpected={unexpected}"
    model.eval()

    # 3. Exportar binario Paged BitCOS v1
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

    # 4. Auditoría estricta de memoria (0 FP32 en RAM) y validación de recarga
    if validate_load:
        print("\n--- AUDITORÍA DE MEMORIA Y RECARGA (PUERTA DE SALIDA FASE 2) ---")
        reader = PagedBitCOSReader(output_path)
        breakdown = reader.get_accounting_breakdown()
        print(f"  Parámetros residentes FP32 (emb, normas, sesgos, cabezal): {breakdown['resident_fp32_params_count']} ({breakdown['resident_fp32_params_bytes'] / 1024:.2f} KB)")
        print(f"  Páginas ternarias empaquetadas (2-bit):                    {breakdown['ternary_pages_count']} ({breakdown['ternary_pages_bytes'] / 1024:.2f} KB)")

        fresh_model = TRACECausalModel(cfg)
        loaded = reader.load_into_model(fresh_model)
        reader.close()

        fp32_count = count_fp32_weights_in_bitlinear(fresh_model)
        print(f"  Módulos recargados desde binario: {loaded}")
        print(f"  Tensores FP32 en matrices BitLinear: {fp32_count} elementos")

        if fp32_count != 0:
            raise AssertionError(f"Incumplimiento de Puerta de Salida: {fp32_count} pesos FP32 detectados en RAM.")
        print("  [OK] Puerta de salida validada: 0 pesos FP32 en RAM durante inferencia paginada.")

    print("\n" + "=" * 80)
    print("Exportación Paged BitCOS v1 finalizada con éxito.")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportador Paged BitCOS v1 — Basic-Lite TRACE")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/trace_s456_fase1_bitnet_last.pt", help="Ruta al checkpoint .pt")
    parser.add_argument("--output", type=str, default="checkpoints/trace_s456_fase1_paged_bitcos.bin", help="Ruta de salida .bin")
    parser.add_argument("--format", type=str, default="fast", choices=["fast", "compact"], help="Formato de página")
    parser.add_argument("--page_size", type=int, default=4096, help="Tamaño de página en bytes")
    parser.add_argument("--no_validate", action="store_true", help="Omitir auditoría de recarga")
    args = parser.parse_args()

    export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        format_type=args.format,
        page_size=args.page_size,
        validate_load=(not args.no_validate),
    )

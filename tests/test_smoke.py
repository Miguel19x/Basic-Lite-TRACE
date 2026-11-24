# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Test de humo para verificar compilación y carga del kernel C y el forward pass de TRACECausalModel.
"""
import ctypes
import os
from pathlib import Path

import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from trace.models.causal import TRACECausalModel

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_kernel_c_compilation_and_load():
    so_path = str(REPO_ROOT / "libpaged_gemm.so")
    assert os.path.exists(so_path), f"El binario {so_path} no existe. Compila con 'make' primero."
    lib = ctypes.CDLL(so_path)
    required_symbols = (
        "paged_direct_matmul_fast_c",
        "paged_direct_matmul_fast_c_multi",
        "paged_lowrank_matmul_row_c",
        "ternary_addsub_cpu_kernel",
        "paged_absorb_uk_f32",
        "paged_absorb_uv_f32",
        "trace_dot_f32",
        "trace_axpy_f32",
        "trace_cpu_active_isa",
        "trace_cpu_set_isa",
    )
    for sym in required_symbols:
        assert hasattr(lib, sym), f"Símbolo '{sym}' no hallado en el kernel {so_path}."

    lib.trace_cpu_active_isa.argtypes = []
    lib.trace_cpu_active_isa.restype = ctypes.c_char_p
    active_isa = lib.trace_cpu_active_isa().decode("utf-8")
    assert active_isa in ("scalar", "avx2"), f"ISA inesperada reportada por trace_cpu_active_isa: '{active_isa}'"


def test_trace_causal_model_forward():
    torch.manual_seed(42)
    # Configuración de producción canónica en CPU con 1 recursión para máxima velocidad
    cfg = HybridConfig(
        device="cpu",
        num_recursions=1,
        adaptive_recursion=False,
    )

    model = TRACECausalModel(cfg)
    model.eval()

    # Muestra real de la partición dev
    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b, _, _, _ = dataset.samples[0]

    # Verificar logits finitos en forward_backbone
    seq = [cfg.bos_token_id] + list(prompt_b) + [cfg.eos_token_id]
    seq = seq + [cfg.pad_token_id] * max(0, cfg.max_seq_len - len(seq))
    x = torch.tensor([seq[: cfg.max_seq_len]], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward_backbone(x, torch.ones(1), stage_id=0)
        assert torch.isfinite(logits).all(), "Logits de forward_backbone contienen NaN o Inf."
        gen_str, meta = model.generate_causal_two_step(prompt_b)

    assert isinstance(gen_str, str), "La salida debe ser una cadena de texto."
    assert len(gen_str) == 35, f"Longitud de salida inesperada (esperado 35 bytes de ancho fijo): {len(gen_str)}"
    assert "watt:" in gen_str, "La salida debe incluir la etiqueta de campo 'watt:'."
    assert "ms:" in gen_str, "La salida debe incluir la etiqueta de campo 'ms:'."
    assert "pw:" in gen_str, "La salida debe incluir la etiqueta de campo 'pw:'."
    assert "pms:" in gen_str, "La salida debe incluir la etiqueta de campo 'pms:'."
    assert "watt" in meta and "ms" in meta, "El diccionario meta debe registrar 'watt' y 'ms'."


if __name__ == "__main__":
    test_kernel_c_compilation_and_load()
    test_trace_causal_model_forward()
    print("✓ Todos los tests de humo pasaron exitosamente.")

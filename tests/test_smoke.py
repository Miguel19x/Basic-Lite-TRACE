"""
Test de humo para verificar compilación y carga del kernel C y el forward pass de TRACECausalModel.
"""
import ctypes
import os
from trace.models.causal import TRACECausalModel

import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset


def test_kernel_c_compilation_and_load():
    so_path = os.path.abspath("libpaged_gemm.so")
    assert os.path.exists(so_path), f"El binario {so_path} no existe. Compila con 'make' primero."
    lib = ctypes.CDLL(so_path)
    assert hasattr(lib, "paged_direct_matmul_fast_c"), "Símbolo 'paged_direct_matmul_fast_c' no hallado en el kernel."


def test_trace_causal_model_forward():
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

    with torch.no_grad():
        gen_str, meta = model.generate_causal_two_step(prompt_b)

    assert isinstance(gen_str, str), "La salida debe ser una cadena de texto."
    assert "watt:" in gen_str, "La salida debe incluir la etiqueta de campo 'watt:'."
    assert "ms:" in gen_str, "La salida debe incluir la etiqueta de campo 'ms:'."
    assert "pw:" in gen_str, "La salida debe incluir la etiqueta de campo 'pw:'."
    assert "pms:" in gen_str, "La salida debe incluir la etiqueta de campo 'pms:'."
    assert "watt" in meta and "ms" in meta, "El diccionario meta debe registrar 'watt' y 'ms'."


if __name__ == "__main__":
    test_kernel_c_compilation_and_load()
    test_trace_causal_model_forward()
    print("✓ Todos los tests de humo pasaron exitosamente.")

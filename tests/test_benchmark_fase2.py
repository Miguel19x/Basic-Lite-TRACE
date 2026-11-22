"""
Basic-Lite TRACE — Benchmark Formal de Inferencia Fase 2 (Hito 2.5).

Evalúa el rendimiento end-to-end de trace_infer frente a:
  1. PyTorch AddSub W1.58A8 (TRACECausalModel con use_add_sub_inference=True)
  2. PyTorch Continuo FP32 (TRACECausalModel con use_add_sub_inference=False)

Condiciones Mandatorias:
  - Hardware: Intel Core i3-1115G4 en 2 hilos físicos (OMP_NUM_THREADS=2, torch.set_num_threads(2))
  - Dataset: 16 muestras reales (8 de dev y 8 de test-composicional)
  - Salida: Coincidencia textual frente a AddSub y FP32
  - Criterio de Salida: Speedup >= 3.0x frente a la referencia PyTorch en CPU
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"

import copy
import ctypes
import sys
import time
import numpy as np
import torch

torch.set_num_threads(2)

from data import ProceduralCompositionalDataset
from trace.models.causal import TRACECausalModel

BIN_PATH = "checkpoints/trace_s456_fase1_paged_bitcos.bin"
PT_CKPT = "checkpoints/trace_s456_fase1_bitnet_last.pt"
SO_PATH = "./libtrace_infer.so"


def run_benchmark():
    print("=" * 78)
    print("  BASIC-LITE TRACE: BENCHMARK FORMAL DE INFERENCIA FASE 2 (HITO 2.5)")
    print("=" * 78)

    # 1. Cargar Checkpoint PyTorch
    assert os.path.exists(PT_CKPT), f"Falta {PT_CKPT}"
    ckpt = torch.load(PT_CKPT, map_location="cpu", weights_only=False)
    cfg_base = ckpt["cfg"]

    cfg_addsub = copy.deepcopy(cfg_base)
    cfg_addsub.device = "cpu"
    cfg_addsub.cpu_threads = 2
    cfg_addsub.use_add_sub_inference = True
    model_addsub = TRACECausalModel(cfg_addsub)
    model_addsub.load_state_dict(ckpt["model_state"], strict=True)
    model_addsub.eval()

    cfg_fp32 = copy.deepcopy(cfg_base)
    cfg_fp32.device = "cpu"
    cfg_fp32.cpu_threads = 2
    cfg_fp32.use_add_sub_inference = False
    model_fp32 = TRACECausalModel(cfg_fp32)
    model_fp32.load_state_dict(ckpt["model_state"], strict=True)
    model_fp32.eval()

    # 2. Cargar Motor C Autónomo
    assert os.path.exists(SO_PATH), f"Falta {SO_PATH}"
    lib = ctypes.CDLL(SO_PATH)

    lib.trace_model_sizeof.restype = ctypes.c_size_t
    model_sz = lib.trace_model_sizeof()
    model_buf = ctypes.create_string_buffer(model_sz)

    lib.trace_model_load_odirect.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.trace_model_load_odirect.restype = ctypes.c_int
    lib.trace_model_free.argtypes = [ctypes.c_void_p]
    lib.trace_model_free.restype = None


    lib.trace_generate_causal_two_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_generate_causal_two_step.restype = ctypes.c_int

    ret = lib.trace_model_load_odirect(BIN_PATH.encode("utf-8"), model_buf)
    assert ret == 0, f"Error cargando motor C: {ret}"
    print(f"  [Motor C]: Cargado exitosamente desde {BIN_PATH} (0 pesos FP32 en BitLinear)")

    # 3. Preparar Muestras Reales (8 dev + 8 test-composicional)
    dev_ds = ProceduralCompositionalDataset(cfg_base, split="dev")
    test_ds = ProceduralCompositionalDataset(cfg_base, split="test-composicional")

    samples = []
    for i in range(8):
        p, _, _, _ = dev_ds.samples[i]
        samples.append(("dev", p))
    for i in range(8):
        p, _, _, _ = test_ds.samples[i]
        samples.append(("test-comp", p))

    # 4. Warmup (3 iteraciones para descartar costes transitorios)
    print("\n  [Calentamiento]: 3 muestras para estabilizar cachés y OpenMP...")
    warmup_buf = ctypes.create_string_buffer(512)
    for _, p_bytes in samples[:3]:
        p_arr = (ctypes.c_uint8 * len(p_bytes))(*p_bytes)
        lib.trace_generate_causal_two_step(model_buf, p_arr, len(p_bytes), warmup_buf, 512)
        with torch.no_grad():
            model_addsub.generate_causal_two_step(p_bytes)
            model_fp32.generate_causal_two_step(p_bytes)


    # 5. Medición Formal
    lat_c = []
    lat_addsub = []
    lat_fp32 = []
    matches_addsub = 0
    matches_fp32 = 0
    total_eval = len(samples)

    print(f"\n  [Evaluación Comparativa sobre {total_eval} muestras reales en 2 hilos CPU]:")

    c_out_buf = ctypes.create_string_buffer(512)
    NUM_ROUNDS = 2  # 32 ejecuciones por motor

    for r in range(NUM_ROUNDS):
        for idx, (split, p_bytes) in enumerate(samples):
            # A. Native C Standalone Engine
            p_arr = (ctypes.c_uint8 * len(p_bytes))(*p_bytes)
            t0 = time.perf_counter()
            lib.trace_generate_causal_two_step(model_buf, p_arr, len(p_bytes), c_out_buf, 512)
            t1 = time.perf_counter()
            c_text = c_out_buf.value.decode("utf-8", errors="replace")
            lat_c.append((t1 - t0) * 1000.0)

            # B. PyTorch AddSub (Referencia W1.58A8)
            t0 = time.perf_counter()
            with torch.no_grad():
                pt_text_addsub, _ = model_addsub.generate_causal_two_step(p_bytes)
            t1 = time.perf_counter()
            lat_addsub.append((t1 - t0) * 1000.0)

            # C. PyTorch Continuo FP32
            t0 = time.perf_counter()
            with torch.no_grad():
                pt_text_fp32, _ = model_fp32.generate_causal_two_step(p_bytes)
            t1 = time.perf_counter()
            lat_fp32.append((t1 - t0) * 1000.0)

            if r == 0:
                m_addsub = (c_text == pt_text_addsub)
                m_fp32 = (c_text == pt_text_fp32)
                if m_addsub:
                    matches_addsub += 1
                if m_fp32:
                    matches_fp32 += 1
                if not m_addsub:
                    print(f"  [DISCREPANCIA vs AddSub] Muestra {idx} ({split}):\n    C: {repr(c_text)}\n    PT: {repr(pt_text_addsub)}")

    lib.trace_model_free(model_buf)

    # 6. Reporte Estadístico
    arr_c = np.array(lat_c)
    arr_addsub = np.array(lat_addsub)
    arr_fp32 = np.array(lat_fp32)

    mean_c, med_c, p95_c = np.mean(arr_c), np.median(arr_c), np.percentile(arr_c, 95)
    mean_addsub, med_addsub, p95_addsub = np.mean(arr_addsub), np.median(arr_addsub), np.percentile(arr_addsub, 95)
    mean_fp32, med_fp32, p95_fp32 = np.mean(arr_fp32), np.median(arr_fp32), np.percentile(arr_fp32, 95)

    speedup_addsub = mean_addsub / mean_c
    speedup_fp32 = mean_fp32 / mean_c
    throughput_c = 1000.0 / mean_c
    throughput_addsub = 1000.0 / mean_addsub

    print("\n" + "-" * 78)
    print("  RESULTADOS DE BENCHMARK (Muestras: %d, Hilos CPU: 2)" % (len(arr_c)))
    print("-" * 78)
    print(f"  1. Motor C Autónomo (trace_infer AVX2 + O_DIRECT):")
    print(f"     - Latencia Media:   {mean_c:6.2f} ms")
    print(f"     - Latencia Mediana: {med_c:6.2f} ms")
    print(f"     - Percentil 95:     {p95_c:6.2f} ms")
    print(f"     - Throughput:       {throughput_c:6.2f} muestras/seg")
    print()
    print(f"  2. PyTorch W1.58A8 AddSub (Referencia):")
    print(f"     - Latencia Media:   {mean_addsub:6.2f} ms")
    print(f"     - Latencia Mediana: {med_addsub:6.2f} ms")
    print(f"     - Percentil 95:     {p95_addsub:6.2f} ms")
    print(f"     - Throughput:       {throughput_addsub:6.2f} muestras/seg")
    print()
    print(f"  3. PyTorch Continuo FP32:")
    print(f"     - Latencia Media:   {mean_fp32:6.2f} ms")
    print(f"     - Latencia Mediana: {med_fp32:6.2f} ms")
    print(f"     - Percentil 95:     {p95_fp32:6.2f} ms")
    print("-" * 78)
    print(f"  SPEEDUP FRENTE A PYTORCH ADDSUB:  {speedup_addsub:6.2f}x")
    print(f"  SPEEDUP FRENTE A PYTORCH FP32:    {speedup_fp32:6.2f}x")
    print(f"  COINCIDENCIA TEXTUAL VS ADDSUB:   {matches_addsub}/{total_eval} ({matches_addsub/total_eval*100:.1f}%)")
    print(f"  COINCIDENCIA TEXTUAL VS FP32:     {matches_fp32}/{total_eval} ({matches_fp32/total_eval*100:.1f}%)")
    print("-" * 78)



    # Verificaciones mandatorias
    assert matches_addsub == total_eval, f"Fallo de coincidencia textual vs AddSub: {matches_addsub}/{total_eval}"
    print("  [AUDITORÍA DE EXACTITUD]: 100.0% de paridad textual verificada vs PyTorch AddSub.")

    assert speedup_addsub >= 3.0, f"Speedup insuficiente frente a PyTorch AddSub: {speedup_addsub:.2f}x < 3.0x"
    print(f"  [CRITERIO DE SALIDA HITO 2.5 SUPERADO]: Speedup {speedup_addsub:.2f}x >= 3.0x.")


if __name__ == "__main__":
    run_benchmark()

# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import struct
import time
from typing import Any

import numpy as np
import torch


def unpack_page_to_dense(page_bytes: bytes) -> tuple[int, int, int, float, torch.Tensor]:
    """
    Decodifica una página binaria Paged BitCOS a un tensor denso int8 {-1, 0, +1}.
    Retorna: (row_start, rows, cols, scale_w, codes_tensor)
    """
    header_fmt = "<IIHHfII"
    header_size = struct.calcsize(header_fmt)
    row_start, rows, cols, fmt_code, scale_w, len_p1, len_p2 = struct.unpack(
        header_fmt, page_bytes[:header_size]
    )
    offset = header_size

    total_weights = rows * cols
    if fmt_code == 0:  # compact: presencia + signos
        pres_bytes = page_bytes[offset : offset + len_p1]
        offset += len_p1
        sign_bytes = page_bytes[offset : offset + len_p2]

        presence = np.unpackbits(np.frombuffer(pres_bytes, dtype=np.uint8))[:total_weights]
        num_nonzero = int(presence.sum())
        signs = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:num_nonzero]

        reconstructed = np.zeros(total_weights, dtype=np.int8)
        reconstructed[presence == 1] = np.where(signs == 1, 1, -1).astype(np.int8)
    else:  # fast: 2 bits directos por peso (filas alineadas a múltiplos de 32 pesos)
        padded_cols = ((cols + 31) // 32) * 32
        packed_bytes = page_bytes[offset : offset + len_p1]
        packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)
        c0 = (packed_arr & 0x03).astype(np.int8) - 1
        c1 = ((packed_arr >> 2) & 0x03).astype(np.int8) - 1
        c2 = ((packed_arr >> 4) & 0x03).astype(np.int8) - 1
        c3 = ((packed_arr >> 6) & 0x03).astype(np.int8) - 1
        reconstructed = np.empty(len(packed_arr) * 4, dtype=np.int8)
        reconstructed[0::4] = c0
        reconstructed[1::4] = c1
        reconstructed[2::4] = c2
        reconstructed[3::4] = c3
        reconstructed = reconstructed.reshape(rows, padded_cols)[:, :cols].copy()

    codes_tensor = torch.from_numpy(reconstructed.reshape(rows, cols))
    return row_start, rows, cols, scale_w, codes_tensor


def paged_direct_block_matmul(
    q_x_int8: torch.Tensor,
    page_bytes: bytes,
) -> tuple[int, int, float, torch.Tensor]:
    """
    Ejecuta la multiplicación de la página empaquetada contra activaciones INT8
    sin reconstruir todo el tensor en FP32.
    Proceso:
      Activación q_x [B, in_features] (INT8)
      ↓
      Desempaquetado local inmediato por filas / bloques de página
      ↓
      Acumulador INT32
    Retorna: (row_start, rows, scale_w, acc_int32 [B, rows])
    """
    row_start, rows, cols, scale_w, codes = unpack_page_to_dense(page_bytes)
    # Matmul aditiva/entera directa sobre el bloque de página
    if q_x_int8.dim() == 1:
        q_x_int8 = q_x_int8.unsqueeze(0)

    # Multiplicación por bloque de página confinado a 'rows' filas
    # En CPU, operar en bloques pequeños cabe en caché L1/L2
    acc_int32 = torch.matmul(q_x_int8.to(torch.int32), codes.to(torch.int32).t())
    return row_start, rows, scale_w, acc_int32


def benchmark_packed_vs_dense(
    batch_size: int = 1,
    in_features: int = 192,
    out_features: int = 384,
    page_size: int = 4096,
    num_runs: int = 50,
) -> dict[str, Any]:
    """
    Compara de forma estricta:
    1. Reconstrucción completa en FP32 + GEMM convencional
    2. Ejecución directa por bloques de páginas empaquetadas
    3. Formato Compact vs. Formato Fast
    Responde a la pregunta 1 del auditor:
    «¿El resultado coincide y cuánto cuesta ejecutarla sin expansión completa?»
    """
    from trace.serialization.paged_bitcos import PagedBitCOSSerializer

    # Generar pesos ternarios sintéticos {-1, 0, +1}
    torch.manual_seed(42)
    raw_weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
    scale_w = 0.05

    # Generar activaciones de prueba
    x_fp32 = torch.randn(batch_size, in_features)
    scale_x = float(x_fp32.abs().max().item()) + 1e-6
    q_x_int8 = torch.clamp(torch.round(x_fp32 * (127.0 / scale_x)), -127, 127).to(torch.int8)

    # Serializar páginas en ambos formatos
    pages_compact = []
    pages_fast = []
    rows_per_page = max(1, (page_size - 64) // max(1, int(in_features * 0.25) + 4))

    for r_start in range(0, out_features, rows_per_page):
        r_end = min(out_features, r_start + rows_per_page)
        chunk = raw_weights[r_start:r_end, :].numpy()

        p_c = PagedBitCOSSerializer.pack_ternary_page(
            r_start, out_features, in_features, scale_w, chunk, page_size, "compact"
        )
        p_f = PagedBitCOSSerializer.pack_ternary_page(
            r_start, out_features, in_features, scale_w, chunk, page_size, "fast"
        )
        pages_compact.append(p_c)
        pages_fast.append(p_f)

    # 1. Baseline: Reconstrucción completa en FP32 + Matmul Convencional
    t0 = time.perf_counter()
    for _ in range(num_runs):
        # Descomprimir tensor entero en RAM
        full_recon = torch.empty(out_features, in_features, dtype=torch.float32)
        for p in pages_compact:
            r_s, r_cnt, _, sc, codes = unpack_page_to_dense(p)
            full_recon[r_s : r_s + r_cnt, :] = codes.to(torch.float32) * sc
        out_fp32 = torch.matmul(x_fp32, full_recon.t())
    lat_fp32_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # 2. Ejecución Paginada Directa (Compact)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_paged = torch.empty(batch_size, out_features, dtype=torch.float32)
        for p in pages_compact:
            r_s, r_cnt, sc, acc_i32 = paged_direct_block_matmul(q_x_int8, p)
            out_paged[:, r_s : r_s + r_cnt] = acc_i32.to(torch.float32) * ((scale_x / 127.0) * sc)
    lat_paged_compact_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # 3. Ejecución Paginada Directa (Fast)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_paged_fast = torch.empty(batch_size, out_features, dtype=torch.float32)
        for p in pages_fast:
            r_s, r_cnt, sc, acc_i32 = paged_direct_block_matmul(q_x_int8, p)
            out_paged_fast[:, r_s : r_s + r_cnt] = acc_i32.to(torch.float32) * ((scale_x / 127.0) * sc)
    lat_paged_fast_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    # Verificación de coincidencia numérica
    # Nota: la ruta entera tiene una leve diferencia de redondeo de cuantización INT8
    # Comprobamos coincidencia de enteros exacta entre los dos desempaquetados
    max_diff_recon = float((out_paged - out_paged_fast).abs().max().item())
    corr_fp32 = float(torch.corrcoef(torch.stack([out_fp32.flatten(), out_paged.flatten()]))[0, 1].item())

    total_bytes_compact = sum(len(p) for p in pages_compact)
    total_bytes_fast = sum(len(p) for p in pages_fast)
    dense_bytes_fp32 = out_features * in_features * 4

    return {
        "dimensions": f"{out_features}x{in_features}",
        "dense_fp32_bytes": dense_bytes_fp32,
        "paged_compact_bytes": total_bytes_compact,
        "paged_fast_bytes": total_bytes_fast,
        "compression_compact_x": round(dense_bytes_fp32 / total_bytes_compact, 2),
        "compression_fast_x": round(dense_bytes_fp32 / total_bytes_fast, 2),
        "lat_fp32_reconstruction_ms": round(lat_fp32_ms, 3),
        "lat_paged_compact_ms": round(lat_paged_compact_ms, 3),
        "lat_paged_fast_ms": round(lat_paged_fast_ms, 3),
        "speedup_fast_vs_reconstruction": round(lat_fp32_ms / max(1e-6, lat_paged_fast_ms), 2),
        "numerical_exact_agreement": max_diff_recon < 1e-4,
        "correlation_with_fp32": round(corr_fp32, 6),
    }


def verify_integer_accumulators_exact_equality(
    out_features: int = 384,
    in_features: int = 192,
    batch_size: int = 4,
    page_size: int = 4096,
) -> dict[str, Any]:
    """
    Verifica la corrección de implementación estricta (Punto 2.A del Auditor):
    Compara los acumuladores enteros INT32 ANTES de aplicar cualquier escala flotante
    entre la referencia y las rutas empaquetadas (Compact y Fast).
    Demuestra si existe igualdad matemática exacta (diff == 0).
    """
    from trace.serialization.paged_bitcos import PagedBitCOSSerializer

    torch.manual_seed(2026)
    w_ref = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
    q_x = torch.randint(-127, 128, (batch_size, in_features), dtype=torch.int8)

    # 1. Acumuladores de referencia matemática exacta
    acc_ref = torch.matmul(q_x.to(torch.int32), w_ref.to(torch.int32).t())

    # 2. Empaquetar páginas
    rows_per_page = max(1, (page_size - 64) // max(1, int(in_features * 0.25) + 4))
    pages_compact = []
    pages_fast = []
    for r_start in range(0, out_features, rows_per_page):
        r_end = min(out_features, r_start + rows_per_page)
        chunk = w_ref[r_start:r_end, :].numpy()
        p_c = PagedBitCOSSerializer.pack_ternary_page(r_start, out_features, in_features, 1.0, chunk, page_size, "compact")
        p_f = PagedBitCOSSerializer.pack_ternary_page(r_start, out_features, in_features, 1.0, chunk, page_size, "fast")
        pages_compact.append(p_c)
        pages_fast.append(p_f)

    # 3. Acumuladores de ruta Compact
    acc_compact = torch.empty(batch_size, out_features, dtype=torch.int32)
    for p in pages_compact:
        r_s, r_cnt, _, acc_blk = paged_direct_block_matmul(q_x, p)
        acc_compact[:, r_s : r_s + r_cnt] = acc_blk

    # 4. Acumuladores de ruta Fast
    acc_fast = torch.empty(batch_size, out_features, dtype=torch.int32)
    for p in pages_fast:
        r_s, r_cnt, _, acc_blk = paged_direct_block_matmul(q_x, p)
        acc_fast[:, r_s : r_s + r_cnt] = acc_blk

    max_err_compact = int((acc_compact - acc_ref).abs().max().item())
    max_err_fast = int((acc_fast - acc_ref).abs().max().item())

    return {
        "dimensions": f"{out_features}x{in_features}",
        "batch_size": batch_size,
        "max_int_diff_compact_vs_ref": max_err_compact,
        "max_int_diff_fast_vs_ref": max_err_fast,
        "exact_integer_equality_compact": max_err_compact == 0,
        "exact_integer_equality_fast": max_err_fast == 0,
    }


def benchmark_against_resident_fp32_timer(
    out_features: int = 384,
    in_features: int = 192,
    batch_size: int = 1,
    page_size: int = 4096,
    num_threads: int = 2,
) -> dict[str, Any]:
    """
    Compara mediante torch.utils.benchmark.Timer con hilos fijos, calentamiento y réplicas:
    1. Matriz FP32 YA RESIDENTE en memoria RAM (lista en memoria, sin reconstrucción).
    2. Ejecución Paginada Directa en Formato Fast (2-bit LUT).
    3. Ejecución Paginada Directa en Formato Compact.
    """
    from trace.serialization.paged_bitcos import PagedBitCOSSerializer

    import torch.utils.benchmark as benchmark

    torch.set_num_threads(num_threads)
    torch.manual_seed(42)

    w_ref = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
    scale_w = 0.05
    w_resident_fp32 = w_ref.to(torch.float32) * scale_w
    x_fp32 = torch.randn(batch_size, in_features)
    q_x = torch.clamp(torch.round(x_fp32 * (127.0 / 2.0)), -127, 127).to(torch.int8)

    rows_per_page = max(1, (page_size - 64) // max(1, int(in_features * 0.25) + 4))
    pages_fast = []
    pages_compact = []
    for r_start in range(0, out_features, rows_per_page):
        r_end = min(out_features, r_start + rows_per_page)
        chunk = w_ref[r_start:r_end, :].numpy()
        pages_fast.append(PagedBitCOSSerializer.pack_ternary_page(r_start, out_features, in_features, scale_w, chunk, page_size, "fast"))
        pages_compact.append(PagedBitCOSSerializer.pack_ternary_page(r_start, out_features, in_features, scale_w, chunk, page_size, "compact"))

    # Función auxiliar para ejecución rápida
    def run_resident_fp32():
        return torch.matmul(x_fp32, w_resident_fp32.t())

    def run_paged_fast():
        out = torch.empty(batch_size, out_features, dtype=torch.float32)
        for p in pages_fast:
            r_s, r_cnt, sc, acc = paged_direct_block_matmul(q_x, p)
            out[:, r_s : r_s + r_cnt] = acc.to(torch.float32) * sc
        return out

    def run_paged_compact():
        out = torch.empty(batch_size, out_features, dtype=torch.float32)
        for p in pages_compact:
            r_s, r_cnt, sc, acc = paged_direct_block_matmul(q_x, p)
            out[:, r_s : r_s + r_cnt] = acc.to(torch.float32) * sc
        return out

    timer_res = benchmark.Timer(
        stmt="run_resident_fp32()",
        globals={"run_resident_fp32": run_resident_fp32},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=0.2)

    timer_fast = benchmark.Timer(
        stmt="run_paged_fast()",
        globals={"run_paged_fast": run_paged_fast},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=0.2)

    timer_compact = benchmark.Timer(
        stmt="run_paged_compact()",
        globals={"run_paged_compact": run_paged_compact},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=0.2)

    med_fp32_us = timer_res.median * 1e6
    med_fast_us = timer_fast.median * 1e6
    med_compact_us = timer_compact.median * 1e6

    iqr_fp32_us = timer_res.iqr * 1e6
    iqr_fast_us = timer_fast.iqr * 1e6
    iqr_compact_us = timer_compact.iqr * 1e6

    return {
        "threads": num_threads,
        "fp32_resident_median_us": round(med_fp32_us, 2),
        "fp32_resident_iqr_us": round(iqr_fp32_us, 2),
        "paged_fast_median_us": round(med_fast_us, 2),
        "paged_fast_iqr_us": round(iqr_fast_us, 2),
        "paged_compact_median_us": round(med_compact_us, 2),
        "paged_compact_iqr_us": round(iqr_compact_us, 2),
        "ratio_paged_fast_vs_resident_fp32": round(med_fast_us / max(1e-6, med_fp32_us), 2),
        "note": "FP32 residente ya tiene los pesos en memoria RAM consumiendo 4 bytes/peso; Paged BitCOS opera con páginas de 2 bits por peso.",
    }
